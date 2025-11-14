# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.8,
                        help="Probability of sampling from TinyStories if present. Default=0.8. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=2,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=2.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=4,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=4.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    # Training/runtime hyperparameters (previously hard-coded)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. Default=16.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs. Default=3.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Optimizer learning rate. Default=1e-3.")
    parser.add_argument("--train_subset_size", type=int, default=20000,
                        help="Number of TinyStories samples to load when enabled. Default=20000.")
    parser.add_argument("--log_interval_steps", type=int, default=100,
                        help="How often (in steps) to log partial loss. Default=100.")
    parser.add_argument("--sample_interval_seconds", type=int, default=30,
                        help="Interval in seconds to print sample generations during training. Default=30.")

    # Model-specific hyperparameters
    parser.add_argument("--lstm_hidden_size", type=int, default=None,
                        help="Hidden size for LSTM. Default=None meaning same as embed_size.")
    parser.add_argument("--transformer_heads", type=int, default=4,
                        help="Number of attention heads in Transformer. Default=4.")
    parser.add_argument("--transformer_blocks", type=int, default=3,
                        help="Number of Transformer blocks. Default=3.")

    # Optional run tag to append to artifact names
    parser.add_argument("--run_tag", type=str, default="",
                        help="Optional short label to append to saved artifact filenames.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded

def compute_accuracy(logits, targets):
    """
    Computes top-1 accuracy for next-token prediction, ignoring padding (target=0).

    logits: (seq_len, batch, vocab_size)
    targets: (seq_len, batch)
    
    We shift to align with next-token prediction:
    - logits[:-1] predicts targets[1:]
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return 0.0  # Need at least 2 positions for next-token prediction
    
    # 1. Shift for next-token prediction (same as in compute_next_token_loss)
    preds_logits = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = targets[1:, :]              # (seq_len-1, batch)
    
    # 2. Get the model's top prediction (token ID) for each position
    predictions = torch.argmax(preds_logits, dim=-1)  # (seq_len-1, batch)

    # 3. Create a mask to ignore padding tokens (where target ID is 0)
    mask = (gold != 0)

    # 4. Compare predictions to targets, only where not padding
    correct = (predictions == gold) & mask

    # 5. Calculate the mean accuracy
    total_non_padded = mask.sum()
    if total_non_padded == 0:
        return 0.0  # Avoid division by zero

    accuracy = correct.sum().float() / total_non_padded
    return accuracy.item()


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # Token embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # Initialize the layers of the MLP
        layers = []
        # Input dimension is k * embed_size (embeddings for k tokens)
        input_dim = self.k * self.embed_size

        # Add the (Linear -> SiLU) inner blocks
        # num_inner_layers=1 means: 1 (Linear->SiLU) block + 1 output Linear
        for _ in range(self.num_inner_layers):
            layers.append(nn.Linear(input_dim, self.embed_size))
            layers.append(nn.SiLU())
            input_dim = self.embed_size

        # Add the final output layer to get logits
        layers.append(nn.Linear(input_dim, self.vocab_size))

        # Set the current network as a sequential model of the above layers
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        # Using Embedding Layer
        device = tokens_seq.device

        # Transpose to (batch, seq_len) for easier processing
        tokens = tokens_seq.transpose(0, 1)  # (batch, seq_len)

        # Add padding of zeros at the beginning for positions < k
        padded_tokens = F.pad(tokens, (self.k - 1, 0), value=0)  # (batch, seq_len + k - 1)

        # Create a sliding window view of size k
        unfolded_tokens = padded_tokens.unfold(dimension=1, size=self.k, step=1)  # (batch, seq_len, k)

        # Create embeddings for each token in the k-grams
        x = self.embedding(unfolded_tokens)  # (batch, seq_len, k, embed_size)

        # Flatten the last two dimensions to create a single input vector for the MLP
        x = x.flatten(start_dim=2)  # (batch, seq_len, k * embed_size)

        # Pass through the MLP to get logits
        logits = self.net(x)  # (batch, seq_len, vocab_size)

        # Transpose back to (seq_len, batch, vocab_size)
        return logits.transpose(0, 1)  # (seq_len, batch, vocab_size)
        ''' Old Forward with one-hot (very memory heavy)
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs
        '''


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        # Calculate the Root Mean Square of the input
        # x.pow(2).mean(-1, keepdim=True) is the Mean Square
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize x and apply scale
        return (x * norm) * self.scale

class Block(nn.Module):
    """ A single Transformer block, as described in the blueprint. """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Each head processes head_dim features
        # Causal Self-Attention
        self.norm1 = RMSNorm(d_model)
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.out_proj = nn.Linear(d_model, d_model)  # Output projection

        # Feedforward network (MLP)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expand to 4x dimension
            nn.SiLU(),  # Smooth nonlinearity (used in modern LLMs)
            nn.Linear(4 * d_model, d_model)  # Project back to d_model
        )

        # Feed-Forward Netowrk (MLP)
        self.norm2 = RMSNorm(d_model)


    def forward(self, x, causal_mask):
        """
        x: (seq_len, batch, d_model)
        causal_mask: (seq_len, seq_len)
        return: (seq_len, batch, d_model)
        """
        b, t, d = x.size()
        # Attention with skip connection
        norm_x = self.norm1(x)

        # Project to Q, K, V and reshape for multi-head attention
        # Shape: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, head_dim)
        q = self.q_proj(norm_x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention: Q @ K^T / sqrt(d_k)
        # Shape: (batch, n_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: prevent position i from attending to position j > i
        if causal_mask is not None:
            # Slice mask to current sequence length (mask is pre-allocated for max_len)
            attn_scores = attn_scores.masked_fill(causal_mask[:, None, :t, :t] == 0, -1e9)
            # Softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq_len, seq_len)

        # Apply attention to values
        attn_out = attn_probs @ v  # (batch, heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.out_proj(attn_out)  # Residual connection

        # ===== FEEDFORWARD BLOCK =====
        xf = self.norm2(x)  # Pre-norm: normalize BEFORE feedforward
        x = x + self.ff(xf)   # Feedforward with residual
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=4, n_blocks=3, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional embedding layer (index by position 0..max_seq_len-1)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads) for _ in range(n_blocks)
        ])

        # Final RMSNorm
        self.final_norm = RMSNorm(d_model)

        # Unembedding layer to produce logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Create boolean causal mask: True means "mask out" (disallow attention)
        # Upper-triangular (k=1) are future positions that should be masked.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.uint8))
        self.register_buffer("causal_mask", causal_mask.unsqueeze(0))

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
        # Convert to (batch, seq_len) for attention computation
        tokens_b = tokens_seq.transpose(0, 1)

        # Get token embeddings
        # Keep shape as (seq_len, batch, d_model) to match MultiheadAttention with batch_first=False
        x = self.token_embedding(tokens_b)  # (seq_len, batch, d_model)

        # Add positional embeddings
        # positions: (seq_len,) => pos_emb: (seq_len, d_model) => expand to (seq_len, batch, d_model)
        # positions = torch.arange(seq_len, device=device).unsqueeze(0) # (1, seq_len)
        # x = x + self.pos_emb(positions)  # (seq_len, batch, d_model)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask=self.causal_mask)

        # Final normalization and LM head to get logits
        x = self.final_norm(x)                 # (seq_len, batch, d_model)
        logits = self.lm_head(x)               # (seq_len, batch, vocab_size)
        return logits.transpose(0, 1)  # (seq_len, batch, vocab_size)

################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    if p >= 1.0:
        # If p=1.0, just sample from the full distribution
        probs = F.softmax(logits, dim=-1)
    else:
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # Find cumulative sum of the probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 4. Find the smallest set of tokens whose prob sum is >= p
        # We create a mask for tokens to remove.
        # Find indices where cumulative prob is > p
        sorted_indices_to_remove = cumulative_probs > p
        # We must keep at least one token. We shift the mask to the right
        # so that the first token that crosses the threshold p
        # is also removed.
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        # But we always keep the top token (index 0)
        sorted_indices_to_remove[0] = False
        # Create mask for original logits tensor
        # We scatter the sorted_indices_to_remove back to original indices
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        # Remove the noise by setting their probabilities to zero
        probs[indices_to_remove] = 0.0
        # Renormalize the probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)
    # Sample from the filtered distribution
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    val_loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    We also add `val_loader` for validation loss computation.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)
            acc = compute_accuracy(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1
            train_losses.append(loss.item())
            train_accuracies.append(acc)

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f} "
                      f"Acc: {acc*100:.2f}%")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        
        # Validation at end of epoch
        model.eval()
        val_total_loss = 0.0
        val_total_acc = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for val_batch_tokens in val_loader:
                val_batch_tokens = val_batch_tokens.to(device)
                val_logits = model(val_batch_tokens)
                val_loss = compute_next_token_loss(val_logits, val_batch_tokens)
                val_acc = compute_accuracy(val_logits, val_batch_tokens)
                val_total_loss += val_loss.item()
                val_total_acc += val_acc
                val_batch_count += 1
        
        if val_batch_count > 0:
            avg_val_loss = val_total_loss / val_batch_count
            avg_val_acc = val_total_acc / val_batch_count
        else:
            avg_val_loss = 0.0
            avg_val_acc = 0.0
        
        val_losses.append((global_step, avg_val_loss))
        val_accuracies.append((global_step, avg_val_acc))
        
        model.train()
        
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc*100:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()
    # Track total execution time of the run for inclusion in artifact names
    overall_start_time = time.time()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    block_size = args.block_size
    train_subset_size = args.train_subset_size
    log_interval_steps = args.log_interval_steps
    sample_interval_seconds = args.sample_interval_seconds

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # Split dataset into train and validation (90/10 split)
    total_size = len(combined_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_hidden = embed_size if args.lstm_hidden_size is None else args.lstm_hidden_size
    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=lstm_hidden
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=args.transformer_heads,
        n_blocks=args.transformer_blocks,
        max_seq_len=block_size,
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "kvcache_transformer": transformer,
    }

    all_model_train_losses = {}
    all_model_train_accuracies = {}
    all_model_val_losses = {}
    all_model_val_accuracies = {}
    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_losses, train_accs, val_losses, val_accs = train_one_model(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )
        all_model_train_losses[model_name] = train_losses
        all_model_train_accuracies[model_name] = train_accs
        all_model_val_losses[model_name] = val_losses
        all_model_val_accuracies[model_name] = val_accs

        # Save the trained model
        checkpoint_path = f"{model_name}_checkpoint.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'vocab_size': vocab_size,
            'embed_size': embed_size,
            'model_config': {
                'kgram_k': k if model_name == 'kgram_mlp_seq' else None,
                'num_inner_layers': num_inner_layers if model_name == 'kgram_mlp_seq' else None,
                'hidden_size': lstm_hidden if model_name == 'lstm_seq' else None,
                'transformer_heads': args.transformer_heads if model_name == 'kvcache_transformer' else None,
                'transformer_blocks': args.transformer_blocks if model_name == 'kvcache_transformer' else None,
            },
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"[{model_name}] Saved checkpoint to {checkpoint_path}")

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")
    # --- PLOT 1: LOSS ---
    plt.figure(figsize=(12, 6))
    for model_name in models.keys():
        # Plot smoothed training loss
        train_losses = all_model_train_losses[model_name]
        if len(train_losses) > 100:
            window = 50
            # Use 'replicate' padding mode to avoid edge artifacts from zero-padding
            losses_tensor = torch.tensor(train_losses).view(1, 1, -1)
            # Manually replicate edges to avoid the zero-padding artifact
            pad_size = window // 2
            padded = F.pad(losses_tensor, (pad_size, pad_size), mode='replicate')
            smooth_losses = torch.nn.functional.conv1d(
                padded,
                torch.ones(1, 1, window) / window,
                padding=0  # No padding since we manually padded
            ).view(-1).tolist()
            x = list(range(1, len(smooth_losses) + 1))
            plt.plot(x, smooth_losses, label=f"{model_name} Train Loss (smoothed)", alpha=0.7, linestyle='--')
        else:
            x = list(range(1, len(train_losses) + 1))
            plt.plot(x, train_losses, label=f"{model_name} Train Loss", alpha=0.7, linestyle='--')

        # Plot validation loss (stepped)
        val_losses = all_model_val_losses[model_name]
        if val_losses:
            steps, losses = zip(*val_losses)
            plt.plot(steps, losses, label=f"{model_name} Val Loss", marker='o', markersize=4)

    plt.title("Model Training & Validation Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_device = str(device).replace(":", "-")
    # Compute elapsed runtime
    elapsed_seconds = int(time.time() - overall_start_time)
    h, rem = divmod(elapsed_seconds, 3600)
    m, s = divmod(rem, 60)
    time_tag = f"{h:02d}h{m:02d}m{s:02d}s"
    tag_suffix = f"_tag{args.run_tag}" if args.run_tag else ""
    config_tag = (
        f"bs{batch_size}_ep{num_epochs}_lr{learning_rate}_"
        f"blk{block_size}_emb{embed_size}_k{k}_layers{num_inner_layers}_"
        f"chunk{chunk_size}_tiny{args.tinystories_weight}_subset{train_subset_size}_"
        f"lstmH{lstm_hidden}_heads{args.transformer_heads}_blocks{args.transformer_blocks}_"
        f"dev{safe_device}_t{time_tag}{tag_suffix}"
    )
    save_path_loss = f"model_loss_comparison_{config_tag}_{timestamp}.png"
    plt.savefig(save_path_loss)
    print(f"Saved loss comparison plot to {save_path_loss}")

    # --- PLOT 2: ACCURACY ---
    plt.figure(figsize=(12, 6))
    for model_name in models.keys():
        # Plot smoothed training accuracy
        train_accs = all_model_train_accuracies[model_name]
        if len(train_accs) > 100:
            window = 50
            # Use 'replicate' padding mode to avoid edge artifacts from zero-padding
            accs_tensor = torch.tensor(train_accs).view(1, 1, -1)
            pad_size = window // 2
            padded = F.pad(accs_tensor, (pad_size, pad_size), mode='replicate')
            smooth_accs = torch.nn.functional.conv1d(
                padded,
                torch.ones(1, 1, window) / window,
                padding=0  # No padding since we manually padded
            ).view(-1).tolist()
            x = list(range(1, len(smooth_accs) + 1))
            plt.plot(x, smooth_accs, label=f"{model_name} Train Acc (smoothed)", alpha=0.7, linestyle='--')
        else:
            x = list(range(1, len(train_accs) + 1))
            plt.plot(x, train_accs, label=f"{model_name} Train Acc", alpha=0.7, linestyle='--')

        # Plot validation accuracy (stepped)
        val_accs = all_model_val_accuracies[model_name]
        if val_accs:
            steps, accs = zip(*val_accs)
            plt.plot(steps, accs, label=f"{model_name} Val Acc", marker='o', markersize=4)

    plt.title("Model Training & Validation Accuracy")
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy (Top-1)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0) # Accuracy is between 0 and 1

    save_path_acc = f"model_accuracy_comparison_{config_tag}_{timestamp}.png"
    plt.savefig(save_path_acc)
    print(f"Saved accuracy comparison plot to {save_path_acc}")

    # plt.show()


if __name__ == "__main__":
    main()
