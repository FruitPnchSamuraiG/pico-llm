import argparse
import time
import random 
import math
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from datasets import load_dataset
import datasets
import tiktoken 

################################################################################
# 1. Command-line argument parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")

    # NEW: TinyStories subset size control
    parser.add_argument(
        "--tinystories_train_subset_size",
        type=int,
        default=20000,
        help="How many TinyStories train examples to load (default: 20000). Increase for larger training.",
    )

    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
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

    # CPU-friendly training controls
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. Reduce on CPU, e.g., 4.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs. Reduce on CPU, e.g., 1-2.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Optimizer learning rate.")

    # Transformer & model selection flags
    parser.add_argument("--enable_transformer", action="store_true", help="Enable training the Transformer model.")
    parser.add_argument("--enable_kgram", action="store_true", help="Enable training the K-gram MLP model.")
    parser.add_argument("--transformer_heads", type=int, default=2, help="Number of attention heads for Transformer.")
    parser.add_argument("--transformer_blocks", type=int, default=2, help="Number of Transformer blocks.")
    parser.add_argument("--ff_mult", type=int, default=4, help="Feedforward expansion multiplier inside Transformer.")
    parser.add_argument("--no_pos_emb", action="store_true", help="Disable learned positional embeddings (for experimentation).")
    parser.add_argument("--disable_lstm", action="store_true", help="Skip training the LSTM baseline model.")
    parser.add_argument("--save_attention_for_prompt", action="store_true", help="Save Transformer attention head heatmaps for the given --prompt.")
    parser.add_argument("--attention_outdir", type=str, default="attn_plots", help="Directory to save attention heatmaps.")
    parser.add_argument("--norm_type", type=str, default='pre', 
                        help="TransformerBlock defaults to pre-normailzation. Set this to 'post' for post-normalization")
    parser.add_argument("--warmup", type=str, default='', 
                        help="Set to 'yes' or 'no'. If set to either, SGD will be enabled instead of Adam")
    
    # NEW: convenience preset for larger transformer training
    parser.add_argument(
        "--transformer_size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help=(
            "Convenience preset that sets embed_size/heads/blocks/ff_mult. "
            "Overrides individual flags if provided. small=384/4/3/2, medium=512/8/6/4."
        ),
    )

    # Training stability and quality improvements
    # parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. Prevents exploding gradients.")
    # parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization weight decay for AdamW.")
    
    # Validation split
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation (0.1 = 10 percent)")

    # Checkpoint / artifact output directory
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/scratch/kk6081/picollm_extend",
        help="Directory to write checkpoints and loss histories (default: /scratch/kk6081/picollm_extend).",
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Optional path to a .pt state_dict to initialize the Transformer from (for finetuning).",
    )

    # Training / speed knobs
    parser.add_argument(
        "--sample_interval_seconds",
        type=int,
        default=120,
        help="Generate sample text every N seconds during training (default: 120). Increase for faster training.",
    )
    parser.add_argument(
        "--sample_every_steps",
        type=int,
        default=0,
        help="If >0, generate sample text every N training steps (overrides --sample_interval_seconds).",
    )

    # Dynamic LR scheduling
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Learning-rate schedule for faster training: none|cosine|linear (default: cosine).",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=200,
        help="Warmup steps for LR schedule (default: 200).",
    )
    parser.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="Min LR as a fraction of base LR for cosine/linear decay (default: 0.1).",
    )

    args = parser.parse_args()

    # Apply transformer size preset (only affects transformer hyperparams)
    if args.transformer_size == "small":
        args.embed_size = 384
        args.transformer_heads = 4
        args.transformer_blocks = 3
        args.ff_mult = 2
    elif args.transformer_size == "medium":
        args.embed_size = 512
        args.transformer_heads = 8
        args.transformer_blocks = 6
        args.ff_mult = 4

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

    # Initialize with zeros (padding token)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    
    # Copy each sequence into the padded tensor
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq 

    return padded


################################################################################
# 4. K-gram MLP: Fixed-window feedforward baseline
#    Simplest sequence model: predict next token from last k tokens
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        # Can't predict next token with only 1 token
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size) - predictions
    gold = tokens[1:, :]       # (seq_len-1, batch) - targets (shifted left)

    # Reshape for cross_entropy: expects (N, vocab_size) and (N,)
    preds = preds.reshape(-1, vocab_size)  # (batch*(seq_len-1), vocab_size)
    gold = gold.reshape(-1)                # (batch*(seq_len-1),)
    
    # Cross-entropy = -log P(gold_token | predictions)
    # Averaged over all positions
    return F.cross_entropy(preds, gold)

class KGramMLPSeqModel(nn.Module):
    """
    K-gram MLP: Predict next token from fixed k-token history.
    
    ARCHITECTURE:
    1. For each position t, extract last k tokens
    2. Embed each token (instead of one-hot for CPU efficiency)
    3. Concatenate embeddings: (k * embed_size,)
    4. Pass through MLP: Linear -> SiLU -> ... -> Linear(vocab_size)
    5. Output: logits for next token
    
    PARAMETERS:
    - vocab_size: Number of unique tokens (GPT-2 = 50257)
    - k: Context window size (how many previous tokens to consider)
    - embed_size: Dimension of token embeddings
    - num_inner_layers: Depth of MLP (more = more capacity, slower)
    - chunk_size: Micro-batching for efficiency
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # Token embedding layer
        # Why? 50k-dim one-hot = 200KB per token, 128-dim embedding = 512 bytes!
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


    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions for training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_seq):
        """
        Process sequence autoregressively: predict each position from its k-token history.
        
        INPUT: tokens_seq (seq_len, batch_size)
        OUTPUT: (seq_len, batch_size, vocab_size) logits
        
        OPTIMIZED ALGORITHM using unfold (vectorized, much faster):
        1. Transpose to (batch, seq_len) for easier processing
        2. Pad left with zeros to handle positions < k
        3. Use unfold to create sliding window of size k
        4. Embed all k-grams in parallel
        5. Flatten embeddings and pass through MLP
        6. Transpose back to (seq_len, batch, vocab_size)
        
        This replaces the slow nested loop implementation with vectorized operations.
        """
        seq_len, batch_size = tokens_seq.shape
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


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    """
    LSTM: Long Short-Term Memory for sequence modeling.
    
    ARCHITECTURE:
    1. Embedding layer: token_id -> embed_size vector
    2. LSTM: processes sequence, maintains hidden state (h_t, c_t)
    3. Linear projection: hidden -> vocab_size logits
    
    PARAMETERS:
    - vocab_size: Number of tokens
    - embed_size: Embedding dimension
    - hidden_size: LSTM hidden state size (typically = embed_size)
    """
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # batch_first=False means input shape is (seq_len, batch, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)


    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions for training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_seq):
        """
        Autoregressive LSTM forward pass.
        
        INPUT: tokens_seq (seq_len, batch_size)
        OUTPUT: (seq_len, batch_size, vocab_size) logits at each position
        
        STEPS:
        1. Embed tokens: (seq_len, batch) -> (seq_len, batch, embed_size)
        2. LSTM processes sequence: output = (seq_len, batch, hidden_size)
           - At each step t, hidden state h_t depends on h_{t-1} and input
        3. Project to vocabulary: (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed_size)
        self.lstm.flatten_parameters()     # Optimization for contiguous memory
        out, _ = self.lstm(emb)            # (seq_len, batch, hidden_size)
        logits = self.linear(out)          # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 6. Transformer: Attention-based decoder-only model (GPT-style)
################################################################################

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    FORMULA:
    - RMS(x) = sqrt(mean(x^2) + eps)  # Root mean square
    - output = (x / RMS(x)) * weight  # Normalize and scale
    
    PARAMETERS:
    - dim: Feature dimension to normalize (last dimension)
    - eps: Small constant for numerical stability (1e-5 typical)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable per-dimension scale
        
    def forward(self, x):  # x: (..., dim)
        """
        Apply RMSNorm to last dimension.
        
        INPUT: x of shape (..., dim)
        OUTPUT: Normalized x of same shape
        
        STEPS:
        1. Compute RMS: sqrt(mean(x^2) + eps)
        2. Divide x by RMS (normalize variance to 1)
        3. Multiply by learnable weight (allow model to adjust scale)
        """
        # Compute RMS along last dimension, keep dimension for broadcasting
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # 1/RMS
        return x * norm * self.weight  # Normalize and scale


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block with multi-head self-attention and feedforward.
    
    ARCHITECTURE (Pre-norm style):
    1. x = x + Attention(RMSNorm(x))  # Attention with residual
    2. x = x + FFN(RMSNorm(x))         # Feedforward with residual
    
    COMPONENTS:
    - Multi-head self-attention: Learn which tokens to attend to
    - Feedforward network: Process each position independently
    - Residual connections: Enable gradient flow through deep networks
    - RMSNorm: Stabilize activations
    
    PARAMETERS:
    - d_model: Model dimension (embed_size)
    - n_heads: Number of attention heads (must divide d_model evenly)
    - ff_mult: Feedforward expansion factor (typically 4x)
    """
    def __init__(self, d_model, n_heads, ff_mult=4, norm_type='pre'):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert norm_type in ['pre', 'post'], "norm_type must be either 'pre or 'post'"
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_type = norm_type
        self.head_dim = d_model // n_heads  # Each head processes head_dim features
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.out_proj = nn.Linear(d_model, d_model)  # Output projection
        
        # Feedforward network (MLP)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),  # Expand to 4x dimension
            nn.SiLU(),  # Smooth nonlinearity (used in modern LLMs)
            nn.Linear(ff_mult * d_model, d_model)  # Project back to d_model
        )
        
        # Normalization layers (Pre-norm style)
        self.norm_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)
        # Optional: cache attention probabilities for interpretability/plotting
        self.save_attention = False
        self.last_attn_probs = None

    def forward(self, x, causal_mask=None):
        """
        Process sequence through attention and feedforward with residuals.
        
        INPUT: x (batch, seq_len, d_model), causal_mask (1, max_len, max_len)
        OUTPUT: (batch, seq_len, d_model)
        
        MULTI-HEAD ATTENTION ALGORITHM:
        1. Normalize input
        2. Project to Q, K, V and split into n_heads
        3. Compute scaled dot-product attention per head
        4. Apply causal mask (prevent attending to future tokens)
        5. Concatenate heads and project
        6. Add residual connection
        
        FEEDFORWARD:
        1. Normalize attention output
        2. Apply MLP: Linear -> SiLU -> Linear
        3. Add residual connection
        """
        b, t, d = x.shape
        
        # ===== ATTENTION BLOCK =====
        if self.norm_type == 'pre':
            xa = self.norm_attn(x)  # Pre-norm: normalize BEFORE attention
        else:
            xa = x
        
        # Project to Q, K, V and reshape for multi-head attention
        # Shape: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, head_dim)
        q = self.q_proj(xa).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(xa).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(xa).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: Q @ K^T / sqrt(d_k)
        # Shape: (batch, n_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask: prevent position i from attending to position j > i
        if causal_mask is not None:
            # Slice mask to current sequence length (mask is pre-allocated for max_len)
            attn_scores = attn_scores.masked_fill(causal_mask[:, None, :t, :t] == 0, -1e9)
        
        # Softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        if self.save_attention:
            self.last_attn_probs = attn_probs.detach().cpu()
        
        # Apply attention to values
        attn_out = attn_probs @ v  # (batch, heads, seq_len, head_dim)
        
        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.out_proj(attn_out)  # Residual connection

        if self.norm_type=='post':
            x = self.norm_attn(x)  # Post-norm: normalize AFTER attention
        
        # ===== FEEDFORWARD BLOCK =====
        if self.norm_type=='pre':
            xf = self.norm_ff(x)  # Pre-norm: normalize BEFORE feedforward
        else:
            xf = x

        x = x + self.ff(xf)   # Feedforward with residual

        if self.norm_type=='post':
            x = self.norm_ff(x)
        
        return x


class TransformerModel(nn.Module):
    """
    Full decoder-only Transformer (GPT-style) for causal language modeling.
    
    ARCHITECTURE:
    1. Token embedding: vocab_size -> d_model
    2. Positional embedding: position -> d_model (optional)
    3. Stack of TransformerBlocks
    4. Final RMSNorm
    5. Linear projection to vocabulary: d_model -> vocab_size
    
    PARAMETERS:
    - vocab_size: Number of tokens in vocabulary
    - d_model: Model dimension (hidden size)
    - n_heads: Number of attention heads per block
    - n_blocks: Number of stacked Transformer blocks (depth)
    - block_size: Maximum sequence length (for positional embeddings)
    - ff_mult: Feedforward expansion factor (typically 4)
    - use_pos_emb: Whether to use learned positional embeddings
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, 
                 block_size=1024, ff_mult=4, use_pos_emb=True, norm_type='pre'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.use_pos_emb = use_pos_emb
        self.norm_type=norm_type
        
        # Embedding layers
        self.embed = nn.Embedding(vocab_size, d_model)  # Token embeddings
        if use_pos_emb:
            self.pos_emb = nn.Embedding(block_size, d_model)  # Learned positions
        else:
            self.pos_emb = None  # Can disable for RoPE or other schemes
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_mult=ff_mult, norm_type=norm_type) for _ in range(n_blocks)
        ])
        
        # Final normalization and projection
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # Language model head
        
        # Precompute causal mask (lower-triangular matrix)
        # Shape: (1, block_size, block_size), 1 = allowed, 0 = masked
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.uint8))
        self.register_buffer("causal_mask", causal.unsqueeze(0))  # Save as non-trainable buffer
        
        # Initialize weights properly (GPT-2 style)
        self.apply(self._init_weights)
        
        # Scale residual connection weights for stability in deep networks
        for block in self.blocks:
            torch.nn.init.normal_(block.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))


    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions for training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_seq):
        """
        Forward pass for causal language modeling.
        
        INPUT: tokens_seq (seq_len, batch_size) - Note: time-first format from DataLoader
        OUTPUT: (seq_len, batch_size, vocab_size) - Logits for next token at each position
        
        STEPS:
        1. Transpose to (batch, seq_len) for easier processing
        2. Embed tokens and add positional embeddings
        3. Pass through all Transformer blocks (with causal masking)
        4. Final normalization
        5. Project to vocabulary logits
        6. Transpose back to (seq_len, batch, vocab_size) for consistency
        """
        seq_len, batch_size = tokens_seq.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}")
        
        # Convert to (batch, seq_len) for attention computation
        tokens_b = tokens_seq.transpose(0, 1)
        
        # Token embeddings
        x = self.embed(tokens_b)  # (batch, seq_len, d_model)
        
        # Add positional embeddings if enabled
        if self.pos_emb is not None:
            pos = torch.arange(seq_len, device=tokens_seq.device).unsqueeze(0)  # (1, seq_len)
            x = x + self.pos_emb(pos)  # Broadcast and add
        
        # Pass through all Transformer blocks
        for blk in self.blocks:
            x = blk(x, causal_mask=self.causal_mask)  # Apply causal masking
        
        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Transpose back to (seq_len, batch, vocab_size) for consistency with LSTM/K-gram
        return logits.transpose(0, 1)


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, monosemantic_info, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Nucleus (top-p) sampling: Sample from smallest set of tokens with cumulative probability >= p.
    
    ALGORITHM:
    1. Convert logits to probabilities via softmax
    2. Sort probabilities in descending order
    3. Compute cumulative probability mass
    4. Find smallest set of tokens with cumulative mass >= p
    5. Renormalize and sample from this "nucleus"
    
    PARAMETERS:
    - logits: (vocab_size,) raw model outputs
    - p: Cumulative probability threshold [0, 1]
          - p=1.0: sample from full distribution (most diverse)
          - p=0.95: typical default (good balance)
          - p=0.0: deterministic (greedy)
    
    RETURNS:
    - token_id: Integer index of sampled token
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # (vocab_size,)
    
    # Sort in descending order of probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probability mass
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: smallest set with cumulative mass >= p
    cutoff_idx = torch.searchsorted(cumulative, torch.tensor(p, device=logits.device))
    cutoff_idx = torch.clamp(cutoff_idx, min=1)  # Ensure at least 1 token (avoid empty nucleus)

    # Extract nucleus (top tokens within threshold)
    kept_probs = sorted_probs[:cutoff_idx]
    kept_indices = sorted_indices[:cutoff_idx]

    # Renormalize probabilities (they may not sum to exactly 1 after truncation)
    kept_probs = kept_probs / kept_probs.sum()
    
    # Sample from nucleus
    sampled_pos = torch.multinomial(kept_probs, num_samples=1).item()
    return kept_indices[sampled_pos].item() # type: ignore


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    Autoregressive text generation: Unified interface for all models.
    
    ALGORITHM (for each new token):
    1. Encode current text to token sequence
    2. Feed entire sequence through model: (seq_len, 1) -> (seq_len, 1, vocab_size)
    3. Extract logits at last position: logits[-1, 0, :]
    4. Sample next token (greedy or top-p)
    5. Append to sequence and repeat
    
    WHY FEED ENTIRE SEQUENCE?
    - LSTM needs full context to build hidden state
    - Transformer uses causal masking (only attends to previous positions)
    - K-gram only uses last k tokens internally
    - Unified interface simplifies code
    
    OPTIMIZATION OPPORTUNITIES:
    - KV-caching for Transformer (reuse past attention keys/values)
    - Hidden state passing for LSTM (don't recompute from scratch)
    - Currently regenerates entire forward pass each step (simple but slow)
    
    PARAMETERS:
    - model: Neural network (LSTM, Transformer, or K-gram MLP)
    - enc: Tokenizer (tiktoken GPT-2 BPE)
    - init_text: Prompt string to continue from
    - max_new_tokens: How many tokens to generate
    - device: "cpu" or "cuda:0"
    - top_p: None for greedy (argmax), float in (0, 1] for nucleus sampling
    - monosemantic_info: Optional interpretability data (currently unused)
    - do_monosemantic: Whether to run interpretability analysis
    
    RETURNS:
    - final_text: Full generated text (prompt + new tokens)
    - annotated_text: Text with interpretability annotations (if enabled)
    """
    was_training = model.training
    model.eval()  # Disable dropout, set batch norm to eval mode
    
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        # Tokenize initial prompt
        context_tokens = enc.encode(init_text)
        annotation_list = []

        # Generate tokens one at a time (autoregressive)
        for step_i in range(max_new_tokens):
            # Convert token list to tensor: (seq_len, 1)
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            
            # Forward pass: get logits for all positions
            logits_seq = model(seq_tensor)  # (seq_len, 1, vocab_size)
            
            # Extract logits for next token (at last position)
            next_logits = logits_seq[-1, 0, :]  # (vocab_size,)

            # Sample next token
            if top_p is None:
                # Greedy decoding: always pick most likely token
                chosen_token = torch.argmax(next_logits).item()
            else:
                # Nucleus sampling: sample from top-p probability mass
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            # Append to context
            context_tokens.append(chosen_token)

            # Optional: Monosemantic analysis (interpretability stub)
            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    # Restore training mode if it was on
    model.train(was_training)

    # Decode final sequence
    final_text = enc.decode(context_tokens)
    
    # Build annotated text (with interpretability info if available)
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
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    warmup='',
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    grad_clip=1.0,
                    weight_decay=0.01,
                    val_loader=None,
                    checkpoint_dir: str = ".",
                    lr_schedule: str = "none",
                    lr_warmup_steps: int = 200,
                    lr_min_ratio: float = 0.1,
                    sample_every_steps: int = 0):
    """
    Train a single model (LSTM, Transformer, or K-gram MLP) on the provided data.
    
    Args:
        model: The neural network model to train
        loader: DataLoader providing training batches
        epochs: Number of full passes through the dataset
        model_name: String identifier for logging and saving
        device: torch.device for GPU/CPU
        lr: Learning rate for optimizer
        log_steps: Print loss every N steps
        sample_interval: Generate text samples every N seconds
        max_steps_per_epoch: Optional cap on steps per epoch (for quick testing)
        enc: Tokenizer for text generation
        monosemantic_info: Optional interpretability data structure
        prompt: Text prompt for generation during training
        grad_clip: Gradient clipping norm to prevent exploding gradients
        weight_decay: L2 regularization strength (AdamW)
        val_loader: Optional validation DataLoader
        checkpoint_dir: Directory to save checkpoints and loss histories
        lr_schedule: Learning-rate schedule for faster training: none|cosine|linear
        lr_warmup_steps: Warmup steps for LR schedule
        lr_min_ratio: Min LR as a fraction of base LR for cosine/linear decay
        sample_every_steps: If >0, generate sample text every N training steps (overrides --sample_interval_seconds)
    
    Returns:
        (train_loss_history, val_loss_history): Tuples of (global_step, loss) for plotting
    """
    if warmup=='yes':
        optimizer = optim.SGD(model.parameters(),lr=lr)
        warmup_steps = 200

        def linear_warmup(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup)

    elif warmup=='no':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        # Use AdamW optimizer with improved hyperparameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),  # Lower beta2 for better convergence
            weight_decay=weight_decay,
            eps=1e-8
        )

    # ===== LR scheduler (for AdamW runs) =====
    # Note: the existing --warmup flag uses SGD (optionally with warmup). We keep that behavior.
    # For AdamW (default), we add an optional warmup+decay schedule.
    total_steps = None
    if max_steps_per_epoch is not None:
        total_steps = epochs * max_steps_per_epoch
    else:
        total_steps = epochs * len(loader)

    lr_sched = None
    if warmup not in ['yes', 'no'] and lr_schedule != "none":
        def lr_lambda(step: int) -> float:
            # warmup
            if lr_warmup_steps > 0 and step < lr_warmup_steps:
                return float(step) / float(max(1, lr_warmup_steps))

            # decay starts after warmup
            t = max(0, step - lr_warmup_steps)
            T = max(1, total_steps - lr_warmup_steps)
            progress = min(1.0, float(t) / float(T))

            min_lr = max(0.0, min(1.0, lr_min_ratio))

            if lr_schedule == "linear":
                return (1.0 - progress) * (1.0 - min_lr) + min_lr
            # cosine
            return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))

        lr_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Track timing for periodic text generation
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0  # Total steps across all epochs
    
    # Track loss history for plotting
    train_loss_history = []
    val_loss_history = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 3  # Stop if no improvement for 3 epochs
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode (enables dropout, etc.)
        total_loss = 0.0  # Accumulate loss for epoch average
        partial_loss = 0.0  # Accumulate loss for logging intervals
        partial_count = 0  # Count batches in logging interval

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            # Move batch to GPU/CPU: shape is (seq_len, batch_size)
            batch_tokens = batch_tokens.to(device)

            # Forward pass: get logits for next token prediction
            # logits shape: (seq_len, batch_size, vocab_size)
            logits = model(batch_tokens)
            
            # Compute cross-entropy loss with target shift
            loss = compute_next_token_loss(logits, batch_tokens)

            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            if warmup == 'yes':
                scheduler.step()
            elif lr_sched is not None:
                lr_sched.step()

            # Track loss statistics
            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1
            
            # Record loss for plotting
            train_loss_history.append((global_step, loss.item()))

            # Log training progress at regular intervals
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                
                # Calculate gradient norm for monitoring stability
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Loss: {avg_part_loss:.4f}, "
                      f"Grad_norm: {total_grad_norm:.3f}, "
                      f"LR: {current_lr:.2e}")
                # Reset partial counters
                partial_loss = 0.0
                partial_count = 0

            # Generate text samples periodically to monitor quality
            do_sample = False
            if enc is not None:
                if sample_every_steps and (global_step % sample_every_steps == 0):
                    do_sample = True
                else:
                    current_time = time.time()
                    if current_time >= next_sample_time:
                        do_sample = True
            
            if do_sample and enc is not None:
                with torch.no_grad():
                    # Generate using greedy decoding (always pick most likely token)
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=None,  # None = greedy
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    # Generate using top-p=0.95 (nucleus sampling, more diverse)
                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=0.95,  # Keep top 95% probability mass
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # Generate using top-p=1.0 (sample from full distribution, most diverse)
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=1.0,  # Sample from entire distribution
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                # Schedule next sampling time (time-based mode)
                if not sample_every_steps:
                    next_sample_time = time.time() + sample_interval

            # Early stopping if max steps reached (useful for quick testing)
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        # Print epoch summary
        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Train Avg Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            
            print(f"[{model_name}] Evaluating on validation set...")
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    val_logits = model(val_batch)
                    val_loss = compute_next_token_loss(val_logits, val_batch)
                    val_loss_total += val_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            print(f"[{model_name}] *** Validation Loss: {avg_val_loss:.4f} ***")
            
            # Record validation loss (use global_step as x-coordinate)
            val_loss_history.append((global_step, avg_val_loss))
            
            model.train()  # Back to training mode

        # Save model checkpoint after each epoch (allows resuming training)
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved {model_name} weights to {checkpoint_path}")
    
    # Return loss histories for plotting
    return train_loss_history, val_loss_history


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Extract hyperparameters from command-line arguments
    k = args.kgram_k  # Window size for k-gram model
    chunk_size = args.kgram_chunk_size  # Micro-batch size for k-gram

    embed_size = args.embed_size  # Dimension of embedding vectors
    batch_size = args.batch_size  # Number of sequences per batch
    num_epochs = args.num_epochs  # Number of full dataset passes
    learning_rate = args.learning_rate  # Optimizer step size
    warmup = args.warmup # Enable SGD, toggle warmup

    block_size = args.block_size  # Maximum sequence length
    train_subset_size = args.tinystories_train_subset_size
    log_interval_steps = 100  # Print loss every N steps
    sample_interval_seconds = args.sample_interval_seconds
    sample_every_steps = args.sample_every_steps

    max_steps_per_epoch = args.max_steps_per_epoch  # Optional cap on training steps
    num_inner_layers = args.num_inner_mlp_layers  # Depth of k-gram MLP

    # Device selection: prefer CUDA if available, otherwise fall back to CPU
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    checkpoint_dir = args.checkpoint_dir

    ############################################################################
    # Data Loading and Tokenization
    ############################################################################
    tinystories_seqs = []  # Tokenized TinyStories data
    other_seqs = []  # Tokenized custom file data

    # Load TinyStories dataset from HuggingFace if requested
    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        # Use slice notation to load only first N examples (faster than .select())
        dataset = load_dataset("roneneldan/TinyStories", split=f"train")
        dataset = dataset.select(range(train_subset_size)) # type: ignore
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    # Load GPT-2 tokenizer (BPE with ~50k vocab)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    # Tokenize TinyStories data
    if dataset is not None:
        for sample in dataset:
            text = sample['text']  # type: ignore # Extract text field from dataset
            tokens = enc.encode(text)  # Convert text to token IDs
            tokens = tokens[:block_size]  # Truncate to max length
            if len(tokens) > 0:  # Skip empty sequences
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    # Tokenize custom input files if provided
    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()  # Remove whitespace
                if not line:  # Skip empty lines
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    # Create mixed dataset that samples from both sources
    p_tiny = args.tinystories_weight  # Probability of sampling TinyStories
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    
    # Combine all sequences for splitting
    all_sequences = tinystories_seqs + other_seqs
    total_seqs = len(all_sequences)
    
    # Split into train and validation
    val_split = args.val_split
    val_size = int(total_seqs * val_split)
    train_size = total_seqs - val_size
    
    print(f"\n📊 Dataset split: {train_size} train, {val_size} validation ({val_split*100:.1f}%)")
    
    # Shuffle and split
    import random as py_random
    py_random.shuffle(all_sequences)
    train_seqs = all_sequences[:train_size]
    val_seqs = all_sequences[train_size:]
    
    # Create datasets
    # For training, we still want mixed sampling behavior
    train_dataset = MixedSequenceDataset(
        tinystories_seqs=train_seqs if len(tinystories_seqs) > 0 else [],
        other_seqs=[] if len(tinystories_seqs) > 0 else train_seqs,
        p_tiny=1.0 if len(tinystories_seqs) > 0 else 0.0
    )
    
    val_dataset = MixedSequenceDataset(
        tinystories_seqs=val_seqs if len(tinystories_seqs) > 0 else [],
        other_seqs=[] if len(tinystories_seqs) > 0 else val_seqs,
        p_tiny=1.0 if len(tinystories_seqs) > 0 else 0.0
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for better training
        num_workers=0,  # Single-threaded (multi-process can cause issues on some systems)
        collate_fn=seq_collate_fn  # Custom collation for variable-length sequences
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
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

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    models = {}
    if args.enable_kgram:
        models["kgram_mlp_seq"] = kgram_model
    if not args.disable_lstm:
        models["lstm_seq"] = lstm_model
    if args.enable_transformer:
        transformer = TransformerModel(
            vocab_size=vocab_size,
            d_model=embed_size,
            n_heads=args.transformer_heads,
            n_blocks=args.transformer_blocks,
            block_size=block_size,
            ff_mult=args.ff_mult,
            use_pos_emb=not args.no_pos_emb
        ).to(device)

        # Optional: initialize from checkpoint (finetuning)
        if args.init_from:
            import os
            if not os.path.exists(args.init_from):
                raise FileNotFoundError(f"--init_from checkpoint not found: {args.init_from}")
            print(f"Loading Transformer weights from --init_from: {args.init_from}")
            state = torch.load(args.init_from, map_location=device)
            transformer.load_state_dict(state)

        models["transformer"] = transformer

    ############################################################################
    # Train each model
    ############################################################################
    all_loss_histories = {}  # Store loss histories for all models
    
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_history, val_history = train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            warmup=warmup,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            sample_every_steps=sample_every_steps,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            lr_schedule=args.lr_schedule,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
        
        # Store loss histories (both train and val)
        all_loss_histories[model_name] = {
            'train': train_history,
            'val': val_history
        }

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
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

    # Optionally: save Transformer attention maps for the prompt
    if args.save_attention_for_prompt and ("transformer" in models):
        import os
        os.makedirs(args.attention_outdir, exist_ok=True)
        transformer_model = models["transformer"]
        # enable caching
        for blk in transformer_model.blocks:
            blk.save_attention = True # type: ignore
        with torch.no_grad():
            tok = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(1)
            _ = transformer_model(tok)
        # plot
        import matplotlib.pyplot as plt
        pos_flag = 1 if transformer_model.pos_emb is not None else 0
        for bi, blk in enumerate(transformer_model.blocks):
            if getattr(blk, "last_attn_probs", None) is None:
                continue
            attn = blk.last_attn_probs[0]  # (heads, T, T) for batch 0
            heads = attn.shape[0]
            for hi in range(heads):
                plt.figure(figsize=(5, 4))
                plt.imshow(attn[hi].numpy(), aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar()
                plt.title(f"Block {bi} Head {hi} Attention (pos_emb={pos_flag})")
                plt.xlabel("Key positions")
                plt.ylabel("Query positions")
                out_path = os.path.join(
                    args.attention_outdir,
                    f"attn_block{bi}_head{hi}_pos{pos_flag}.png"
                )
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
        # disable caching
        for blk in transformer_model.blocks:
            blk.save_attention = False # type: ignore

    # Save loss histories for plotting
    import pickle
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_path = os.path.join(checkpoint_dir, "loss_histories.pkl")
    with open(loss_path, "wb") as f:
        pickle.dump(all_loss_histories, f)
    print(f"\n✅ Saved loss histories to {loss_path}")
    print("Run 'python plot_losses.py --input <that_file>' to visualize training curves!")
    
    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()