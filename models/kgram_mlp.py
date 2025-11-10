"""
K-gram MLP: Fixed-window feedforward baseline model.
Simplest sequence model: predict next token from last k tokens.
"""

import torch
import torch.nn as nn


class KGramMLPSeqModel(nn.Module):
    """
    K-gram MLP: Predict next token from fixed k-token history.
    
    ARCHITECTURE:
    1. For each position t, extract last k tokens
    2. Embed each token (instead of one-hot for CPU efficiency)
    3. Concatenate embeddings: (k * embed_size,)
    4. Pass through MLP: Linear -> SiLU -> ... -> Linear(vocab_size)
    5. Output: logits for next token
    
    LIMITATIONS:
    - Fixed context window (can't see beyond k tokens)
    - Processes each position independently (no parameter sharing across positions)
    - Memory/compute scales linearly with sequence length
    
    OPTIMIZATIONS:
    - Uses nn.Embedding instead of one-hot vectors (50k vocab one-hot is huge!)
    - chunk_size: Process multiple timesteps together to reduce loop overhead
    
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

        # CPU-friendly: use embeddings instead of huge one-hot vectors
        # Why? 50k-dim one-hot = 200KB per token, 128-dim embedding = 512 bytes!
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        
        # Build MLP: stack Linear -> SiLU layers
        layers = []
        in_dim = k * embed_size  # Input: concatenated k embeddings
        hidden_dim = embed_size
        for _ in range(max(0, num_inner_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())  # Smooth activation (used in modern LLMs)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, vocab_size))  # Final projection to logits
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        Process sequence autoregressively: predict each position from its k-token history.
        
        INPUT: tokens_seq (seq_len, batch_size)
        OUTPUT: (seq_len, batch_size, vocab_size) logits
        
        ALGORITHM:
        For each position t in sequence:
            1. Extract context: tokens[t-k:t] (pad with 0 if t < k)
            2. Embed context tokens: (k, embed_size)
            3. Flatten: (k * embed_size,)
            4. MLP forward: (vocab_size,)
        
        WARNING: This is slow (nested loops) but correct. For production, use 
        batched convolution or parallel processing.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        # Process in chunks to reduce Python loop overhead
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            
            for t in range(start, end):
                batch_logits = []
                
                # Process each batch item separately (could be vectorized)
                for b in range(batch_size):
                    # Get k-token context (pad left with 0 if needed)
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    # Embedding-based context representation
                    ctx_ids_tensor = torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device)
                    ctx_emb = self.token_embed(ctx_ids_tensor)  # (k, embed_size)
                    context_flat = ctx_emb.flatten().unsqueeze(0)  # (1, k*embed_size)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                    
                # Concatenate batch dimension
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs
