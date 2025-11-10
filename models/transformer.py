"""
Transformer: Attention-based decoder-only model (GPT-style).
State-of-the-art architecture for language modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    MOTIVATION:
    - Simpler than LayerNorm (no mean subtraction, no bias)
    - Used in LLaMA, GPT-NeoX, other modern LLMs
    - Empirically works as well as LayerNorm with less computation
    
    FORMULA:
    - RMS(x) = sqrt(mean(x^2) + eps)  # Root mean square
    - output = (x / RMS(x)) * weight  # Normalize and scale
    
    WHY IT WORKS:
    - Normalizes variance to 1 (stabilizes training)
    - Learnable weight allows model to control scale per dimension
    - Eps prevents division by zero
    
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
    
    WHY PRE-NORM?
    - Normalizes BEFORE attention/FFN (not after)
    - More stable training for deep networks
    - Used in GPT-2, GPT-3, LLaMA
    
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
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Each head processes head_dim features
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model)  # Value projection
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
        xa = self.norm_attn(x)  # Pre-norm: normalize BEFORE attention
        
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
        
        # Apply attention to values
        attn_out = attn_probs @ v  # (batch, heads, seq_len, head_dim)
        
        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.out_proj(attn_out)  # Residual connection
        
        # ===== FEEDFORWARD BLOCK =====
        xf = self.norm_ff(x)  # Pre-norm: normalize BEFORE feedforward
        x = x + self.ff(xf)   # Feedforward with residual
        
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
    
    KEY FEATURES:
    - Causal masking: Position i can only attend to positions <= i
    - Autoregressive generation: Generate one token at a time
    - Parallel training: All positions processed simultaneously
    
    WHY TRANSFORMERS DOMINATE:
    - Parallelizable (unlike RNNs)
    - Unlimited context (in theory, limited by memory in practice)
    - Learns complex relationships via attention
    
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
                 block_size=1024, ff_mult=4, use_pos_emb=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.use_pos_emb = use_pos_emb
        
        # Embedding layers
        self.embed = nn.Embedding(vocab_size, d_model)  # Token embeddings
        if use_pos_emb:
            self.pos_emb = nn.Embedding(block_size, d_model)  # Learned positions
        else:
            self.pos_emb = None  # Can disable for RoPE or other schemes
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_mult=ff_mult) for _ in range(n_blocks)
        ])
        
        # Final normalization and projection
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # Language model head
        
        # Precompute causal mask (lower-triangular matrix)
        # Shape: (1, block_size, block_size), 1 = allowed, 0 = masked
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.uint8))
        self.register_buffer("causal_mask", causal.unsqueeze(0))  # Save as non-trainable buffer

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
        
        CAUSAL PROPERTY:
        - Logits at position i depend only on tokens [0:i]
        - Enables autoregressive generation (predict token i+1 from [0:i])
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
