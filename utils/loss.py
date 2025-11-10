"""
Loss computation and training utilities.
"""

import torch
import torch.nn.functional as F


def compute_next_token_loss(logits, tokens):
    """
    Compute cross-entropy loss for next-token prediction.
    
    AUTOREGRESSIVE SETUP:
    - Given tokens [t0, t1, t2, ..., tN]
    - Model predicts: logits[i] = P(t_{i+1} | t0...ti)
    - Loss: cross_entropy(logits[i], t_{i+1}) for all i
    
    IMPLEMENTATION:
    - logits[:-1] are predictions (drop last timestep, no target for it)
    - tokens[1:] are targets (shift left by 1)
    - Flatten to (batch*seq_len, vocab) for efficient cross_entropy
    
    EDGE CASE:
    - If seq_len < 2, no pairs to predict, return 0 loss
    
    PARAMETERS:
    - logits: (seq_len, batch_size, vocab_size) - Model predictions
    - tokens: (seq_len, batch_size) - Ground truth tokens
    
    RETURNS:
    - Scalar tensor: average cross-entropy loss over all (token, target) pairs
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


def nucleus_sampling(logits, p=0.95):
    """
    Nucleus (top-p) sampling: Sample from smallest set of tokens with cumulative probability >= p.
    
    MOTIVATION:
    - Greedy (argmax): Repetitive, boring text
    - Pure random: Incoherent nonsense
    - Top-k: Arbitrary cutoff, doesn't adapt to distribution shape
    - Top-p (nucleus): Adaptive cutoff based on probability mass
    
    ALGORITHM:
    1. Convert logits to probabilities via softmax
    2. Sort probabilities in descending order
    3. Compute cumulative probability mass
    4. Find smallest set of tokens with cumulative mass >= p
    5. Renormalize and sample from this "nucleus"
    
    EXAMPLE (p=0.9):
    - Original probs: [0.5, 0.3, 0.15, 0.04, 0.01]
    - Cumulative: [0.5, 0.8, 0.95, 0.99, 1.0]
    - Nucleus (>= 0.9): first 3 tokens [0.5, 0.3, 0.15]
    - Renormalize: [0.526, 0.316, 0.158]
    - Sample from these 3 tokens
    
    WHY IT WORKS:
    - Adapts to certainty: sharp distribution → few tokens, flat distribution → many tokens
    - Prevents low-probability garbage while allowing diversity
    - Used in GPT-3, ChatGPT, and most modern LLMs
    
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
