"""
LSTM: Recurrent baseline with hidden state.
Learns to compress variable-length context into fixed-size hidden state.
"""

import torch
import torch.nn as nn


class LSTMSeqModel(nn.Module):
    """
    LSTM: Long Short-Term Memory for sequence modeling.
    
    KEY ADVANTAGE OVER K-GRAM:
    - Variable-length context (hidden state summarizes entire history)
    - Parameter sharing across positions (same LSTM cell processes all tokens)
    - Learns what to remember/forget via gates
    
    ARCHITECTURE:
    1. Embedding layer: token_id -> embed_size vector
    2. LSTM: processes sequence, maintains hidden state (h_t, c_t)
    3. Linear projection: hidden -> vocab_size logits
    
    WHY LSTM WORKS:
    - Hidden state h_t = compressed summary of tokens[0:t]
    - Cell state c_t = long-term memory (gradient highway)
    - Gates control information flow (prevent vanishing gradients)
    
    LIMITATIONS:
    - Sequential processing (can't parallelize across time)
    - Struggles with very long sequences (hidden state bottleneck)
    - Modern Transformers outperform on most tasks
    
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
        
        NOTE: We output logits for ALL positions, but only use [:seq_len-1] 
        for loss (since we're predicting next token).
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed_size)
        self.lstm.flatten_parameters()     # Optimization for contiguous memory
        out, _ = self.lstm(emb)            # (seq_len, batch, hidden_size)
        logits = self.linear(out)          # (seq_len, batch, vocab_size)
        return logits
