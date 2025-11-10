"""
Data handling utilities for variable-length sequences with dynamic batching.
"""

import random
import torch


class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    Custom dataset that mixes TinyStories and user-provided text files.
    
    ARCHITECTURE:
    - Stores pre-tokenized sequences (not raw text) for efficiency
    - Each sequence is <= block_size tokens
    - At runtime, randomly samples from TinyStories vs custom data with probability p_tiny
    
    WHY THIS DESIGN?
    - Allows flexible data mixing without reloading
    - Enables domain-specific fine-tuning (e.g., mix fairy tales with code)
    - Random sampling at __getitem__ provides natural data augmentation
    
    PARAMETERS:
    - tinystories_seqs: List of tokenized TinyStories sequences
    - other_seqs: List of tokenized custom file sequences  
    - p_tiny: Probability [0,1] of sampling from TinyStories (vs custom data)
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
    Custom collation for variable-length sequences in a batch.
    
    PROBLEM: DataLoader expects uniform tensor shapes, but our sequences have different lengths
    SOLUTION: Pad all sequences to max_len with zeros (token 0 is typically padding/unknown)
    
    INPUT: batch = [seq1, seq2, ...] where each seq is a 1D LongTensor
    OUTPUT: (max_len, batch_size) padded tensor
    
    WHY (seq_len, batch) NOT (batch, seq_len)?
    - PyTorch RNNs expect (seq_len, batch, features) by default when batch_first=False
    - Consistent with LSTM convention used in LSTMSeqModel
    - Can transpose if needed (Transformer uses batch_first internally)
    """
    max_len = max(len(seq) for seq in batch)  # Find longest sequence
    batch_size = len(batch)

    # Initialize with zeros (padding token)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    
    # Copy each sequence into the padded tensor
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq  # Fill column i with sequence

    return padded
