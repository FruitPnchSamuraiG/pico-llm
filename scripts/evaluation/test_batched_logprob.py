#!/usr/bin/env python3
"""
Test script to verify batched log-prob computation matches sequential.
Run this to ensure optimizations are correct.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Import from dpo_grpo_training
sys.path.insert(0, str(Path(__file__).parent))
from dpo_grpo_training import compute_logprob_and_len, compute_logprob_and_len_batched

def create_dummy_model(vocab_size=100, d_model=64):
    """Create a simple dummy transformer for testing."""
    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, d_model)
            self.linear = torch.nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            # x: (seq_len, batch_size)
            emb = self.embed(x)  # (seq_len, batch_size, d_model)
            logits = self.linear(emb)  # (seq_len, batch_size, vocab_size)
            return logits
    
    return DummyTransformer()

def test_batched_vs_sequential():
    """Test that batched computation matches sequential."""
    print("=" * 80)
    print("Testing Batched Log-Prob Computation")
    print("=" * 80)
    
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = create_dummy_model().to(device)
    model.eval()
    
    # Create test data
    batch_tokens = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24, 25, 26],
    ]
    prompt_lens = [3, 2, 4]
    
    print(f"\nTest setup:")
    print(f"  Batch size: {len(batch_tokens)}")
    print(f"  Sequence lengths: {[len(t) for t in batch_tokens]}")
    print(f"  Prompt lengths: {prompt_lens}")
    
    # Sequential computation
    print("\n1. Sequential computation (OLD)...")
    seq_logps = []
    seq_lens = []
    with torch.no_grad():
        for tokens, prompt_len in zip(batch_tokens, prompt_lens):
            logp, cont_len = compute_logprob_and_len(model, tokens, prompt_len, device)
            seq_logps.append(logp.item())
            seq_lens.append(cont_len.item())
    
    print(f"   Sequential log-probs: {[f'{lp:.4f}' for lp in seq_logps]}")
    print(f"   Continuation lengths: {seq_lens}")
    
    # Batched computation
    print("\n2. Batched computation (NEW)...")
    max_len = max(len(t) for t in batch_tokens)
    with torch.no_grad():
        batch_logps_tensors, batch_lens_tensors = compute_logprob_and_len_batched(
            model, batch_tokens, prompt_lens, device, max_len
        )
    batch_logps = [lp.item() for lp in batch_logps_tensors]
    batch_lens = [cl.item() for cl in batch_lens_tensors]
    
    print(f"   Batched log-probs: {[f'{lp:.4f}' for lp in batch_logps]}")
    print(f"   Continuation lengths: {batch_lens}")
    
    # Compare
    print("\n3. Comparison...")
    max_diff = max(abs(s - b) for s, b in zip(seq_logps, batch_logps))
    print(f"   Max difference: {max_diff:.6e}")
    
    if max_diff < 1e-5:
        print("   ✅ PASS: Batched matches sequential!")
    else:
        print("   ❌ FAIL: Significant difference detected!")
        return False
    
    # Performance test
    print("\n4. Performance comparison...")
    print("   (Run with larger batch for realistic speedup measurement)")
    
    # Create larger batch for timing
    large_batch = [
        [i + j for j in range(50)] for i in range(0, 32 * 20, 20)
    ]
    large_prompts = [10] * len(large_batch)
    max_len_large = max(len(t) for t in large_batch)
    
    import time
    
    # Sequential timing
    start = time.time()
    with torch.no_grad():
        for tokens, prompt_len in zip(large_batch, large_prompts):
            _ = compute_logprob_and_len(model, tokens, prompt_len, device)
    seq_time = time.time() - start
    
    # Batched timing
    start = time.time()
    with torch.no_grad():
        _ = compute_logprob_and_len_batched(
            model, large_batch, large_prompts, device, max_len_large
        )
    batch_time = time.time() - start
    
    speedup = seq_time / batch_time
    print(f"   Sequential time: {seq_time:.3f}s ({len(large_batch)} sequences)")
    print(f"   Batched time: {batch_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x ⚡")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_batched_vs_sequential()
    sys.exit(0 if success else 1)
