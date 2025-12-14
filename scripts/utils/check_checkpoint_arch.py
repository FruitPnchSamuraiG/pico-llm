#!/usr/bin/env python3
"""
Quick utility to inspect a transformer checkpoint's architecture.

Usage:
    python scripts/check_checkpoint_arch.py <checkpoint.pt>
    python scripts/check_checkpoint_arch.py /scratch/kk6081/picollm_extend/transformer_epoch1.pt
"""

import sys
import torch
from pathlib import Path


def check_checkpoint(ckpt_path: str) -> None:
    """Inspect and print checkpoint architecture."""
    path = Path(ckpt_path)
    if not path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"📦 Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Extract architecture parameters
    try:
        embed_size = ckpt['embed.weight'].shape[1]
        vocab_size = ckpt['embed.weight'].shape[0]
        block_size = ckpt['pos_emb.weight'].shape[0] if 'pos_emb.weight' in ckpt else 'N/A'
        
        # Count blocks
        n_blocks = max([int(k.split('.')[1]) for k in ckpt.keys() if k.startswith('blocks.')]) + 1
        
        # Calculate ff_mult
        ff_size = ckpt['blocks.0.ff.0.weight'].shape[0]
        ff_mult = ff_size // embed_size
        
        # NOTE: Cannot reliably detect n_heads from checkpoint structure
        # In this architecture, q_proj maps d_model -> d_model (not d_model -> d_model * n_heads)
        # The heads dimension is internal to the TransformerBlock forward pass
        n_heads = "UNKNOWN"  # Cannot be reliably detected
        
        # Determine size preset (based on embed/blocks/ff_mult only)
        if embed_size == 384 and n_blocks == 3 and ff_mult == 2:
            size_preset = "small (typical: 4 heads)"
        elif embed_size == 512 and n_blocks == 6 and ff_mult == 4:
            size_preset = "medium (typical: 8 heads)"
        else:
            size_preset = "custom"
        
        print("\n" + "="*60)
        print("✅ Checkpoint Architecture")
        print("="*60)
        print(f"  Preset:        {size_preset.upper()}")
        print(f"  embed_size:    {embed_size}")
        print(f"  n_heads:       {n_heads} ⚠️  (must match training config)")
        print(f"  n_blocks:      {n_blocks}")
        print(f"  ff_mult:       {ff_mult} (ff_size={ff_size})")
        print(f"  block_size:    {block_size}")
        print(f"  vocab_size:    {vocab_size}")
        print("="*60)
        print("\n⚠️  Note: n_heads cannot be detected from checkpoint weights!")
        print("    You must use the same n_heads value that was used during training.")
        print("    Common values: 4 (small), 8 (medium)")
        print("="*60)
        
        # Print usage examples
        print("\n📋 To use this checkpoint for finetuning:")
        # Suggest typical head counts based on preset
        if embed_size == 384:
            suggested_heads = 4
        elif embed_size == 512:
            suggested_heads = 8
        else:
            suggested_heads = "4_or_8"
        
        print(f"\n  Option 1: Use environment variables")
        print(f"    EMBED={embed_size} HEADS={suggested_heads} BLOCKS={n_blocks} FF_MULT={ff_mult} \\")
        print(f"      bash scripts/train_transformer_gsm8k.sh")
        
        print(f"\n  Option 2: Modify training script defaults")
        print(f"    # In train_transformer_gsm8k.sh or train_transformer_reasoning.sh:")
        print(f"    EMBED=${{EMBED:-{embed_size}}}")
        print(f"    HEADS=${{HEADS:-{suggested_heads}}}  # ⚠️  Must match training!")
        print(f"    BLOCKS=${{BLOCKS:-{n_blocks}}}")
        print(f"    FF_MULT=${{FF_MULT:-{ff_mult}}}")
        
        print(f"\n  Option 3: Use interpret_transformer.py")
        print(f"    python scripts/interpret_transformer.py \\")
        print(f"      --checkpoint {ckpt_path} \\")
        print(f"      --embed_size {embed_size} --transformer_heads {suggested_heads} \\")
        print(f"      --transformer_blocks {n_blocks} --ff_mult {ff_mult} \\")
        print(f"      --analysis attention,logit_lens,neurons \\")
        print(f"      --out_dir /scratch/kk6081/picollm_extend/interpretability_test")
        
        print(f"\n  ⚠️  Remember: If your checkpoint was trained with a different n_heads value,")
        print(f"     you MUST use that same value here!")
        
        print("\n" + "="*60)
        
    except KeyError as e:
        print(f"❌ Missing key in checkpoint: {e}")
        print("   This may not be a valid Transformer checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing checkpoint: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_checkpoint_arch.py <checkpoint.pt>")
        print("\nExample:")
        print("  python scripts/check_checkpoint_arch.py /scratch/kk6081/picollm_extend/transformer_epoch1.pt")
        sys.exit(1)
    
    check_checkpoint(sys.argv[1])


if __name__ == '__main__':
    main()
