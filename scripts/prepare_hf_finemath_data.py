#!/usr/bin/env python3
"""
Download and prepare FineMath-4plus dataset from HuggingFace for base training.
FineMath provides high-quality mathematical reasoning text that's perfect for
training models that will later be finetuned on GSM8K.

Usage:
    python scripts/prepare_hf_finemath_data.py --output_dir data --max_samples 100000
"""

import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace FineMath dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for train/val text files")
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Maximum number of samples to extract (default: 100k)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05 = 5%)")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum text length to include (chars)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum text length to include (chars)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "finemath_train.txt"
    val_file = output_dir / "finemath_val.txt"
    
    print(f"Downloading FineMath-4plus dataset (max {args.max_samples:,} samples)...")
    print(f"Min length: {args.min_length} chars, Max length: {args.max_length} chars")
    
    try:
        from datasets import load_dataset
        
        # Load FineMath-4plus (filtered subset with score >= 4)
        # This is a high-quality mathematical reasoning dataset
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            "HuggingFaceTB/finemath",
            "finemath-4plus",
            split="train",
            streaming=True  # Stream to avoid loading entire dataset
        )
        
        examples = []
        print(f"Processing samples...")
        
        for idx, item in enumerate(dataset):
            if len(examples) >= args.max_samples:
                break
            
            # Extract text content
            text = item.get("text", "").strip()
            
            # Filter by length
            if len(text) < args.min_length or len(text) > args.max_length:
                continue
            
            # Clean up text (remove excessive whitespace)
            text = " ".join(text.split())
            
            examples.append(text)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} items, kept {len(examples):,} examples")
        
        print(f"\n✓ Collected {len(examples):,} math examples")
        
        # Shuffle and split
        random.shuffle(examples)
        val_size = int(len(examples) * args.val_split)
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]
        
        # Write train file
        print(f"Writing {len(train_examples):,} training examples to {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(ex + '\n')
        
        # Write validation file
        print(f"Writing {len(val_examples):,} validation examples to {val_file}")
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(ex + '\n')
        
        print(f"\n✅ FineMath data prepared successfully!")
        print(f"   Train: {train_file} ({len(train_examples):,} examples)")
        print(f"   Val:   {val_file} ({len(val_examples):,} examples)")
        print(f"\nSample example:")
        print(f"  {train_examples[0][:200]}...")
        
    except ImportError:
        print("❌ Error: 'datasets' library not found")
        print("Install with: pip install datasets")
        return 1
    except Exception as e:
        print(f"❌ Error downloading FineMath: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
