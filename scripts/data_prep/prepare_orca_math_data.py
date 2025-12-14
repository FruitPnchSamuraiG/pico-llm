#!/usr/bin/env python3
"""
Download and prepare Orca-Math-Word-Problems-200k dataset from HuggingFace.
This dataset contains 200k high-quality math word problems with solutions.
Perfect for training models that will be finetuned on GSM8K.

Usage:
    python scripts/prepare_orca_math_data.py --output_dir data --max_samples 100000
"""

import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download Orca-Math-Word-Problems dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for train/val text files")
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Maximum number of samples (default: 100k)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05 = 5%)")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum text length to include (chars)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum text length to include (chars)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "orca_math_train.txt"
    val_file = output_dir / "orca_math_val.txt"
    
    print(f"📥 Downloading Orca-Math-Word-Problems (max {args.max_samples:,} samples)...")
    print(f"   Min length: {args.min_length} chars")
    print(f"   Max length: {args.max_length} chars")
    print()
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            "microsoft/orca-math-word-problems-200k",
            split="train"
        )
        
        print(f"✓ Loaded {len(dataset):,} examples from Orca-Math")
        print()
        
        examples = []
        skipped_too_short = 0
        skipped_too_long = 0
        
        print("Processing and filtering examples...")
        for idx, item in enumerate(dataset):
            if len(examples) >= args.max_samples:
                break
            
            # Extract question and answer
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            
            if not question or not answer:
                continue
            
            # Format: Q: ... A: ... (single line for pico-llm)
            text = f"Q: {question} A: {answer}"
            
            # Filter by length
            if len(text) < args.min_length:
                skipped_too_short += 1
                continue
            if len(text) > args.max_length:
                skipped_too_long += 1
                continue
            
            # Clean up excessive whitespace
            text = " ".join(text.split())
            
            examples.append(text)
            
            if (idx + 1) % 20000 == 0:
                print(f"  Processed {idx + 1:,} items, kept {len(examples):,} examples")
        
        print()
        print(f"✓ Collected {len(examples):,} math word problems")
        print(f"  Skipped {skipped_too_short:,} too short (< {args.min_length} chars)")
        print(f"  Skipped {skipped_too_long:,} too long (> {args.max_length} chars)")
        print()
        
        # Calculate average length
        avg_len = sum(len(ex) for ex in examples) / len(examples) if examples else 0
        print(f"📊 Average example length: {avg_len:.0f} chars")
        
        # Shuffle and split
        random.shuffle(examples)
        val_size = int(len(examples) * args.val_split)
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]
        
        # Write train file
        print(f"\n📝 Writing {len(train_examples):,} training examples to {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(ex + '\n')
        
        # Write validation file
        print(f"📝 Writing {len(val_examples):,} validation examples to {val_file}")
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(ex + '\n')
        
        print()
        print("=" * 70)
        print("✅ Orca-Math data prepared successfully!")
        print("=" * 70)
        print(f"   Train: {train_file} ({len(train_examples):,} examples)")
        print(f"   Val:   {val_file} ({len(val_examples):,} examples)")
        print()
        print("📝 Sample examples:")
        for i, ex in enumerate(train_examples[:3], 1):
            print(f"\n   Example {i}:")
            preview = ex[:150] + "..." if len(ex) > 150 else ex
            print(f"   {preview}")
        print()
        
    except ImportError:
        print("❌ Error: 'datasets' library not found")
        print("Install with: pip install datasets")
        return 1
    except Exception as e:
        print(f"❌ Error downloading Orca-Math: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
