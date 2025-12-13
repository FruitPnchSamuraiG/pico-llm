#!/usr/bin/env python3
"""
Analyze FineMath data quality and identify issues
"""
import sys
from pathlib import Path

def analyze_data(filepath, max_lines=1000):
    print(f"\n📊 Analyzing: {filepath}")
    print("=" * 60)
    
    lengths = []
    has_latex = 0
    has_long_words = 0
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            lengths.append(len(line))
            
            # Check for LaTeX
            if '\\' in line or '$' in line:
                has_latex += 1
            
            # Check for very long words (technical jargon)
            words = line.split()
            if any(len(w) > 20 for w in words):
                has_long_words += 1
            
            # Collect samples
            if i < 5:
                samples.append(line[:200] + "..." if len(line) > 200 else line)
    
    if not lengths:
        print("❌ No data found!")
        return
    
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    
    print(f"📈 Statistics ({len(lengths)} examples analyzed):")
    print(f"   Average length: {avg_len:.0f} chars")
    print(f"   Min length:     {min_len} chars")
    print(f"   Max length:     {max_len} chars")
    print(f"   LaTeX content:  {has_latex}/{len(lengths)} ({100*has_latex/len(lengths):.1f}%)")
    print(f"   Long words:     {has_long_words}/{len(lengths)} ({100*has_long_words/len(lengths):.1f}%)")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if avg_len > 800:
        print(f"   ❌ TOO COMPLEX: Avg {avg_len:.0f} chars (should be < 500)")
        print(f"      → Model can't learn patterns from graduate-level problems")
    elif avg_len > 500:
        print(f"   ⚠️  MODERATELY COMPLEX: Avg {avg_len:.0f} chars")
        print(f"      → May cause slow learning")
    else:
        print(f"   ✅ GOOD COMPLEXITY: Avg {avg_len:.0f} chars")
    
    if has_latex / len(lengths) > 0.5:
        print(f"   ⚠️  High LaTeX content ({100*has_latex/len(lengths):.1f}%)")
        print(f"      → May confuse tokenizer")
    
    print(f"\n📝 Sample examples:")
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n   Example {i}:")
        print(f"   {sample}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if avg_len > 600:
        print(f"   1. Regenerate data with max_length=512")
        print(f"      bash scripts/regenerate_finemath_simple.sh")
        print(f"   2. Restart training with:")
        print(f"      bash scripts/train_transformer_finemath_v2.sh")
    else:
        print(f"   ✅ Data quality looks good!")
        print(f"   If training stagnates, try:")
        print(f"   - Higher LR: LR=3e-4 (vs current 2e-4)")
        print(f"   - More epochs: EPOCHS=8")
        print(f"   - Higher min LR: LR_MIN_RATIO=0.2")

if __name__ == "__main__":
    data_dir = Path("data")
    
    train_file = data_dir / "finemath_train.txt"
    val_file = data_dir / "finemath_val.txt"
    
    if not train_file.exists():
        print("❌ Error: finemath_train.txt not found!")
        print("Run: bash scripts/train_transformer_finemath.sh")
        sys.exit(1)
    
    analyze_data(train_file, max_lines=1000)
    
    if val_file.exists():
        analyze_data(val_file, max_lines=200)
