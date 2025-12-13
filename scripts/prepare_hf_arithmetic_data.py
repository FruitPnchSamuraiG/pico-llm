#!/usr/bin/env python3
"""
Download and prepare arithmetic reasoning datasets from HuggingFace.
Uses multiple datasets for curriculum learning:
1. Simple arithmetic operations (generated)
2. Elementary math problems (from HF datasets)
3. GSM8K-style reasoning (bridges to full GSM8K)
"""
import argparse
from pathlib import Path
from datasets import load_dataset
import random


def format_gsm8k_style(question, answer_text, final_answer):
    """Format in GSM8K style with chain-of-thought."""
    return f"Q: {question}\nA: {answer_text}\n#### {final_answer}"


def prepare_aqua_rat_subset(output_dir, max_samples=3000):
    """
    AQuA-RAT: Algebra Question Answering with Rationales
    Multiple choice algebra problems with reasoning steps.
    """
    print("📥 Downloading AQuA-RAT dataset...")
    try:
        dataset = load_dataset("deepmind/aqua_rat", split="train")
        
        train_problems = []
        for item in dataset:
            if len(train_problems) >= max_samples:
                break
            
            # Extract question and reasoning
            question = item.get('question', '').strip()
            rationale = item.get('rationale', '').strip()
            correct = item.get('correct', '').strip()
            
            if question and rationale and correct:
                # Format as GSM8K style
                formatted = f"Q: {question}\nA: {rationale}\n#### {correct}"
                train_problems.append(formatted)
        
        # Split train/val
        random.shuffle(train_problems)
        split_idx = int(len(train_problems) * 0.9)
        train_data = train_problems[:split_idx]
        val_data = train_problems[split_idx:]
        
        return train_data, val_data, "aqua_rat"
    
    except Exception as e:
        print(f"⚠️  Could not load AQuA-RAT: {e}")
        return [], [], "aqua_rat"


def prepare_math_qa(output_dir, max_samples=5000):
    """
    MathQA: Math word problems with step-by-step solutions.
    Good for curriculum learning before GSM8K.
    """
    print("📥 Downloading MathQA dataset...")
    try:
        dataset = load_dataset("math_qa", split="train")
        
        train_problems = []
        for item in dataset:
            if len(train_problems) >= max_samples:
                break
            
            question = item.get('Problem', '').strip()
            rationale = item.get('Rationale', '').strip()
            answer = item.get('correct', '').strip()
            
            if question and rationale and answer:
                # Clean up rationale (MathQA has some formatting issues)
                rationale = rationale.replace('\\n', ' ').strip()
                formatted = f"Q: {question}\nA: {rationale}\n#### {answer}"
                train_problems.append(formatted)
        
        # Split train/val
        random.shuffle(train_problems)
        split_idx = int(len(train_problems) * 0.9)
        train_data = train_problems[:split_idx]
        val_data = train_problems[split_idx:]
        
        return train_data, val_data, "math_qa"
    
    except Exception as e:
        print(f"⚠️  Could not load MathQA: {e}")
        return [], [], "math_qa"


def prepare_asdiv(output_dir, max_samples=2000):
    """
    ASDiv: Academia Sinica Diverse MWP Dataset
    Elementary school math word problems (easier than GSM8K).
    Perfect for curriculum learning!
    """
    print("📥 Downloading ASDiv dataset...")
    try:
        # Try multiple possible dataset names/configs
        dataset = None
        possible_names = [
            ("EleutherAI/asdiv", None),
            ("MU-NLPC/Calc-asdiv_a", None),
            ("asdiv", None),
        ]
        
        for name, config in possible_names:
            try:
                print(f"   Trying: {name}")
                if config:
                    dataset = load_dataset(name, config, split="train")
                else:
                    dataset = load_dataset(name, split="train")
                print(f"   ✅ Loaded: {name}")
                break
            except:
                continue
        
        if dataset is None:
            raise Exception("Could not find ASDiv dataset")
        
        train_problems = []
        for item in dataset:
            if len(train_problems) >= max_samples:
                break
            
            # Try different field names
            question = item.get('body', item.get('question', item.get('problem', ''))).strip()
            answer = str(item.get('answer', item.get('solution', ''))).strip()
            
            if question and answer:
                # ASDiv doesn't have rationales, so we create simple ones
                formatted = f"Q: {question}\nA: The answer is {answer}.\n#### {answer}"
                train_problems.append(formatted)
        
        if not train_problems:
            raise Exception("No problems extracted from dataset")
        
        # Split train/val
        random.shuffle(train_problems)
        split_idx = int(len(train_problems) * 0.9)
        train_data = train_problems[:split_idx]
        val_data = train_problems[split_idx:]
        
        return train_data, val_data, "asdiv"
    
    except Exception as e:
        print(f"⚠️  Could not load ASDiv: {e}")
        return [], [], "asdiv"


def generate_simple_arithmetic(num_samples=5000):
    """
    Generate very simple arithmetic problems (fallback if HF datasets fail).
    Addition, subtraction, multiplication.
    """
    print("🔢 Generating simple arithmetic problems...")
    
    problems = []
    templates = {
        'addition': lambda a, b: (f"What is {a} + {b}?", a + b),
        'subtraction': lambda a, b: (f"What is {a} - {b}?", a - b) if a >= b else (f"What is {b} - {a}?", b - a),
        'multiplication': lambda a, b: (f"What is {a} × {b}?", a * b),
    }
    
    for _ in range(num_samples):
        op_type = random.choice(list(templates.keys()))
        a = random.randint(1, 100)
        b = random.randint(1, 50 if op_type == 'multiplication' else 100)
        
        question, answer = templates[op_type](a, b)
        formatted = f"Q: {question}\nA: {answer}\n#### {answer}"
        problems.append(formatted)
    
    # Split train/val
    random.shuffle(problems)
    split_idx = int(len(problems) * 0.9)
    return problems[:split_idx], problems[split_idx:], "simple_arithmetic"


def main():
    parser = argparse.ArgumentParser(description="Prepare arithmetic datasets from HuggingFace")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples per dataset")
    parser.add_argument("--datasets", type=str, default="asdiv,math_qa,simple", 
                        help="Comma-separated list: asdiv,math_qa,aqua_rat,simple")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("📚 Preparing Arithmetic Curriculum Datasets from HuggingFace")
    print("=" * 80)
    
    requested_datasets = [d.strip() for d in args.datasets.split(',')]
    
    all_train = []
    all_val = []
    
    # Try each requested dataset
    for dataset_name in requested_datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset_name}")
        print('=' * 80)
        
        if dataset_name == "asdiv":
            train, val, name = prepare_asdiv(output_dir, args.max_samples)
        elif dataset_name == "math_qa":
            train, val, name = prepare_math_qa(output_dir, args.max_samples)
        elif dataset_name == "aqua_rat":
            train, val, name = prepare_aqua_rat_subset(output_dir, args.max_samples)
        elif dataset_name == "simple":
            train, val, name = generate_simple_arithmetic(args.max_samples)
        else:
            print(f"⚠️  Unknown dataset: {dataset_name}")
            continue
        
        if train:
            print(f"✅ {name}: {len(train)} train, {len(val)} val")
            all_train.extend(train)
            all_val.extend(val)
        else:
            print(f"❌ {name}: Failed to load")
    
    if not all_train:
        print("\n❌ No datasets loaded successfully!")
        print("   Falling back to simple arithmetic generation...")
        all_train, all_val, _ = generate_simple_arithmetic(args.max_samples * 3)
    
    # Shuffle combined data
    random.shuffle(all_train)
    random.shuffle(all_val)
    
    # Write output files
    train_file = output_dir / "curriculum_arith_train.txt"
    val_file = output_dir / "curriculum_arith_val.txt"
    
    print("\n" + "=" * 80)
    print("💾 Writing output files...")
    
    with open(train_file, 'w') as f:
        f.write('\n\n'.join(all_train))
    
    with open(val_file, 'w') as f:
        f.write('\n\n'.join(all_val))
    
    print(f"✅ Train: {len(all_train)} problems → {train_file}")
    print(f"✅ Val:   {len(all_val)} problems → {val_file}")
    print(f"✅ Total: {len(all_train) + len(all_val)} problems")
    
    # Sample preview
    print("\n" + "=" * 80)
    print("📝 Sample problems:")
    print("=" * 80)
    for i, problem in enumerate(random.sample(all_train, min(3, len(all_train))), 1):
        print(f"\nExample {i}:")
        print(problem[:200] + "..." if len(problem) > 200 else problem)
    
    print("\n" + "=" * 80)
    print("✅ Dataset preparation complete!")
    print(f"   Train file: {train_file}")
    print(f"   Val file:   {val_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
