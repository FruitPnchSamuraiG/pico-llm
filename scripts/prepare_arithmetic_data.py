#!/usr/bin/env python3
"""
Generate simple arithmetic problems for curriculum learning.
Teaches basic math operations before complex GSM8K reasoning.
"""
import random
import argparse
from pathlib import Path


def generate_addition_problems(num_samples, max_num=100):
    """Generate simple addition problems: a + b = ?"""
    problems = []
    for _ in range(num_samples):
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        answer = a + b
        
        # Format like GSM8K
        question = f"Q: What is {a} + {b}?"
        reasoning = f"A: {a} + {b} = {answer}"
        formatted = f"{question}\n{reasoning}\n#### {answer}"
        problems.append(formatted)
    
    return problems


def generate_subtraction_problems(num_samples, max_num=100):
    """Generate simple subtraction problems: a - b = ?"""
    problems = []
    for _ in range(num_samples):
        a = random.randint(1, max_num)
        b = random.randint(1, min(a, max_num))  # Ensure positive result
        answer = a - b
        
        question = f"Q: What is {a} - {b}?"
        reasoning = f"A: {a} - {b} = {answer}"
        formatted = f"{question}\n{reasoning}\n#### {answer}"
        problems.append(formatted)
    
    return problems


def generate_multiplication_problems(num_samples, max_num=20):
    """Generate simple multiplication problems: a × b = ?"""
    problems = []
    for _ in range(num_samples):
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        answer = a * b
        
        question = f"Q: What is {a} times {b}?"
        reasoning = f"A: {a} × {b} = {answer}"
        formatted = f"{question}\n{reasoning}\n#### {answer}"
        problems.append(formatted)
    
    return problems


def generate_word_problems(num_samples, max_num=50):
    """Generate simple word problems with basic reasoning."""
    problems = []
    templates = [
        ("apples", "bought", "buy"),
        ("cookies", "ate", "eat"),
        ("books", "read", "read"),
        ("toys", "got", "get"),
        ("pencils", "found", "find"),
    ]
    
    for _ in range(num_samples):
        item, past_verb, verb = random.choice(templates)
        start = random.randint(1, max_num)
        gained = random.randint(1, max_num)
        answer = start + gained
        
        question = f"Q: If you have {start} {item} and {verb} {gained} more, how many {item} do you have?"
        reasoning = f"A: You start with {start} {item}. Then you {past_verb} {gained} more. So you have {start} + {gained} = {answer} {item}."
        formatted = f"{question}\n{reasoning}\n#### {answer}"
        problems.append(formatted)
    
    return problems


def generate_multi_step_problems(num_samples, max_num=30):
    """Generate 2-step arithmetic problems."""
    problems = []
    
    for _ in range(num_samples):
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        c = random.randint(1, max_num)
        
        # Random operation combination
        op_type = random.choice(['add_add', 'add_sub', 'mult_add'])
        
        if op_type == 'add_add':
            step1 = a + b
            answer = step1 + c
            question = f"Q: What is {a} + {b} + {c}?"
            reasoning = f"A: First, {a} + {b} = {step1}. Then, {step1} + {c} = {answer}."
        elif op_type == 'add_sub':
            step1 = a + b
            answer = step1 - c if step1 > c else step1 + c  # Ensure positive
            if step1 > c:
                question = f"Q: What is ({a} + {b}) - {c}?"
                reasoning = f"A: First, {a} + {b} = {step1}. Then, {step1} - {c} = {answer}."
            else:
                question = f"Q: What is ({a} + {b}) + {c}?"
                reasoning = f"A: First, {a} + {b} = {step1}. Then, {step1} + {c} = {answer}."
        else:  # mult_add
            step1 = a * b
            answer = step1 + c
            question = f"Q: What is ({a} × {b}) + {c}?"
            reasoning = f"A: First, {a} × {b} = {step1}. Then, {step1} + {c} = {answer}."
        
        formatted = f"{question}\n{reasoning}\n#### {answer}"
        problems.append(formatted)
    
    return problems


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic training data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5000, help="Samples per category")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔢 Generating arithmetic training data...")
    print(f"   Samples per category: {args.num_samples}")
    print(f"   Validation split: {args.val_split}")
    
    # Generate problems
    all_problems = []
    
    print("   Generating addition problems...")
    all_problems.extend(generate_addition_problems(args.num_samples))
    
    print("   Generating subtraction problems...")
    all_problems.extend(generate_subtraction_problems(args.num_samples))
    
    print("   Generating multiplication problems...")
    all_problems.extend(generate_multiplication_problems(args.num_samples))
    
    print("   Generating word problems...")
    all_problems.extend(generate_word_problems(args.num_samples))
    
    print("   Generating multi-step problems...")
    all_problems.extend(generate_multi_step_problems(args.num_samples))
    
    # Shuffle
    random.shuffle(all_problems)
    
    # Split train/val
    num_val = int(len(all_problems) * args.val_split)
    val_problems = all_problems[:num_val]
    train_problems = all_problems[num_val:]
    
    # Write files
    train_file = output_dir / "reasoning_arith_train.txt"
    val_file = output_dir / "reasoning_arith_val.txt"
    
    with open(train_file, 'w') as f:
        f.write('\n\n'.join(train_problems))
    
    with open(val_file, 'w') as f:
        f.write('\n\n'.join(val_problems))
    
    print(f"✅ Generated:")
    print(f"   Train: {len(train_problems)} problems → {train_file}")
    print(f"   Val:   {len(val_problems)} problems → {val_file}")
    print(f"   Total: {len(all_problems)} problems")


if __name__ == "__main__":
    main()
