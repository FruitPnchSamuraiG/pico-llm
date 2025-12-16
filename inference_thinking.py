#!/usr/bin/env python3
"""
Inference script for generating reasoning with thinking blocks from SFT checkpoint.

Usage:
    # Single question
    python inference_thinking.py \
        --checkpoint /scratch/kk6081/picollm_extend/finetune_gsm8k/transformer_epoch5.pt \
        --question "Janet has 3 apples. She buys 2 more. How many apples does she have?"
    
    # Multiple questions from file
    python inference_thinking.py \
        --checkpoint /scratch/kk6081/picollm_extend/finetune_gsm8k/transformer_epoch5.pt \
        --input_file data/gsm8k_test.txt \
        --output_file results.jsonl \
        --num_samples 5
"""

import argparse
import json
import torch
import tiktoken
import sys
from pathlib import Path

# Import pico-llm modules
import importlib.util
spec = importlib.util.spec_from_file_location("pico_llm", "pico-llm.py")
pico_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pico_llm)


def load_model(checkpoint_path, device, transformer_size="medium", block_size=256):
    """Load transformer model from checkpoint."""
    enc = tiktoken.get_encoding("gpt2")
    
    # Model architecture configs
    if transformer_size == "small":
        embed_size, heads, blocks, ff_mult = 384, 4, 3, 2
    elif transformer_size == "medium":
        embed_size, heads, blocks, ff_mult = 512, 8, 6, 4
    else:  # gpt2-small
        embed_size, heads, blocks, ff_mult = 768, 12, 12, 4
    
    model = pico_llm.TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=block_size,
        d_model=embed_size,
        n_heads=heads,
        n_blocks=blocks,
        ff_mult=ff_mult,
    )
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model, enc


def format_prompt(question):
    """Format question as prompt for thinking model."""
    return f"Q: {question} A: <thinking>"


def extract_answer(text):
    """Extract numerical answer from generated text."""
    if "####" in text:
        try:
            answer_part = text.split("####")[-1].strip()
            # Extract first number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer_part)
            if numbers:
                return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
        except:
            pass
    return None


def generate_single(model, enc, question, device, max_thinking=800, max_answer=200, 
                    top_p=0.95, temperature=0.9):
    """Generate reasoning for a single question."""
    prompt = format_prompt(question)
    
    output, phase_info = pico_llm.generate_text_with_thinking(
        model, enc, prompt,
        max_thinking_tokens=max_thinking,
        max_answer_tokens=max_answer,
        device=str(device),
        top_p=top_p,
        temperature=temperature
    )
    
    # Extract components
    thinking = ""
    answer = ""
    
    if "<thinking>" in output and "</thinking>" in output:
        thinking_part = output.split("<thinking>")[1].split("</thinking>")[0]
        thinking = thinking_part.strip()
    
    if "</thinking>" in output:
        answer_part = output.split("</thinking>")[1]
        answer = answer_part.strip()
    
    predicted = extract_answer(output)
    
    return {
        "question": question,
        "prompt": prompt,
        "full_output": output,
        "thinking": thinking,
        "answer": answer,
        "predicted_value": predicted,
        "phase_info": phase_info
    }


def main():
    parser = argparse.ArgumentParser(description="Inference with thinking blocks")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--question", help="Single question to answer")
    parser.add_argument("--input_file", help="File with questions (one per line or Q: format)")
    parser.add_argument("--output_file", help="Output JSONL file for batch inference")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--max_thinking", type=int, default=800, help="Max thinking tokens")
    parser.add_argument("--max_answer", type=int, default=200, help="Max answer tokens")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--transformer_size", default="medium", choices=["small", "medium", "gpt2-small"])
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_questions", type=int, help="Limit number of questions")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device(args.device)
    model, enc = load_model(args.checkpoint, device, args.transformer_size, args.block_size)
    print("✓ Model loaded\n")
    
    # Single question mode
    if args.question:
        print("="*80)
        print(f"Question: {args.question}")
        print("="*80)
        
        result = generate_single(
            model, enc, args.question, device,
            args.max_thinking, args.max_answer,
            args.top_p, args.temperature
        )
        
        print(f"\n🧠 THINKING ({result['phase_info']['thinking_tokens']} tokens):")
        print("-" * 80)
        print(result['thinking'])
        print()
        
        print(f"📝 ANSWER ({result['phase_info']['answer_tokens']} tokens):")
        print("-" * 80)
        print(result['answer'])
        print()
        
        if result['predicted_value'] is not None:
            print(f"🎯 Extracted Answer: {result['predicted_value']}")
        print()
        
        return
    
    # Batch mode
    if args.input_file:
        print(f"Reading questions from {args.input_file}...")
        
        # Read questions
        questions = []
        gold_answers = []
        
        with open(args.input_file) as f:
            content = f.read()
        
        # Parse format
        if "Q:" in content:
            # GSM8K format: "Q: question A: #### answer"
            for line in content.strip().split('\n'):
                if line.startswith('Q:'):
                    if '####' in line:
                        q_part = line.split('A:')[0].replace('Q:', '').strip()
                        a_part = line.split('####')[1].strip().split()[0]
                        questions.append(q_part)
                        try:
                            gold_answers.append(float(a_part) if '.' in a_part else int(a_part))
                        except:
                            gold_answers.append(None)
                    else:
                        questions.append(line.replace('Q:', '').strip())
                        gold_answers.append(None)
        else:
            # Plain format: one question per line
            questions = [line.strip() for line in content.strip().split('\n') if line.strip()]
            gold_answers = [None] * len(questions)
        
        if args.max_questions:
            questions = questions[:args.max_questions]
            gold_answers = gold_answers[:args.max_questions]
        
        print(f"✓ Loaded {len(questions)} questions\n")
        
        # Generate
        results = []
        correct = 0
        total = 0
        
        for i, (question, gold) in enumerate(zip(questions, gold_answers)):
            print(f"[{i+1}/{len(questions)}] Processing: {question[:60]}...")
            
            for sample_idx in range(args.num_samples):
                result = generate_single(
                    model, enc, question, device,
                    args.max_thinking, args.max_answer,
                    args.top_p, args.temperature
                )
                
                result['question_id'] = i
                result['sample_id'] = sample_idx
                result['gold_answer'] = gold
                
                # Check correctness
                if gold is not None and result['predicted_value'] is not None:
                    is_correct = abs(result['predicted_value'] - gold) < 1e-3
                    result['correct'] = is_correct
                    if is_correct:
                        correct += 1
                    total += 1
                
                results.append(result)
            
            # Show accuracy so far
            if total > 0:
                acc = 100.0 * correct / total
                print(f"   Current accuracy: {correct}/{total} = {acc:.1f}%")
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            print(f"\n✓ Saved {len(results)} results to {args.output_file}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total samples: {len(results)}")
        if total > 0:
            acc = 100.0 * correct / total
            print(f"Accuracy: {correct}/{total} = {acc:.2f}%")
        print()
        
        return
    
    print("Error: Must specify either --question or --input_file")
    sys.exit(1)


if __name__ == "__main__":
    main()
