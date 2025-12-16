#!/usr/bin/env python3
"""
Multi-GPU GRPO group generation for 2x speedup.

Splits the dataset across 2 GPUs and runs generation in parallel.

Usage:
    # Use both GPUs (2x faster than single GPU)
    python generate_grpo_groups_multi_gpu.py \
        --checkpoint /scratch/.../gsm8k_transformer_epoch5.pt \
        --input_data data/gsm8k_train_reasoning_structured.txt \
        --output_file /scratch/.../gsm8k_grpo_groups_s8.jsonl \
        --num_samples 8 \
        --batch_size 8 \
        --devices cuda:0 cuda:1
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import tiktoken

# Import pico-llm modules
import importlib.util
spec = importlib.util.spec_from_file_location("pico_llm", "pico-llm.py")
pico_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pico_llm)

spec = importlib.util.spec_from_file_location("inference", "inference.py")
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)

# Import reasoning training for answer extraction
# Add scripts/utils to path
utils_path = Path(__file__).parent.parent / "utils"
if utils_path.exists():
    sys.path.append(str(utils_path))
else:
    # Fallback
    sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "utils"))

try:
    from gsm8k_utils import extract_answer, split_qa  # type: ignore
except ImportError:
    print("Warning: Could not import gsm8k_utils. Ensure scripts/utils/gsm8k_utils.py exists.")
    # Define fallbacks if import fails
    def extract_answer(text): return text.split("####")[-1].strip() if "####" in text else ""
    def split_qa(text): return text.split("A:")[0] + "A:", text.split("A:")[-1]


def worker_process(
    gpu_id: str,
    checkpoint_path: str,
    problems: List[Dict],
    num_samples: int,
    batch_size: int,
    transformer_size: str,
    block_size: int,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    output_queue: mp.Queue,
    progress_queue: mp.Queue
):
    """Worker process that generates GRPO groups on one GPU."""
    
    try:
        device = torch.device(gpu_id)
        enc = tiktoken.get_encoding("gpt2")
        
        # Load model
        if transformer_size == "small":
            embed_size, heads, blocks, ff_mult = 384, 4, 3, 2
        elif transformer_size == "medium":
            embed_size, heads, blocks, ff_mult = 512, 8, 6, 4
        else:  # gpt2-small
            embed_size, heads, blocks, ff_mult = 768, 12, 12, 4
        
        model = inf.TransformerModel(
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
        
        progress_queue.put({
            'type': 'info',
            'gpu': gpu_id,
            'message': f'Model loaded on {gpu_id}'
        })
        
        # Generate groups for assigned problems
        correct_count = 0
        device_obj = torch.device(gpu_id)
        
        with torch.no_grad():
            for prob_idx, prob in enumerate(problems):
                prompt = prob["prompt"]
                gold = prob["gold_answer"]
                
                # Generate N samples for this problem IN PARALLEL
                prompt_tokens = enc.encode(prompt)
                prompt_len = len(prompt_tokens)
                
                # Create batch: N copies of the same prompt
                batch_tokens = torch.tensor([prompt_tokens] * num_samples, 
                                           dtype=torch.long, device=device_obj)
                
                # Thinking-aware generation state
                thinking_phase = [True] * num_samples
                thinking_budget = [800] * num_samples
                answer_budget = [200] * num_samples
                finished = [False] * num_samples
                
                # Autoregressive generation
                # Initialize KV cache for each sample in the batch
                # Since we are doing batch generation, we need a batch-aware KV cache
                # The current TransformerModel implementation supports KV cache but expects (batch, seq_len) input
                # and returns a list of caches.
                
                # Optimization: Use KV-cache for batch generation
                use_kv_cache = True
                kv_cache = None
                
                # Initial pass with prompt
                # batch_tokens shape: (num_samples, prompt_len)
                # Transpose for model: (prompt_len, num_samples)
                logits, kv_cache = model(batch_tokens.transpose(0, 1), kv_cache=None, use_cache=True)
                next_token_logits = logits[-1, :, :]
                
                for step in range(max_new_tokens):
                    if all(finished):
                        break
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top_p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        
                        for i in range(num_samples):
                            if finished[i]:
                                continue
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i, indices_to_remove] = float('-inf')
                    
                    # Sample next tokens
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    
                    # Update sequences and prepare next input
                    next_input_tokens = []
                    
                    for i in range(num_samples):
                        if finished[i]:
                            # If finished, just feed a padding token to keep batch size consistent
                            # (or we could mask it out, but padding is easier)
                            next_input_tokens.append(50256) # Padding token
                            continue
                        
                        token = next_tokens[i].item()
                        
                        # Append to full sequence (for decoding later)
                        # Note: batch_tokens grows, but we only feed the new token to the model
                        # new_seq = torch.cat([batch_tokens[i], torch.tensor([token], device=device_obj)])
                        # batch_tokens[i] = new_seq # Update the main storage
                        # Wait, batch_tokens is a tensor, we can't resize it in-place if it's not a list
                        # We need to handle the growing sequence.
                        
                        next_input_tokens.append(token)
                        
                        # EARLY STOPPING: Check for token repetition (prevents !!!! spam)
                        # Use batch_tokens[i] which is the history so far
                        current_seq = batch_tokens[i]
                        if len(current_seq) > prompt_len + 10:
                            # Check last 5 tokens for repetition
                            # We need to include the new token in the check
                            last_5 = [int(t) for t in current_seq[-4:].tolist()] + [token]
                            if len(set(last_5)) == 1:  # All same token
                                finished[i] = True
                                continue
                            # Check last 10 tokens - if >7 are the same, stop
                            if len(current_seq) >= 9:
                                last_10 = [int(t) for t in current_seq[-9:].tolist()] + [token]
                                if len(last_10) == 10:
                                    most_common = max(set(last_10), key=last_10.count)
                                    if last_10.count(most_common) >= 7:
                                        finished[i] = True
                                        continue
                        
                        # Check for phase transition
                        decoded_token = enc.decode([int(token)])
                        
                        # Update thinking/answer budgets
                        if thinking_phase[i]:
                            thinking_budget[i] -= 1
                            if "</thinking>" in decoded_token or thinking_budget[i] <= 0:
                                thinking_phase[i] = False
                                if thinking_budget[i] <= 0 and "</thinking>" not in decoded_token:
                                    # Force close thinking
                                    # This is tricky with batch KV cache because we need to insert tokens
                                    # For now, just let it transition without inserting tokens to keep batch sync simple
                                    # Or we could insert them into the sequence but not feed them?
                                    # Let's just mark it as answer phase.
                                    pass
                        else:
                            answer_budget[i] -= 1
                            
                            # Check for natural stopping: look for answer completion
                            # Stop if we see "####" followed by a number
                            # decoded_so_far = enc.decode([int(t) for t in new_seq[prompt_len:].tolist()])
                            # Use current_seq + token
                            full_seq_list = current_seq.tolist() + [token]
                            decoded_so_far = enc.decode([int(t) for t in full_seq_list[prompt_len:]])
                            
                            if "####" in decoded_so_far:
                                # Extract text after ####
                                after_hash = decoded_so_far.split("####")[-1].strip()
                                # If we have a number and some trailing content, check if done
                                if after_hash and len(after_hash) > 0:
                                    # Stop if we see newline, or repeated punctuation (!!!, <<<, etc)
                                    if '\n' in after_hash or after_hash.count('!') >= 3 or after_hash.count('<') >= 3:
                                        finished[i] = True
                                    # Also stop if we have extracted a clear number (e.g., "15" or "15\n")
                                    first_word = after_hash.split()[0] if after_hash.split() else ""
                                    if first_word.replace('.','').replace('-','').isdigit():
                                        # We have a clean number, stop after a few more tokens
                                        if len(after_hash) > len(first_word) + 2:
                                            finished[i] = True
                            
                            if answer_budget[i] <= 0:
                                finished[i] = True
                    
                    # Prepare next input tensor
                    # Shape: (num_samples, 1) -> transpose to (1, num_samples)
                    next_input = torch.tensor(next_input_tokens, dtype=torch.long, device=device_obj).unsqueeze(1).transpose(0, 1)
                    
                    # Forward pass with cache
                    logits, kv_cache = model(next_input, kv_cache=kv_cache, use_cache=True)
                    next_token_logits = logits[-1, :, :]
                    
                    # Handle growing batch_tokens tensor
                    # We need to re-stack because sequences might have different lengths if we were appending
                    # But here we are generating token by token, so they grow together.
                    # However, `batch_tokens[i] = new_seq` above fails if sizes differ.
                    # So we should keep batch_tokens as a list of tensors and stack only when needed?
                    # Or just re-stack every time.
                    
                    # Actually, we can just concatenate the new tokens to the batch tensor
                    # batch_tokens: (num_samples, current_len)
                    # next_tokens: (num_samples,)
                    
                    # But wait, `finished` sequences shouldn't grow?
                    # We can just let them grow with padding tokens, and filter later.
                    
                    next_tokens_tensor = torch.tensor(next_input_tokens, dtype=torch.long, device=device_obj).unsqueeze(1)
                    batch_tokens = torch.cat([batch_tokens, next_tokens_tensor], dim=1)

                # Decode samples
                group_samples = []
                for i in range(num_samples):
                    tokens = batch_tokens[i].tolist()
                    # Remove padding and stop at first padding
                    clean_tokens = []
                    for t in tokens:
                        if t == 50256: break
                        clean_tokens.append(t)
                    
                    text = enc.decode(clean_tokens)
                    
                    pred_answer = extract_answer(text)
                    reward = 1.0 if (pred_answer == gold) else 0.0
                    
                    if reward > 0:
                        correct_count += 1
                    
                    group_samples.append({
                        "text": text,
                        "reward": reward,
                        "predicted_answer": pred_answer
                    })
                
                # Send result
                result_data = {
                    "prompt": prompt,
                    "gold_answer": gold,
                    "samples": group_samples,
                    "problem_idx": prob_idx
                }
                output_queue.put(result_data)
                
                # Send sample output every 10 problems
                if (prob_idx + 1) % 10 == 0:
                    progress_queue.put({
                        'type': 'sample',
                        'gpu': gpu_id,
                        'problem_idx': prob_idx,
                        'prompt': prompt,
                        'gold': gold,
                        'sample': group_samples[0]  # Show first sample
                    })
                
                # Send progress update
                if (prob_idx + 1) % 8 == 0 or prob_idx == len(problems) - 1:
                    progress_queue.put({
                        'type': 'progress',
                        'gpu': gpu_id,
                        'processed': prob_idx + 1,
                        'total': len(problems),
                        'correct': correct_count,
                        'total_samples': (prob_idx + 1) * num_samples
                    })
        
        progress_queue.put({
            'type': 'done',
            'gpu': gpu_id,
            'message': f'{gpu_id} finished processing {len(problems)} problems'
        })
        
    except Exception as e:
        progress_queue.put({
            'type': 'error',
            'gpu': gpu_id,
            'message': f'Error on {gpu_id}: {str(e)}'
        })


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU GRPO group generation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--transformer_size", type=str, default="medium")
    parser.add_argument("--devices", nargs='+', default=["cuda:0", "cuda:1"],
                        help="List of GPU devices (e.g., cuda:0 cuda:1)")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load and parse problems
    print(f"🔧 Loading data from: {args.input_data}")
    lines = [ln.strip() for ln in Path(args.input_data).read_text().splitlines() if ln.strip()]
    
    if args.max_problems:
        lines = lines[:args.max_problems]
        print(f"   ⚠️  Limited to {args.max_problems} problems for testing")
    
    problems = []
    for line in lines:
        prompt, gold = split_qa(line)
        if gold:
            problems.append({"prompt": prompt, "gold_answer": gold})
    
    print(f"   ✓ {len(problems)} valid problems loaded")
    
    # Split problems across GPUs
    num_gpus = len(args.devices)
    problems_per_gpu = len(problems) // num_gpus
    
    problem_splits = []
    for i in range(num_gpus):
        start_idx = i * problems_per_gpu
        end_idx = start_idx + problems_per_gpu if i < num_gpus - 1 else len(problems)
        problem_splits.append(problems[start_idx:end_idx])
    
    print(f"\n⚡ Multi-GPU Mode: Using {num_gpus} GPUs")
    for i, (device, split) in enumerate(zip(args.devices, problem_splits)):
        print(f"   {device}: {len(split)} problems")
    print(f"   Expected speedup: ~{num_gpus}x")
    print("")
    
    # Create queues for communication
    output_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # Start worker processes
    start_time = time.time()
    processes = []
    
    for gpu_id, problems_subset in zip(args.devices, problem_splits):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id, args.checkpoint, problems_subset,
                args.num_samples, args.batch_size, args.transformer_size,
                args.block_size, args.max_new_tokens, args.top_p, args.temperature,
                output_queue, progress_queue
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results and monitor progress
    results = []
    gpu_stats = {device: {'processed': 0, 'correct': 0, 'total_samples': 0} 
                 for device in args.devices}
    finished_gpus = set()
    
    while len(finished_gpus) < num_gpus:
        # Check for progress updates
        while not progress_queue.empty():
            update = progress_queue.get()
            
            if update['type'] == 'info':
                print(f"  [{update['gpu']}] {update['message']}")
            
            elif update['type'] == 'sample':
                # Print sample generation output
                print(f"\n{'='*80}")
                print(f"  [{update['gpu']}] Sample Output (Problem #{update['problem_idx']+1})")
                print(f"{'='*80}")
                print(f"  Prompt: {update['prompt'][:100]}...")
                print(f"  Gold Answer: {update['gold']}")
                sample = update['sample']
                print(f"  Predicted: {sample['predicted_answer']}")
                print(f"  Reward: {sample['reward']}")
                
                # Show generated text (truncated)
                text = sample['text']
                if '<thinking>' in text:
                    parts = text.split('<thinking>')
                    if len(parts) > 1:
                        thinking_part = parts[1].split('</thinking>')[0] if '</thinking>' in parts[1] else parts[1]
                        print(f"  Thinking (first 150 chars): {thinking_part[:150]}...")
                
                if '<answer>' in text:
                    answer_part = text.split('<answer>')[-1]
                    print(f"  Answer: {answer_part[:100]}")
                else:
                    # Show last 100 chars if no <answer> tag
                    print(f"  Output (last 100 chars): ...{text[-100:]}")
                print(f"{'='*80}\n")
            
            elif update['type'] == 'progress':
                gpu = update['gpu']
                gpu_stats[gpu]['processed'] = update['processed']
                gpu_stats[gpu]['correct'] = update['correct']
                gpu_stats[gpu]['total_samples'] = update['total_samples']
                
                # Print overall progress
                total_processed = sum(s['processed'] for s in gpu_stats.values())
                total_correct = sum(s['correct'] for s in gpu_stats.values())
                total_samples = sum(s['total_samples'] for s in gpu_stats.values())
                
                elapsed = time.time() - start_time
                rate = total_processed / (elapsed / 60) if elapsed > 0 else 0
                eta = (len(problems) - total_processed) / rate if rate > 0 else 0
                acc = total_correct / total_samples * 100 if total_samples > 0 else 0
                
                print(f"  [{total_processed}/{len(problems)}] Acc: {acc:.1f}% ({total_correct}/{total_samples}) | "
                      f"Rate: {rate:.1f} problems/min | ETA: {eta:.1f}min")
            
            elif update['type'] == 'done':
                finished_gpus.add(update['gpu'])
                print(f"  ✓ {update['message']}")
            
            elif update['type'] == 'error':
                print(f"  ❌ {update['message']}")
                finished_gpus.add(update['gpu'])
        
        # Collect results
        while not output_queue.empty():
            results.append(output_queue.get())
        
        time.sleep(0.1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Collect any remaining results
    while not output_queue.empty():
        results.append(output_queue.get())
    
    # Write results to file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    elapsed = time.time() - start_time
    total_samples = len(results) * args.num_samples
    total_correct = sum(sum(1 for s in r['samples'] if s['reward'] > 0) for r in results)
    avg_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    
    print(f"\n✅ Generation complete!")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Output: {output_path}")
    print(f"   Total problems: {len(results)}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Average accuracy: {avg_acc:.1f}% ({total_correct}/{total_samples})")
    print(f"   Rate: {len(results)/(elapsed/60):.1f} problems/min")
    print(f"   Speedup: ~{num_gpus}x (using {num_gpus} GPUs)")


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
