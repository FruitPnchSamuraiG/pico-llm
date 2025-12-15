#!/usr/bin/env python3
"""
Example: Testing Reasoning Model with <thinking> blocks

Demonstrates:
1. Basic inference with thinking blocks
2. Best-of-N sampling
3. Self-consistency decoding
4. Process reward scoring
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.reasoning_training import (
    best_of_n_sampling,
    self_consistency_decoding,
    ProcessRewardModel,
    extract_thinking_and_answer,
    _load_inference_module
)
import tiktoken
import torch

def main():
    print("🧠 Reasoning Model Demo")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = "/scratch/kk6081/picollm_extend/reasoning_structured_orm/transformer_dpo_final.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Test problems
    problems = [
        {
            "prompt": "Q: If John has 3 apples and buys 5 more, how many does he have? A:",
            "gold": "8"
        },
        {
            "prompt": "Q: A bakery sells cupcakes for $2 each. If you buy 4 cupcakes, how much do you pay? A:",
            "gold": "8"
        },
        {
            "prompt": "Q: If there are 12 months in a year, how many months are in 2 years? A:",
            "gold": "24"
        }
    ]
    
    print(f"📦 Loading model from: {checkpoint_path}")
    print(f"🔧 Device: {device}\n")
    
    # Load model
    inf = _load_inference_module()
    enc = tiktoken.get_encoding('gpt2')
    
    # Model architecture (must match training)
    model = inf.TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=384,
        d_model=512,
        n_heads=8,
        n_blocks=6,
        ff_mult=4
    )
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!\n")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nPlease train a reasoning model first:")
        print("  bash scripts/train_reasoning.sh")
        return
    
    # Demo 1: Basic Inference
    print("=" * 60)
    print("DEMO 1: Basic Inference with Thinking Blocks")
    print("=" * 60)
    
    problem = problems[0]
    print(f"\n📝 Problem: {problem['prompt']}")
    print(f"🎯 Gold answer: {problem['gold']}\n")
    
    with torch.no_grad():
        # Use thinking-aware generation if available
        if hasattr(inf, 'generate_text_with_thinking'):
            text, phase_info = inf.generate_text_with_thinking(
                model,
                enc,
                problem['prompt'],
                max_thinking_tokens=800,
                max_answer_tokens=200,
                device=str(device),
                top_p=0.95,
                temperature=1.0
            )
            print(f"📊 Token usage: {phase_info['thinking_tokens']} thinking + {phase_info['answer_tokens']} answer")
        else:
            text, _ = inf.generate_text(
                model,
                enc,
                problem['prompt'],
                max_new_tokens=256,
                device=str(device),
                top_p=0.95,
                temperature=1.0
            )
    
    print("🤖 Model output:")
    print(text)
    print()
    
    # Parse thinking and answer
    thinking, answer = extract_thinking_and_answer(text)
    if thinking:
        print("💭 Thinking block:")
        print(f"   {thinking[:200]}..." if len(thinking) > 200 else f"   {thinking}")
    if answer:
        print(f"📄 Answer block: {answer[:100]}...")
    print()
    
    # Demo 2: Best-of-N Sampling
    print("=" * 60)
    print("DEMO 2: Best-of-N Sampling (N=8)")
    print("=" * 60)
    
    problem = problems[1]
    print(f"\n📝 Problem: {problem['prompt']}")
    print(f"🎯 Gold answer: {problem['gold']}")
    print(f"\n🔄 Generating 8 solutions...\n")
    
    best, best_score, all_results = best_of_n_sampling(
        model, enc, inf,
        problem['prompt'],
        problem['gold'],
        n=8,
        scoring_method="orm",  # Try "prm" or "logprob"
        device=str(device)
    )
    
    print(f"✨ Best solution (score: {best_score:.3f}):")
    print(best[:300] + "..." if len(best) > 300 else best)
    print()
    
    print(f"📊 All scores:")
    for i, (_, score) in enumerate(all_results, 1):
        print(f"   Sample {i}: {score:.3f} {'✓' if score > 0 else '✗'}")
    print()
    
    # Demo 3: Self-Consistency
    print("=" * 60)
    print("DEMO 3: Self-Consistency Decoding (N=12)")
    print("=" * 60)
    
    problem = problems[2]
    print(f"\n📝 Problem: {problem['prompt']}")
    print(f"🎯 Gold answer: {problem['gold']}")
    print(f"\n🔄 Generating 12 solutions with majority vote...\n")
    
    best_comp, majority, counts = self_consistency_decoding(
        model, enc, inf,
        problem['prompt'],
        n=12,
        temperature=0.7,  # Lower temp for more focused sampling
        device=str(device)
    )
    
    print(f"🗳️  Majority answer: {majority}")
    print(f"📊 Vote distribution:")
    total_votes = sum(counts.values())
    for answer, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_votes
        bar = "█" * int(pct / 5)
        print(f"   {answer:>4s}: {count:2d}/{total_votes:2d} ({pct:5.1f}%) {bar}")
    print()
    
    print(f"✨ Best completion with majority answer:")
    print(best_comp[:300] + "..." if len(best_comp) > 300 else best_comp)
    print()
    
    # Demo 4: Process Reward Scoring
    print("=" * 60)
    print("DEMO 4: Process Reward Model Scoring")
    print("=" * 60)
    
    example_thinking = """
Let me break this down step by step:
1. We start with 5 items
2. We add 3 more items
3. Total = 5 + 3 = 8 items
Therefore, the answer is 8.
"""
    
    print(f"\n💭 Example thinking block:")
    print(example_thinking)
    
    prm = ProcessRewardModel(gold_answer="8")
    avg_score, step_scores = prm.score_thinking(example_thinking)
    
    print(f"\n📊 PRM Analysis:")
    print(f"   Average score: {avg_score:.3f}")
    print(f"   Step scores:")
    
    steps = [s.strip() for s in example_thinking.split('\n') if s.strip()]
    for i, (step, score) in enumerate(zip(steps, step_scores), 1):
        quality = "⭐⭐⭐" if score > 0.7 else "⭐⭐" if score > 0.5 else "⭐"
        print(f"      Step {i} ({score:.2f}) {quality}")
        print(f"      → {step[:60]}...")
    
    print()
    print("=" * 60)
    print("✅ Demo complete!")
    print()
    print("💡 Try these commands:")
    print()
    print("# Train your own reasoning model:")
    print("bash scripts/train_reasoning.sh")
    print()
    print("# Test with different problems:")
    print("python scripts/inference_dpo.py \\")
    print("  --checkpoint <path> \\")
    print("  --prompt 'Q: Your question here A:'")
    print()
    print("# Read full guide:")
    print("cat scripts/REASONING_MODEL_GUIDE.md")
    print()

if __name__ == "__main__":
    main()
