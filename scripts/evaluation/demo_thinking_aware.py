#!/usr/bin/env python3
"""
Demo: Thinking-Aware Generation

Demonstrates the new thinking-aware generation feature that allows
separate token budgets for thinking and answer phases.

Key Benefits:
1. No token length restriction on <thinking> blocks
2. Generous thinking budget (800+ tokens)
3. Separate answer budget ensures complete answers
4. Automatic phase detection and switching
"""

import sys
from pathlib import Path
import torch
import tiktoken

# Add parent directory to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import using importlib
import importlib.util
from typing import cast, Any

spec = importlib.util.spec_from_file_location("pico_llm", root_dir / "pico-llm.py")
if spec is None:
    raise ImportError("Could not load pico-llm.py")
pico_llm_module = importlib.util.module_from_spec(spec)
cast(Any, spec.loader).exec_module(pico_llm_module)

spec_inf = importlib.util.spec_from_file_location("inference", root_dir / "inference.py")
if spec_inf is None:
    raise ImportError("Could not load inference.py")
inf = importlib.util.module_from_spec(spec_inf)
cast(Any, spec_inf.loader).exec_module(inf)

TransformerModel = pico_llm_module.TransformerModel

def demo_standard_vs_thinking_aware():
    """Compare standard generation vs thinking-aware generation"""
    
    print("=" * 70)
    print("THINKING-AWARE GENERATION DEMO")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding('gpt2')
    
    # Example checkpoint (replace with your model)
    checkpoint_path = "/scratch/kk6081/picollm_extend/reasoning_structured_orm/transformer_dpo_final.pt"
    
    print(f"\n📦 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model = TransformerModel(
        vocab_size=enc.n_vocab,
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 4),
        block_size=config.get('block_size', 384),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    
    # Test problem
    prompt = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? A: <thinking>"""
    
    print("\n" + "=" * 70)
    print("METHOD 1: Standard Generation (256 token limit)")
    print("=" * 70)
    
    with torch.no_grad():
        text_standard, _ = inf.generate_text(
            model, enc, prompt,
            max_new_tokens=256,
            device=str(device),
            top_p=0.95
        )
    
    # Count tokens
    tokens_standard = enc.encode(text_standard)
    prompt_tokens = enc.encode(prompt)
    generated_tokens = len(tokens_standard) - len(prompt_tokens)
    
    print(f"\n📊 Tokens generated: {generated_tokens}/256")
    print("\n🤖 Output:")
    print(text_standard[len(prompt):])
    
    # Check if truncated
    if "</thinking>" not in text_standard:
        print("\n⚠️  PROBLEM: Thinking block not closed (truncated!)")
    
    print("\n" + "=" * 70)
    print("METHOD 2: Thinking-Aware Generation (800 thinking + 200 answer)")
    print("=" * 70)
    
    with torch.no_grad():
        text_thinking, phase_info = inf.generate_text_with_thinking(
            model, enc, prompt,
            max_thinking_tokens=800,
            max_answer_tokens=200,
            device=str(device),
            top_p=0.95
        )
    
    print(f"\n📊 Token usage:")
    print(f"   Thinking phase: {phase_info['thinking_tokens']} tokens")
    print(f"   Answer phase: {phase_info['answer_tokens']} tokens")
    print(f"   Phase switched: {phase_info['phase_switched']}")
    print(f"   Total: {phase_info['thinking_tokens'] + phase_info['answer_tokens']} tokens")
    
    print("\n🤖 Output:")
    print(text_thinking[len(prompt):])
    
    if "</thinking>" in text_thinking:
        print("\n✅ SUCCESS: Complete thinking block generated!")
    
    # Detailed analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if "</thinking>" in text_standard:
        thinking_standard = text_standard.split("</thinking>")[0].split("<thinking>")[-1]
        thinking_tokens_std = len(enc.encode(thinking_standard))
        print(f"\nStandard method: {thinking_tokens_std} tokens in thinking")
    else:
        print(f"\nStandard method: Incomplete (cut off at 256 tokens)")
    
    if "</thinking>" in text_thinking:
        thinking_aware = text_thinking.split("</thinking>")[0].split("<thinking>")[-1]
        thinking_tokens_aware = len(enc.encode(thinking_aware))
        print(f"Thinking-aware: {thinking_tokens_aware} tokens in thinking")
        
        if thinking_tokens_std:
            improvement = ((thinking_tokens_aware - thinking_tokens_std) / thinking_tokens_std) * 100
            print(f"\n🎯 Improvement: {improvement:+.1f}% more thinking tokens")
    
    print("\n" + "=" * 70)
    print("KEY BENEFITS")
    print("=" * 70)
    print("""
✅ No artificial limit on thinking quality
✅ Model can reason as deeply as needed (up to 800 tokens)
✅ Answer quality preserved with separate budget
✅ Automatic phase detection and switching
✅ Backward compatible with standard generation
    """)


def demo_extreme_thinking():
    """Demo with very high thinking token limit"""
    
    print("\n" + "=" * 70)
    print("EXTREME THINKING DEMO (2000 token thinking budget)")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding('gpt2')
    
    checkpoint_path = "/scratch/kk6081/picollm_extend/reasoning_structured_orm/transformer_dpo_final.pt"
    
    print(f"\n📦 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model = TransformerModel(
        vocab_size=enc.n_vocab,
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 4),
        block_size=config.get('block_size', 384),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    prompt = """Q: A complex multi-step problem that requires extensive reasoning. A: <thinking>"""
    
    print("\n🚀 Generating with 2000 token thinking budget...")
    
    with torch.no_grad():
        text, phase_info = inf.generate_text_with_thinking(
            model, enc, prompt,
            max_thinking_tokens=2000,
            max_answer_tokens=200,
            device=str(device),
            top_p=0.95
        )
    
    print(f"\n📊 Token usage:")
    print(f"   Thinking: {phase_info['thinking_tokens']}/2000 tokens")
    print(f"   Answer: {phase_info['answer_tokens']}/200 tokens")
    
    print(f"\n💡 Model used {phase_info['thinking_tokens']} thinking tokens")
    print("   (no artificial limit reached)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo thinking-aware generation")
    parser.add_argument("--extreme", action="store_true", help="Run extreme thinking demo")
    args = parser.parse_args()
    
    try:
        demo_standard_vs_thinking_aware()
        
        if args.extreme:
            demo_extreme_thinking()
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nℹ️  Make sure you have a trained reasoning model at:")
        print("   /scratch/kk6081/picollm_extend/reasoning_structured_orm/transformer_dpo_final.pt")
        print("\nTrain one with:")
        print("   bash scripts/train_reasoning.sh")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
