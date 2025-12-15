#!/usr/bin/env python3
"""
Quick verification that thinking-aware generation is properly installed.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import importlib.util
        
        # Load pico-llm.py
        root = Path(__file__).parent.parent.parent.absolute()
        spec = importlib.util.spec_from_file_location("pico_llm", root / "pico-llm.py")
        pico_llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pico_llm)
        
        # Load inference.py
        spec_inf = importlib.util.spec_from_file_location("inference", root / "inference.py")
        inf = importlib.util.module_from_spec(spec_inf)
        spec_inf.loader.exec_module(inf)
        
        print("  ✓ Modules loaded")
        return pico_llm, inf
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return None, None

def test_function_exists(pico_llm, inf):
    """Test that generate_text_with_thinking exists"""
    print("\nTesting function existence...")
    
    if pico_llm is None or inf is None:
        print("  ✗ Modules not loaded")
        return False
    
    # Check pico_llm module
    if hasattr(pico_llm, 'generate_text_with_thinking'):
        print("  ✓ generate_text_with_thinking found in pico_llm module")
    else:
        print("  ✗ generate_text_with_thinking NOT found in pico_llm module")
        return False
    
    # Check inference module
    if hasattr(inf, 'generate_text_with_thinking'):
        print("  ✓ generate_text_with_thinking imported in inference module")
    else:
        print("  ✗ generate_text_with_thinking NOT imported in inference module")
        return False
    
    return True

def test_function_signature(pico_llm):
    """Test function signature is correct"""
    print("\nTesting function signature...")
    
    if pico_llm is None:
        print("  ✗ Module not loaded")
        return False
    
    import inspect
    
    func = pico_llm.generate_text_with_thinking
    sig = inspect.signature(func)
    
    expected_params = [
        'model', 'enc', 'init_text',
        'max_thinking_tokens', 'max_answer_tokens',
        'device', 'top_p', 'temperature', 'repetition_penalty'
    ]
    
    actual_params = list(sig.parameters.keys())
    
    for param in expected_params:
        if param in actual_params:
            print(f"  ✓ Parameter '{param}' found")
        else:
            print(f"  ✗ Parameter '{param}' MISSING")
            return False
    
    # Check defaults
    defaults = {
        'max_thinking_tokens': 800,
        'max_answer_tokens': 200,
        'device': 'cpu',
        'temperature': 1.0
    }
    
    for param, expected_default in defaults.items():
        actual_default = sig.parameters[param].default
        if actual_default == expected_default:
            print(f"  ✓ Default for '{param}' correct: {expected_default}")
        else:
            print(f"  ⚠ Default for '{param}' is {actual_default}, expected {expected_default}")
    
    return True

def test_reasoning_functions():
    """Test that reasoning_training functions are updated"""
    print("\nTesting reasoning_training module...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.absolute()))
        import reasoning_training
        
        import inspect
        
        # Check best_of_n_sampling
        sig = inspect.signature(reasoning_training.best_of_n_sampling)
        params = list(sig.parameters.keys())
        
        if 'max_thinking_tokens' in params and 'use_thinking_mode' in params:
            print("  ✓ best_of_n_sampling updated with new parameters")
        else:
            print("  ✗ best_of_n_sampling missing new parameters")
            return False
        
        # Check self_consistency_decoding
        sig = inspect.signature(reasoning_training.self_consistency_decoding)
        params = list(sig.parameters.keys())
        
        if 'max_thinking_tokens' in params and 'use_thinking_mode' in params:
            print("  ✓ self_consistency_decoding updated with new parameters")
        else:
            print("  ✗ self_consistency_decoding missing new parameters")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_documentation():
    """Test that documentation files exist"""
    print("\nTesting documentation...")
    
    root = Path(__file__).parent.parent.parent.absolute()
    
    docs = [
        ('THINKING_AWARE_GENERATION.md', 'Main documentation'),
        ('IMPLEMENTATION_SUMMARY.md', 'Implementation summary'),
        ('README.md', 'Main README'),
        ('scripts/evaluation/demo_thinking_aware.py', 'Demo script')
    ]
    
    all_exist = True
    for doc_path, desc in docs:
        full_path = root / doc_path
        if full_path.exists():
            print(f"  ✓ {desc}: {doc_path}")
        else:
            print(f"  ✗ {desc} MISSING: {doc_path}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 70)
    print("THINKING-AWARE GENERATION VERIFICATION")
    print("=" * 70)
    
    results = []
    
    # Test imports
    pico_llm, inf = test_imports()
    results.append(pico_llm is not None and inf is not None)
    
    # Test function exists
    results.append(test_function_exists(pico_llm, inf))
    
    # Test function signature
    results.append(test_function_signature(pico_llm))
    
    # Test reasoning functions
    results.append(test_reasoning_functions())
    
    # Test documentation
    results.append(test_documentation())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nThinking-aware generation is properly installed and ready to use.")
        print("\nNext steps:")
        print("  1. Run demo: python scripts/evaluation/demo_thinking_aware.py")
        print("  2. Train model: bash scripts/train_reasoning.sh")
        print("  3. Read docs: THINKING_AWARE_GENERATION.md")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease check the errors above and verify the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
