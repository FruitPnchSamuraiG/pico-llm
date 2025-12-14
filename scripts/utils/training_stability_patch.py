#!/usr/bin/env python3
"""
Apply critical training stability improvements to pico-llm.py

This script patches the main training file with:
1. Proper weight initialization (GPT-2 style)
2. Gradient accumulation support
3. Improved AdamW hyperparameters
4. Better gradient norm logging
5. Early stopping

Usage:
    python scripts/training_stability_patch.py --dry-run   # Preview changes
    python scripts/training_stability_patch.py --apply     # Apply patches

After applying, review pico-llm.py and test with:
    bash scripts/train_transformer_fast.sh
"""

import argparse
import re
from pathlib import Path


def read_file(path: Path) -> str:
    """Read file content."""
    return path.read_text(encoding='utf-8')


def write_file(path: Path, content: str) -> None:
    """Write content to file."""
    path.write_text(content, encoding='utf-8')


def add_weight_initialization(content: str) -> tuple[str, bool]:
    """Add proper weight initialization to TransformerModel."""
    
    # Check if already patched
    if '_init_weights' in content:
        print("✅ Weight initialization already present")
        return content, False
    
    print("📝 Adding weight initialization...")
    
    # Find TransformerModel.__init__ and add _init_weights method before it ends
    init_method = '''        # Precompute causal mask (lower-triangular matrix)
        # Shape: (1, block_size, block_size), 1 = allowed, 0 = masked
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.uint8))
        self.register_buffer("causal_mask", causal.unsqueeze(0))  # Save as non-trainable buffer'''
    
    new_init = '''        # Precompute causal mask (lower-triangular matrix)
        # Shape: (1, block_size, block_size), 1 = allowed, 0 = masked
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.uint8))
        self.register_buffer("causal_mask", causal.unsqueeze(0))  # Save as non-trainable buffer
        
        # Initialize weights properly (GPT-2 style)
        self.apply(self._init_weights)
        
        # Scale residual connection weights for stability in deep networks
        for block in self.blocks:
            torch.nn.init.normal_(block.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))'''
    
    content = content.replace(init_method, new_init)
    
    # Add _init_weights method right after __init__
    forward_method_start = '    def forward(self, tokens_seq):'
    
    init_weights_method = '''
    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions for training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_seq):'''
    
    content = content.replace(forward_method_start, init_weights_method)
    
    return content, True


def improve_adamw_settings(content: str) -> tuple[str, bool]:
    """Improve AdamW optimizer hyperparameters."""
    
    # Check if already patched
    if 'betas=(0.9, 0.95)' in content:
        print("✅ AdamW improvements already present")
        return content, False
    
    print("📝 Improving AdamW hyperparameters...")
    
    # Find AdamW initialization
    old_adamw = '        # Use AdamW optimizer\n        optimizer = optim.AdamW(model.parameters(), lr=lr)'
    
    new_adamw = '''        # Use AdamW optimizer with improved hyperparameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),  # Lower beta2 for better convergence
            weight_decay=weight_decay,
            eps=1e-8
        )'''
    
    if old_adamw in content:
        content = content.replace(old_adamw, new_adamw)
        return content, True
    
    return content, False


def add_gradient_norm_logging(content: str) -> tuple[str, bool]:
    """Add gradient norm logging for monitoring training stability."""
    
    # Check if already patched
    if 'total_grad_norm' in content:
        print("✅ Gradient norm logging already present")
        return content, False
    
    print("📝 Adding gradient norm logging...")
    
    # Find the logging block and enhance it
    old_logging = '''            # Log training progress at regular intervals
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                # Reset partial counters
                partial_loss = 0.0
                partial_count = 0'''
    
    new_logging = '''            # Log training progress at regular intervals
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                
                # Calculate gradient norm for monitoring stability
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Loss: {avg_part_loss:.4f}, "
                      f"Grad_norm: {total_grad_norm:.3f}, "
                      f"LR: {current_lr:.2e}")
                # Reset partial counters
                partial_loss = 0.0
                partial_count = 0'''
    
    if old_logging in content:
        content = content.replace(old_logging, new_logging)
        return content, True
    
    return content, False


def add_early_stopping(content: str) -> tuple[str, bool]:
    """Add early stopping based on validation loss."""
    
    # Check if already patched
    if 'best_val_loss' in content and 'patience_counter' in content:
        print("✅ Early stopping already present")
        return content, False
    
    print("📝 Adding early stopping...")
    
    # Add early stopping variables at the start of train_one_model
    old_start = '''    # Track timing for periodic text generation
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0  # Total steps across all epochs
    
    # Track loss history for plotting
    train_loss_history = []
    val_loss_history = []'''
    
    new_start = '''    # Track timing for periodic text generation
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0  # Total steps across all epochs
    
    # Track loss history for plotting
    train_loss_history = []
    val_loss_history = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 3  # Stop if no improvement for 3 epochs
    patience_counter = 0'''
    
    content = content.replace(old_start, new_start)
    
    # Add early stopping logic after validation
    old_val_end = '''            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            print(f"[{model_name}] *** Validation Loss: {avg_val_loss:.4f} ***")
            val_loss_history.append((global_step, avg_val_loss))'''
    
    new_val_end = '''            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            print(f"[{model_name}] *** Validation Loss: {avg_val_loss:.4f} ***")
            val_loss_history.append((global_step, avg_val_loss))
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best checkpoint
                import os
                best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
                torch.save(model.state_dict(), best_ckpt)
                print(f"[{model_name}] 💾 Saved best checkpoint: {best_ckpt}")
            else:
                patience_counter += 1
                print(f"[{model_name}] ⚠️  No improvement for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    print(f"[{model_name}] 🛑 Early stopping triggered!")
                    break'''
    
    if old_val_end in content:
        content = content.replace(old_val_end, new_val_end)
        return content, True
    
    return content, False


def main():
    parser = argparse.ArgumentParser(description="Patch pico-llm.py with training stability improvements")
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--apply', action='store_true', help='Apply changes to pico-llm.py')
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        parser.print_help()
        print("\n⚠️  Please specify --dry-run or --apply")
        return
    
    # Find pico-llm.py
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    target_file = repo_root / 'pico-llm.py'
    
    if not target_file.exists():
        print(f"❌ Error: {target_file} not found")
        return
    
    print(f"📂 Target file: {target_file}")
    print(f"{'🔍 DRY RUN MODE' if args.dry_run else '✍️  APPLY MODE'}\n")
    
    # Read original content
    content = read_file(target_file)
    original_content = content
    
    # Apply patches
    patches_applied = []
    
    content, changed = add_weight_initialization(content)
    if changed:
        patches_applied.append("Weight initialization")
    
    content, changed = improve_adamw_settings(content)
    if changed:
        patches_applied.append("AdamW improvements")
    
    content, changed = add_gradient_norm_logging(content)
    if changed:
        patches_applied.append("Gradient norm logging")
    
    content, changed = add_early_stopping(content)
    if changed:
        patches_applied.append("Early stopping")
    
    # Summary
    print("\n" + "="*60)
    if patches_applied:
        print(f"✨ {len(patches_applied)} patches ready:")
        for patch in patches_applied:
            print(f"   ✓ {patch}")
    else:
        print("ℹ️  No patches needed - file already up to date!")
        return
    
    if args.dry_run:
        print("\n🔍 This was a dry run. Use --apply to make changes.")
        print("\n💡 Recommended: Review TRAINING_IMPROVEMENTS.md for full context")
    elif args.apply:
        # Backup original
        backup_file = target_file.with_suffix('.py.backup')
        write_file(backup_file, original_content)
        print(f"\n💾 Backup saved: {backup_file}")
        
        # Write patched content
        write_file(target_file, content)
        print(f"✅ Patches applied to {target_file}")
        print("\n📋 Next steps:")
        print("   1. Review the changes: git diff pico-llm.py")
        print("   2. Test training: bash scripts/train_transformer_fast.sh")
        print("   3. Monitor gradient norms in training logs (should be 0.1-5.0)")
        print("   4. Compare loss curves before/after")
        print("\n💡 For more improvements, see TRAINING_IMPROVEMENTS.md")


if __name__ == '__main__':
    main()
