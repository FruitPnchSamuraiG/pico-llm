"""
Training loop implementation.
"""

import time
import torch
import torch.optim as optim

from utils.loss import compute_next_token_loss
from utils.generation import generate_text


def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    grad_clip=1.0,
                    weight_decay=0.01):
    """
    Train a single model (LSTM, Transformer, or K-gram MLP) on the provided data.
    
    Args:
        model: The neural network model to train
        loader: DataLoader providing training batches
        epochs: Number of full passes through the dataset
        model_name: String identifier for logging and saving
        device: torch.device for GPU/CPU
        lr: Learning rate for optimizer
        log_steps: Print loss every N steps
        sample_interval: Generate text samples every N seconds
        max_steps_per_epoch: Optional cap on steps per epoch (for quick testing)
        enc: Tokenizer for text generation
        monosemantic_info: Optional interpretability data structure
        prompt: Text prompt for generation during training
        grad_clip: Gradient clipping norm to prevent exploding gradients
        weight_decay: L2 regularization strength (AdamW)
    """
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track timing for periodic text generation
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0  # Total steps across all epochs

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode (enables dropout, etc.)
        total_loss = 0.0  # Accumulate loss for epoch average
        partial_loss = 0.0  # Accumulate loss for logging intervals
        partial_count = 0  # Count batches in logging interval

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            # Move batch to GPU/CPU: shape is (seq_len, batch_size)
            batch_tokens = batch_tokens.to(device)

            # Forward pass: get logits for next token prediction
            # logits shape: (seq_len, batch_size, vocab_size)
            logits = model(batch_tokens)
            
            # Compute cross-entropy loss with target shift
            loss = compute_next_token_loss(logits, batch_tokens)

            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients (common in RNNs/Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update model parameters
            optimizer.step()

            # Track loss statistics
            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            # Log training progress at regular intervals
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                # Reset partial counters
                partial_loss = 0.0
                partial_count = 0

            # Generate text samples periodically to monitor quality
            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():  # Disable gradients for faster generation
                    # Generate using greedy decoding (always pick most likely token)
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=None,  # None = greedy
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    # Generate using top-p=0.95 (nucleus sampling, more diverse)
                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=0.95,  # Keep top 95% probability mass
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # Generate using top-p=1.0 (sample from full distribution, most diverse)
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=str(device),
                        top_p=1.0,  # Sample from entire distribution
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                # Schedule next sampling time
                next_sample_time = current_time + sample_interval

            # Early stopping if max steps reached (useful for quick testing)
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        # Print epoch summary
        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint after each epoch (allows resuming training)
        torch.save(model.state_dict(), f"{model_name}_epoch{epoch}.pt")
        print(f"Saved {model_name} weights to {model_name}_epoch{epoch}.pt")
