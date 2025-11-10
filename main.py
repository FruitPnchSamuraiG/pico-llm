#!/usr/bin/env python3
"""
main.py - Train and compare multiple language models on TinyStories dataset

PURPOSE: Train and compare multiple language models from scratch on TinyStories dataset:
  1. K-gram MLP: Fixed-window feedforward model (baseline)
  2. LSTM: Recurrent model with hidden state (handles variable context)
  3. Transformer: Attention-based decoder-only model (state-of-the-art)

KEY FEATURES:
  - Nucleus (top-p) sampling for diverse text generation
  - RMSNorm for stable training (used in LLaMA models)
  - Causal decoder-only Transformer with multi-head self-attention
  - CPU-friendly implementation with GPU scaling support
  - Model checkpointing and gradient clipping for stability
  - Flexible CLI for experimentation

Extended implementation for CSCI-GA.2565 (NYU Deep Learning)
Starter code by matus & o1-pro
"""

import argparse
import random
import torch

from datasets import load_dataset
import tiktoken

from models import KGramMLPSeqModel, LSTMSeqModel, TransformerModel
from utils import MixedSequenceDataset, seq_collate_fn, generate_text
from training import train_one_model


def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    DESIGN DECISIONS:
    - Default values chosen for reasonable CPU performance
    - GPU users should increase: batch_size, train_subset_size, block_size, embed_size
    - Transformer disabled by default (use --enable_transformer to activate)
    
    KEY HYPERPARAMETERS:
    - block_size: Maximum sequence length (memory scales quadratically in Transformer!)
    - embed_size: Hidden dimension (affects model capacity and memory)
    - batch_size: Trade-off between gradient noise and memory
    - learning_rate: Too high causes instability, too low slows convergence
    """
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    # CPU-friendly training controls
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. Reduce on CPU, e.g., 4.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs. Reduce on CPU, e.g., 1-2.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Optimizer learning rate.")
    parser.add_argument("--train_subset_size", type=int, default=20000,
                        help="How many TinyStories samples to load. Reduce on CPU, e.g., 1000-5000.")
    parser.add_argument("--sample_interval", type=int, default=30,
                        help="Seconds between text samples during training.")

    # Transformer & model selection flags
    parser.add_argument("--enable_transformer", action="store_true", help="Enable training the Transformer model.")
    parser.add_argument("--enable_kgram", action="store_true", help="Enable training the K-gram MLP model.")
    parser.add_argument("--transformer_heads", type=int, default=2, help="Number of attention heads for Transformer.")
    parser.add_argument("--transformer_blocks", type=int, default=2, help="Number of Transformer blocks.")
    parser.add_argument("--ff_mult", type=int, default=4, help="Feedforward expansion multiplier inside Transformer.")
    parser.add_argument("--no_pos_emb", action="store_true", help="Disable learned positional embeddings (for experimentation).")
    
    # Training stability and quality improvements
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. Prevents exploding gradients.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization weight decay for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps for stable training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for regularization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args


def load_and_tokenize_data(args, enc, vocab_size):
    """
    Load and tokenize TinyStories and custom text files.
    
    Returns:
        train_loader: PyTorch DataLoader for training
    """
    tinystories_seqs = []  # Tokenized TinyStories data
    other_seqs = []  # Tokenized custom file data

    # Load TinyStories dataset from HuggingFace if requested
    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        # Use slice notation to load only first N examples (faster than .select())
        dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{args.train_subset_size}]")
        
        # Tokenize TinyStories data
        for sample in dataset:
            text = sample['text']  # type: ignore
            tokens = enc.encode(text)  # Convert text to token IDs
            tokens = tokens[:args.block_size]  # Truncate to max length
            if len(tokens) > 0:  # Skip empty sequences
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")
    else:
        print("TinyStories weight=0 => skipping TinyStories.")

    # Tokenize custom input files if provided
    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()  # Remove whitespace
                if not line:  # Skip empty lines
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:args.block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    # Create mixed dataset that samples from both sources
    p_tiny = args.tinystories_weight  # Probability of sampling TinyStories
    if len(tinystories_seqs) == 0 and p_tiny > 0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # Create DataLoader for batching and shuffling
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle for better training
        num_workers=0,  # Single-threaded (multi-process can cause issues on some systems)
        collate_fn=seq_collate_fn  # Custom collation for variable-length sequences
    )
    
    return train_loader


def create_models(args, vocab_size, device):
    """
    Instantiate models based on command-line arguments.
    
    Returns:
        models: Dictionary mapping model names to model instances
    """
    models = {}
    
    # K-gram MLP (optional)
    if args.enable_kgram:
        kgram_model = KGramMLPSeqModel(
            vocab_size=vocab_size,
            k=args.kgram_k,
            embed_size=args.embed_size,
            num_inner_layers=args.num_inner_mlp_layers,
            chunk_size=args.kgram_chunk_size
        ).to(device)
        models["kgram_mlp_seq"] = kgram_model

    # LSTM (always included for baseline)
    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.embed_size
    ).to(device)
    models["lstm_seq"] = lstm_model
    
    # Transformer (optional)
    if args.enable_transformer:
        transformer = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.embed_size,
            n_heads=args.transformer_heads,
            n_blocks=args.transformer_blocks,
            block_size=args.block_size,
            ff_mult=args.ff_mult,
            use_pos_emb=not args.no_pos_emb
        ).to(device)
        models["transformer"] = transformer
    
    return models


def main():
    args = parse_args()

    # Set random seeds for reproducibility across runs
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection: prefer CUDA if available, otherwise fall back to CPU
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={args.block_size}, kgram_k={args.kgram_k}, "
          f"chunk_size={args.kgram_chunk_size}, embed_size={args.embed_size}")

    # Load GPT-2 tokenizer (BPE with ~50k vocab)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    # Load and tokenize data
    train_loader = load_and_tokenize_data(args, enc, vocab_size)

    # Create models
    models = create_models(args, vocab_size, device)

    # Train each model
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=args.num_epochs,
            model_name=model_name,
            device=device,
            lr=args.learning_rate,
            log_steps=100,
            sample_interval=args.sample_interval,
            max_steps_per_epoch=args.max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay
        )

        # Final generation from the user-provided prompt
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=str(device),
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
