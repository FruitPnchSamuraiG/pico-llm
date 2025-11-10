"""
Text generation utilities.
"""

import torch
from .loss import nucleus_sampling


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    Autoregressive text generation: Unified interface for all models.
    
    ALGORITHM (for each new token):
    1. Encode current text to token sequence
    2. Feed entire sequence through model: (seq_len, 1) -> (seq_len, 1, vocab_size)
    3. Extract logits at last position: logits[-1, 0, :]
    4. Sample next token (greedy or top-p)
    5. Append to sequence and repeat
    
    WHY FEED ENTIRE SEQUENCE?
    - LSTM needs full context to build hidden state
    - Transformer uses causal masking (only attends to previous positions)
    - K-gram only uses last k tokens internally
    - Unified interface simplifies code
    
    OPTIMIZATION OPPORTUNITIES:
    - KV-caching for Transformer (reuse past attention keys/values)
    - Hidden state passing for LSTM (don't recompute from scratch)
    - Currently regenerates entire forward pass each step (simple but slow)
    
    PARAMETERS:
    - model: Neural network (LSTM, Transformer, or K-gram MLP)
    - enc: Tokenizer (tiktoken GPT-2 BPE)
    - init_text: Prompt string to continue from
    - max_new_tokens: How many tokens to generate
    - device: "cpu" or "cuda:0"
    - top_p: None for greedy (argmax), float in (0, 1] for nucleus sampling
    - monosemantic_info: Optional interpretability data (currently unused)
    - do_monosemantic: Whether to run interpretability analysis
    
    RETURNS:
    - final_text: Full generated text (prompt + new tokens)
    - annotated_text: Text with interpretability annotations (if enabled)
    """
    was_training = model.training
    model.eval()  # Disable dropout, set batch norm to eval mode
    
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        # Tokenize initial prompt
        context_tokens = enc.encode(init_text)
        annotation_list = []

        # Generate tokens one at a time (autoregressive)
        for step_i in range(max_new_tokens):
            # Convert token list to tensor: (seq_len, 1)
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            
            # Forward pass: get logits for all positions
            logits_seq = model(seq_tensor)  # (seq_len, 1, vocab_size)
            
            # Extract logits for next token (at last position)
            next_logits = logits_seq[-1, 0, :]  # (vocab_size,)

            # Sample next token
            if top_p is None:
                # Greedy decoding: always pick most likely token
                chosen_token = torch.argmax(next_logits).item()
            else:
                # Nucleus sampling: sample from top-p probability mass
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            # Append to context
            context_tokens.append(chosen_token)

            # Optional: Monosemantic analysis (interpretability stub)
            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    # Restore training mode if it was on
    model.train(was_training)

    # Decode final sequence
    final_text = enc.decode(context_tokens)
    
    # Build annotated text (with interpretability info if available)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


def monosemantic_analysis_for_token(token_id, model, monosemantic_info, enc, device="cpu", top_n=5):
    """
    Placeholder for monosemantic analysis (interpretability).
    """
    return []
