#!/bin/bash
# Quick training test to verify training works at all

cd /home/kk6081/pico_llm_extend/pico-llm
source /scratch/kk6081/ml_fall25/venv/bin/activate

echo "=========================================="
echo "🧪 Quick Training Test (1 epoch, small data)"
echo "=========================================="
echo ""
echo "Goal: Verify that training produces a working checkpoint"
echo "Training on: First 100 examples of GSM8K"
echo "Epochs: 1"
echo "This should take ~5 minutes"
echo ""

# Create test data (first 100 examples)
head -400 data/gsm8k_train_reasoning_structured.txt > /tmp/gsm8k_test_100.txt

# Train from scratch (no base checkpoint)
python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id cuda:0 \
  --checkpoint_dir /scratch/kk6081/picollm_extend/test_training \
  --input_files /tmp/gsm8k_test_100.txt \
  --batch_size 4 \
  --num_epochs 1 \
  --block_size 256 \
  --transformer_size medium \
  --learning_rate 3e-4 \
  --val_split 0.1 \
  --sample_every_steps 50 \
  --prompt "Q: What is 2+2? A:"

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Testing checkpoint..."
echo "=========================================="

# Test the checkpoint
python -c "
import torch, tiktoken
import importlib.util
spec = importlib.util.spec_from_file_location('inference', 'inference.py')
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)
enc = tiktoken.get_encoding('gpt2')
device = torch.device('cuda:0')
model = inf.TransformerModel(vocab_size=enc.n_vocab, block_size=256, d_model=512, n_heads=8, n_blocks=6, ff_mult=4)
state = torch.load('/scratch/kk6081/picollm_extend/test_training/transformer_epoch1.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()
prompt = 'Q: What is 2 + 2? A:'
ids = enc.encode(prompt)
x = torch.tensor([ids], device=device)
print('Generating with greedy decoding...')
print(f'Prompt: {prompt}')
with torch.no_grad():
    for i in range(50):
        logits = model(x)
        next_id = torch.argmax(logits[0, -1, :])
        x = torch.cat([x, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
        if x.size(1) > 256: x = x[:, -256:]
        if i % 10 == 0:
            print(f'Step {i}: {enc.decode(x[0].tolist()[-10:])}')
print('\nFinal output:')
print(enc.decode(x[0].tolist()))
"
