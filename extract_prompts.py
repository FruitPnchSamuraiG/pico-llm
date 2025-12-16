#!/usr/bin/env python3
# Extract just questions (prompts) from gsm8k data

# Read reasoning structured file to get questions and answers
with open('data/gsm8k_train_reasoning_structured.txt', 'r') as f:
    content = f.read()

# Parse blocks
blocks = []
lines = content.split('\n')
current_block = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    if line.startswith('Q:'):
        if current_block:
            blocks.append('\n'.join(current_block))
        current_block = [line]
    else:
        current_block.append(line)

if current_block:
    blocks.append('\n'.join(current_block))

print(f"✅ Found {len(blocks)} problems")

# Extract just Q: ... A: part (before <thinking>)
prompts_and_answers = []
for block in blocks:
    # Extract the question line (first line is "Q: ... A: <thinking>")
    first_line = block.split('\n')[0]  # "Q: ... A: <thinking>"
    
    # Remove everything after "A:" to get just the question
    if ' A:' in first_line:
        question = first_line.split(' A:')[0]  # Just "Q: ..."
    else:
        question = first_line
    
    # Extract answer from the block
    if '#### ' in block:
        answer = block.split('#### ')[-1].strip().split()[0]
    else:
        answer = ""
    
    # Create prompt format: "Q: ... A: #### answer"
    prompt_line = f"{question} A: #### {answer}"
    prompts_and_answers.append(prompt_line)

print(f"\n=== First prompt ===")
print(prompts_and_answers[0])

# Save prompts only
with open('data/gsm8k_train_prompts_only.txt', 'w') as f:
    for prompt_line in prompts_and_answers:
        f.write(f"{prompt_line}\n")

print(f"\n✅ Saved {len(prompts_and_answers)} prompts to data/gsm8k_train_prompts_only.txt")
print("Format: 'Q: ... A: #### answer'")
print("\n=== First 3 prompts ===")
for i in range(min(3, len(prompts_and_answers))):
    print(f"{i+1}. {prompts_and_answers[i]}")
