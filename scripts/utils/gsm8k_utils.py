import re
from typing import Optional, Tuple, List

# ============================================================================
# Special Tokens for Reasoning
# ============================================================================

THINKING_START = "<thinking>"
THINKING_END = "</thinking>"
ANSWER_START = "<answer>"

# These will be added to the vocabulary
SPECIAL_TOKENS = [THINKING_START, THINKING_END, ANSWER_START]

# ============================================================================
# Regex patterns for answer extraction (GSM8K format)
# ============================================================================
RE_ANSWER = re.compile(r"####\s*([^\n\r]+)")
RE_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
RE_LAST_NUMBER = re.compile(r"([-+]?\d+(?:\.\d+)?)")
RE_THINKING_BLOCK = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)(?:####|$)", re.DOTALL)

def extract_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8K-style text.
    GSM8K format: "... #### 42" (answer after ####)
    Also handles LaTeX \boxed{} and fallback to last number.
    """
    # Try GSM8K format first: #### answer
    m = RE_ANSWER.search(text)
    if m:
        ans = m.group(1).strip()
        # Extract just the number if there's extra text
        nums = RE_LAST_NUMBER.findall(ans)
        return nums[-1] if nums else ans
    
    # Try LaTeX \boxed{} format
    m = RE_BOXED.search(text)
    if m:
        ans = m.group(1).strip()
        nums = RE_LAST_NUMBER.findall(ans)
        return nums[-1] if nums else ans
    
    # Fallback: last number in text
    nums = RE_LAST_NUMBER.findall(text)
    if nums:
        return nums[-1]
    
    return None

def extract_thinking_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract thinking block and answer from generated text.
    
    Returns:
        (thinking_content, answer_content)
    """
    thinking = None
    answer_text = None
    
    # Extract thinking block
    m = RE_THINKING_BLOCK.search(text)
    if m:
        thinking = m.group(1).strip()
    
    # Extract answer block
    m = RE_ANSWER_BLOCK.search(text)
    if m:
        answer_text = m.group(1).strip()
    elif THINKING_END in text:
        # Everything after </thinking> is the answer
        answer_text = text.split(THINKING_END, 1)[-1].strip()
    else:
        # No thinking block, treat everything as answer
        answer_text = text
    
    return thinking, answer_text

def split_qa(line: str) -> Tuple[str, str]:
    """Split GSM8K line into (prompt, gold_answer).

    Expected format: "Q: ... A: ... #### 42".

    Returns:
        prompt: prompt ending with " A:"
        gold: extracted gold answer (string)
    """
    if "####" in line:
        q_part, ans_part = line.split("####", 1)
        # GSM8K answers can be multi-token; use numeric extraction if possible.
        gold = extract_answer("#### " + ans_part.strip()) or ans_part.strip().split()[0]
        prompt = q_part.strip()
        if not prompt.endswith(" A:"):
            prompt += " A:"
        return prompt, gold

    if " A: " in line:
        q, rest = line.split(" A: ", 1)
        gold = extract_answer(rest) or ""
        return (q.strip() + " A:"), gold

    return line.strip(), ""
