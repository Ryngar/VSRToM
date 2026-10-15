"""Answer extraction adapted from the VSRToM lmms-eval task."""
import re

def parse_vsrtom_response(resp: str):

    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", resp, re.IGNORECASE | re.DOTALL)
    answer_half_match = re.findall(r"<answer>\s*(.*)$", resp, re.IGNORECASE | re.DOTALL)
    boxed_answer_match = re.findall(r"\\boxed\{\s*(.*?)\s*\}", resp, re.IGNORECASE | re.DOTALL)

    if answer_match:
        content = answer_match[-1]
    elif answer_half_match:
        content = answer_half_match[-1]
    elif boxed_answer_match:
        content = boxed_answer_match[-1]
    else:
        content = resp
    

    m = re.search(r"\b([A-D])\s*\.", content)
    if m:
        return m.group(1)

    m = re.search(r":\s*([A-Za-z]+)", content)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", content)
    if m:
        return m.group(1)

    m = re.search(r"\b([a-d])\b", content)
    if m:
        return m.group(1).upper()

    return "OTHER"
