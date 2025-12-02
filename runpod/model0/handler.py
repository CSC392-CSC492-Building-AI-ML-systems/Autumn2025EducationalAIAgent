# =============================
# FILE: model0/handler.py (clean JSON + thinking tokens)
# =============================
import os
import re
import torch
from typing import Dict, Any
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import runpod

# ---- Config ----
ADAPTER_ID = os.getenv("MODEL0_ADAPTER_ID", "patea4/deepseek-r1-educational-lora-tuned")
MAX_NEW_TOKENS = int(os.getenv("MODEL0_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("MODEL0_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("MODEL0_TOP_P", "1.0"))

ANSWER_RE = re.compile(r"(?i)answer\s*[:\-]?\s*(new|\d+)")

print("[Model0] Loading…")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, use_fast=True)
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_ID,
    device_map="cuda:0",
)
model.eval()
print("[Model0] Ready.")

def _apply_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _split_thinking(text: str):
    """Return (thinking, rest) where thinking is content inside </think> if present."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        think = parts[0]
        if "<think>" in think:
            think = think.split("<think>", 1)[1]
        return think.strip(), parts[1].strip()
    return "", text

def _generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0.0),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    gen = out[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def _parse_answer(text: str):
    m = ANSWER_RE.search(text)
    if not m:
        return {"new": None, "group": None}
    v = m.group(1)
    if v.lower() == "new":
        return {"new": True, "group": None}
    try:
        return {"new": False, "group": int(v)}
    except ValueError:
        return {"new": None, "group": None}

# RunPod handler
def handler(event: Dict[str, Any]):
    """
    Payload (custom worker /runsync):
      {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}]}
    Response:
      {"decision": {"new": bool|None, "group": int|None},
       "thinking": "<model internal reasoning>",
       "text": "<full model output including any trailing content>"}
    """
    payload = event.get("input") or event
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not messages:
        return {"error": "Missing 'messages'"}
    prompt = _apply_template(messages)
    full_text = _generate(prompt)
    thinking, remainder = _split_thinking(full_text)
    decision = _parse_answer(remainder)
    return {"decision": decision, "thinking": thinking, "text": full_text}

runpod.serverless.start({"handler": handler})
