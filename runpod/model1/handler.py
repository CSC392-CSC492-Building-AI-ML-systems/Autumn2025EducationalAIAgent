import os
import json
from typing import Dict, Any, List

import runpod
from vllm import LLM, SamplingParams

# ---- Config ----
MODEL_ID = os.getenv("MODEL1_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
MAX_NEW_TOKENS = int(os.getenv("MODEL1_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("MODEL1_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("MODEL1_TOP_P", "1.0"))
REPETITION_PENALTY = float(os.getenv("MODEL1_REP_PENALTY", "1.15"))
GPU_UTIL = float(os.getenv("MODEL1_GPU_UTIL", "0.90"))
MAX_MODEL_LEN = int(os.getenv("MODEL1_MAX_MODEL_LEN", "8192"))

# ---- Load model with vLLM ----
print("[Model1/vLLM] Loading…")
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    gpu_memory_utilization=GPU_UTIL,
    max_model_len=MAX_MODEL_LEN,
    dtype="bfloat16",
)
tok = llm.get_tokenizer()
print("[Model1/vLLM] Ready.")

def _apply_template(messages: List[Dict[str, str]]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _generate(prompt: str) -> str:
    params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        skip_special_tokens=True,
    )
    outs = llm.generate([prompt], params)
    return outs[0].outputs[0].text.strip()

def _split_thinking(text: str):
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        if "<think>" in before:
            before = before.split("<think>", 1)[1]
        return before.strip(), after.strip()
    return "", text

def _extract_first_json(text: str):
    # Strict parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Line-by-line fallback
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        try:
            return json.loads(t)
        except Exception:
            continue
    return {"error": "no_valid_json", "raw": text[:2000]}

# RunPod handler
def handler_m1(event: Dict[str, Any]):
    """
    Payload:
      {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"<inputs>...</inputs>"}]}
    Response:
      {"json": {"annotation": str, "depth": int, ...},
       "thinking": "<model internal reasoning>",
       "text": "<full model output>"}
    """
    payload = event.get("input") or event
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not messages:
        return {"error": "Missing 'messages'"}

    prompt = _apply_template(messages)
    full_text = _generate(prompt)
    thinking, remainder = _split_thinking(full_text)
    js = _extract_first_json(remainder)

    return {"json": js, "thinking": thinking, "text": full_text}

runpod.serverless.start({"handler": handler_m1})
