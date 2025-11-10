import os
import json
import logging
import time
from typing import Dict, Any, List

import runpod
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Config ----
MODEL_ID = os.getenv("MODEL1_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
MAX_NEW_TOKENS = int(os.getenv("MODEL1_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("MODEL1_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("MODEL1_TOP_P", "1.0"))
REPETITION_PENALTY = float(os.getenv("MODEL1_REP_PENALTY", "1.15"))
GPU_UTIL = float(os.getenv("MODEL1_GPU_UTIL", "0.90"))
MAX_MODEL_LEN = int(os.getenv("MODEL1_MAX_MODEL_LEN", "8192"))

logger.info("=" * 60)
logger.info("Configuration loaded:")
logger.info(f"  MODEL_ID: {MODEL_ID}")
logger.info(f"  MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
logger.info(f"  TEMPERATURE: {TEMPERATURE}")
logger.info(f"  TOP_P: {TOP_P}")
logger.info(f"  REPETITION_PENALTY: {REPETITION_PENALTY}")
logger.info(f"  GPU_UTIL: {GPU_UTIL}")
logger.info(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
logger.info("=" * 60)

# ---- Load model with vLLM ----
logger.info(f"[Model1/vLLM] Starting model load: {MODEL_ID}")
model_load_start = time.time()

try:
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
    )
    tok = llm.get_tokenizer()
    model_load_time = time.time() - model_load_start
    logger.info(f"[Model1/vLLM] Model loaded successfully in {model_load_time:.2f} seconds")
    logger.info(f"[Model1/vLLM] Tokenizer vocab size: {tok.vocab_size if hasattr(tok, 'vocab_size') else 'Unknown'}")
except Exception as e:
    logger.error(f"[Model1/vLLM] Failed to load model after {time.time() - model_load_start:.2f} seconds")
    logger.error(f"[Model1/vLLM] Error: {str(e)}", exc_info=True)
    raise

def _apply_template(messages: List[Dict[str, str]]) -> str:
    logger.debug(f"Applying chat template to {len(messages)} messages")
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug(f"Generated prompt length: {len(prompt)} characters")
        return prompt
    except Exception as e:
        logger.error(f"Error applying chat template: {e}", exc_info=True)
        raise

def _generate(prompt: str) -> str:
    logger.info(f"Starting generation with prompt length: {len(prompt)} characters")
    gen_start = time.time()
    
    try:
        params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=True,
        )
        logger.debug(f"Sampling params: temp={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_NEW_TOKENS}")
        
        outs = llm.generate([prompt], params)
        generated_text = outs[0].outputs[0].text.strip()
        
        gen_time = time.time() - gen_start
        tokens_generated = len(generated_text.split())  # Rough estimate
        logger.info(f"Generation completed in {gen_time:.2f}s")
        logger.info(f"Generated ~{tokens_generated} tokens (~{tokens_generated/gen_time:.1f} tokens/sec)")
        logger.debug(f"Generated text length: {len(generated_text)} characters")
        
        return generated_text
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise

def _split_thinking(text: str):
    logger.debug("Splitting thinking tags from response")
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        if "<think>" in before:
            before = before.split("<think>", 1)[1]
        thinking_len = len(before.strip())
        remainder_len = len(after.strip())
        logger.info(f"Found thinking section: {thinking_len} chars, remainder: {remainder_len} chars")
        return before.strip(), after.strip()
    logger.debug("No thinking tags found in response")
    return "", text

def _extract_first_json(text: str):
    logger.debug(f"Extracting JSON from text of length {len(text)}")
    
    # Strict parse
    try:
        parsed = json.loads(text)
        logger.info("Successfully parsed JSON on first attempt")
        return parsed
    except Exception as e:
        logger.debug(f"First JSON parse attempt failed: {e}")
    
    # Line-by-line fallback
    logger.debug("Attempting line-by-line JSON parsing")
    for line_num, line in enumerate(text.splitlines(), 1):
        t = line.strip()
        if not t:
            continue
        try:
            parsed = json.loads(t)
            logger.info(f"Successfully parsed JSON from line {line_num}")
            return parsed
        except Exception:
            continue
    
    logger.warning("No valid JSON found in response, returning error dict")
    return {"error": "no_valid_json", "raw": text[:2000]}

# RunPod handler
def handler_m1(event: Dict[str, Any]):
    """
    Payload:
      {"input": {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"<inputs>...</inputs>"}]}}
    Response:
      {"json": {"annotation": str, "depth": int, ...},
       "thinking": "<model internal reasoning>",
       "text": "<full model output>"}
    """
    request_start = time.time()
    request_id = event.get("id", "unknown")
    
    logger.info("=" * 60)
    logger.info(f"NEW REQUEST - ID: {request_id}")
    logger.info("=" * 60)
    
    try:
        logger.debug(f"Raw event keys: {list(event.keys())}")
        
        payload = event.get("input") or event
        messages = payload.get("messages") if isinstance(payload, dict) else None
        
        if not messages:
            logger.error("Missing 'messages' in payload")
            logger.error(f"Payload structure: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}")
            return {"error": "Missing 'messages' field in request"}

        logger.info(f"Processing {len(messages)} messages")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_preview = str(msg.get("content", ""))[:100]
            logger.debug(f"  Message {i+1}: role={role}, content={content_preview}...")
        
        # Apply template
        template_start = time.time()
        prompt = _apply_template(messages)
        logger.info(f"Template application took {time.time() - template_start:.3f}s")
        
        # Generate
        full_text = _generate(prompt)
        
        # Parse response
        parse_start = time.time()
        thinking, remainder = _split_thinking(full_text)
        js = _extract_first_json(remainder)
        logger.info(f"Response parsing took {time.time() - parse_start:.3f}s")
        
        total_time = time.time() - request_start
        logger.info(f"REQUEST COMPLETED in {total_time:.2f}s")
        logger.info(f"  - Thinking tokens: {len(thinking.split())}")
        logger.info(f"  - JSON fields: {list(js.keys())}")
        logger.info("=" * 60)
        
        return {"json": js, "thinking": thinking, "text": full_text}
    
    except Exception as e:
        error_time = time.time() - request_start
        logger.error("=" * 60)
        logger.error(f"REQUEST FAILED after {error_time:.2f}s - ID: {request_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        logger.error("=" * 60)
        return {"error": str(e), "type": type(e).__name__}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    logger.info(f"Handler function: handler_m1")
    runpod.serverless.start({"handler": handler_m1})