#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-pass runner (with Unified-Style Prompts + Fewshots):
- Pass 1: extract info (THINK model: short <think> only, then force JSON with schema)
- Pass 2: annotate (INSTRUCT model → concise action string)
- Pass 3: depth (CODER model → integer from allowed set)

Drop-in compatible with your current script. Key changes:
- Prompts for each pass mirror the unified single-pass prompt sections
  (<role>, <output_format>, <thinking_and_reasoning>, <rules>, <annotation_rules>,
   <depth_rules>, <instruction>, <inputs>), adapted per-pass.
- Fewshots are included again via FEWSHOTS_BLOCK and can be toggled on/off.

Usage:
  python three_pass_runner_unified_prompts.py

Tune speed via SPEED_KNOBS.
"""
from __future__ import annotations
import os
import re
import json
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from lxml import etree
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =============================
# SPEED_KNOBS
# =============================
THINK_MAX_NEW_TOKENS = 4096      # short, we only want a brief chain-of-thought
JSON_MAX_NEW_TOKENS  = 200       # room for schema JSON
ANNOTATE_MAX_NEW     = 96
DEPTH_MAX_NEW        = 64
USE_INT4 = True                  # nf4 4-bit quant for speed + memory
USE_BF16 = True                  # if GPU supports bf16
SEED = None                      # None = random each run
RANDOMIZE_SEED = True
INCLUDE_FEWSHOTS = True          # 🔁 toggle fewshots on or off

# =============================
# PATHS (same shape as your runner)
# =============================
XML_PATH = os.environ.get("MODEL1_XML", "../../data/model_1/inputs/1727009412_parsed.xml")

# =============================
# MODELS
# =============================
THINKING_ID = "Qwen/Qwen3-4B-Thinking-2507"
INSTRUCT_ID = "Qwen/Qwen3-4B-Instruct-2507"
CODER_ID    = "Qwen/Qwen2.5-Coder-7B-Instruct"

# =============================
# FEWSHOTS (borrowed from unified prompt)
# =============================
FEWSHOTS_BLOCK = (
    """
examples (for format/logic only — do not output these)

note: event xml often shows keystroke-by-keystroke input with echoed characters, not a full command.

example a — new nested subtask (depth = -1)
neighbor_tail:
- id=10 depth=0  summary=open editor on /etc/ntopng/ntopng.conf
- id=11 depth=0  summary=navigate within config to log options
currDepth before target: -1
input xml:
<event>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>!</user_input><system_output>!</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>-</user_input><system_output>-</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
</event>
output (json):
{"annotation": "spawn shell from editor to grep syslog for errors", "depth": -1}

example b — same-level continuation (depth = 0)
neighbor_tail:
- id=20 depth=-1 summary:spawn shell from editor
- id=21 depth=0  summary:view /var/log/syslog in pager
currDepth before target: -2
input xml:
<event>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <system_output>--- syslog ---</system_output>
</event>
output (json):
{"annotation": "view syslog content within the spawned shell", "depth": 0}

example c — exit one level (depth = +1)
neighbor_tail:
- id=30 depth=-1 summary:open file in editor from shell
- id=31 depth=0  summary:edit configuration options
currDepth before target: -1
input xml:
<event>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>w</user_input><system_output>w</system_output>
  <user_input>q</user_input><system_output>q</system_output>
  <system_output>[wrote config] demo@host:/etc/ntopng$ </system_output>
</event>
output (json):
{"annotation": "save changes and exit the editor back to the shell", "depth": 1}

example d — same-level keystroke sequence (depth = 0)
neighbor_tail:
- id=40 depth=0  summary:interactive shell at home directory
- id=41 depth=0  summary:prepare to connect to a remote host
currDepth before target: 0
input xml:
<event>
  <system_output>demo@boxtop:~$ </system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>7</user_input><system_output>7</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>3</user_input><system_output>3</system_output>
  <user_input>8</user_input><system_output>8</system_output>
</event>
output (json):
{"annotation": "initiate ssh connection to 10.0.7.138", "depth": 0}

example e — explicit address allowed (depth = 0)
neighbor_tail:
- id=50 depth=0  summary:running curl to query service endpoint
- id=51 depth=0  summary:checking api health for troubleshooting
currDepth before target: 0
input xml:
<event>
  <system_output>demo@host:~$ </system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>9</user_input><system_output>9</system_output>
  <user_input>2</user_input><system_output>2</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>6</user_input><system_output>6</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>5</user_input><system_output>5</system_output>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>s</user_input><system_output>s</system_output>
</event>
output (json):
{"annotation": "check api status at http://192.168.0.5:8080/status", "depth": 0}
"""
).strip()

# =============================
# DATA TYPES
# =============================
@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int]
    summary_xml: Optional[str]
    reasoning: Optional[str]

# Globals
pred: Dict[int, Dict[str, object]] = {}
events: List[Event] = []

# =============================
# MINI UTILS
# =============================
REC_BLOCK_RE = re.compile(r"<recording\b[^>]*>.*?</recording>", re.DOTALL)
_MINIFY_TS_RE = re.compile(r"\s+timestamp=\"[^\"]*\"")
_MINIFY_WS_RE = re.compile(r"\s+")
_DEC_PRIV = re.compile(r"\[\?\d{1,3}[hl]")
_CSI_SEQ  = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")
_OSC_SEQ  = re.compile(r"\x1B\][^\x07]*\x07")
_CTRL     = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def _extract_recording_block(text: str) -> str:
    m = REC_BLOCK_RE.search(text)
    if not m:
        raise ValueError("Could not find a <recording>...</recording> block in the XML file.")
    return m.group(0)


def sanitize_term_noise(s: str) -> str:
    s = _DEC_PRIV.sub("", s)
    s = _CSI_SEQ.sub("", s)
    s = _OSC_SEQ.sub("", s)
    s = _CTRL.sub("", s)
    return s


def minify_xml(xml: str) -> str:
    x = _MINIFY_TS_RE.sub("", xml)
    x = sanitize_term_noise(x)
    x = _MINIFY_WS_RE.sub(" ", x).strip()
    return x


def load_events(xml_path: str) -> List[Event]:
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    recording_xml = _extract_recording_block(raw)
    root = etree.fromstring(recording_xml.encode("utf-8"))
    evs: List[Event] = []
    for i, ev in enumerate(root.findall(".//event")):
        xml_str = etree.tostring(ev, encoding="unicode")
        evs.append(Event(idx=i, xml=minify_xml(xml_str), depth_xml=None, summary_xml=None, reasoning=None))
    return evs

# =============================
# DEPTH STACK
# =============================
class DepthState:
    def __init__(self) -> None:
        self.curr = 0

    def apply_depth(self, d: int) -> None:
        if d == -1:
            self.curr -= 1
        elif d > 0:
            self.curr += d
        if self.curr > 0:
            self.curr = 0

def compute_curr_depth_upto(idx_exclusive: int) -> int:
    ds = DepthState()
    for i in range(idx_exclusive):
        d: Optional[int] = None
        if 0 <= i < len(events) and events[i].depth_xml is not None:
            d = events[i].depth_xml
        elif i in pred and isinstance(pred[i].get("depth"), int):
            d = pred[i]["depth"]
        if isinstance(d, int):
            ds.apply_depth(d)
    return ds.curr

# =============================
# MODEL LOADING (cached)
# =============================
_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def _load(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    quant = None
    if USE_INT4 and torch.cuda.is_available():
        try:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if (USE_BF16 and torch.cuda.is_bf16_supported()) else torch.float16,
            )
        except Exception:
            quant = None

    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=(torch.bfloat16 if (USE_BF16 and torch.cuda.is_bf16_supported()) else torch.float16),
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        quantization_config=quant,
    )
    m.config.use_cache = True
    if getattr(m, "generation_config", None):
        m.generation_config.use_cache = True

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    _MODEL_CACHE[model_id] = (m, tok)
    return m, tok

# =============================
# GENERATION HELPERS
# =============================

def _apply_chat(tok, messages: List[Dict[str, str]]):
    input_ids = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    return input_ids


def _gen(m, tok, messages, max_new: int, eos_token_ids: Optional[List[int]] = None,
         prefix_allowed_tokens_fn=None) -> str:
    input_ids = _apply_chat(tok, messages)
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tok.eos_token_id,
    }
    if eos_token_ids:
        kwargs["eos_token_id"] = eos_token_ids
    if prefix_allowed_tokens_fn is not None:
        kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
    out = m.generate(**kwargs)
    cont = out[0, input_ids.shape[-1]:]
    return tok.decode(cont, skip_special_tokens=True)


def _think_then(text: str) -> Tuple[str, str]:
    pos = text.find("</think>")
    if pos == -1:
        return "", text
    return text[:pos], text[pos + len("</think>"):]

# =============================
# PASS 1 — Extract info (unified-style prompt + fewshots, but output <think> only)
# =============================
PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "tokens": {"type": "array", "items": {"type": "string"}},
        "addresses": {"type": "array", "items": {"type": "string"}},
        "files": {"type": "array", "items": {"type": "string"}},
        "packages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["command", "tokens"],
    "additionalProperties": False,
}


def _pass1_system() -> str:
    return (
        "<role>you are an event annotator for a linux terminal session.</role>\n\n"
        "<thinking_and_reasoning>\n"
        "- Be concise\n"
        "- Do not output the entire event in your thinking tokens\n"
        "In your thinking, analyze step-by-step:\n"
        "1. Reconstruct the command from keystroke tokens (if any)\n"
        "2. List all EXPLICIT addresses, paths, filenames, packages\n"
        "3. Note any system prompts or pager/editor context visible in the event\n"
        "</thinking_and_reasoning>\n\n"
        "<rules>\n"
        "- Keystrokes appear separately; combine them before interpreting\n"
        "- Do not invent details\n"
        "- This pass should output ONLY a <think>...</think> block; no JSON here\n"
        "</rules>\n\n"
        + (f"<examples>\n{FEWSHOTS_BLOCK}\n</examples>\n\n" if INCLUDE_FEWSHOTS else "")
        + "<instruction>Think in a short <think> block and stop.</instruction>"
    )


def pass1_extract(event_xml: str) -> Dict:
    m_t, t_t = _load(THINKING_ID)

    sys = {"role": "system", "content": _pass1_system()}
    usr = {"role": "user", "content": (
        "<inputs>\n  <target_event>\n" + event_xml + "\n  </target_event>\n</inputs>" 
        + "\n\nFor the next step, we will convert your notes to JSON with this schema: "
        + json.dumps(PASS1_SCHEMA)
    )}
    raw = _gen(m_t, t_t, [sys, usr], max_new=THINK_MAX_NEW_TOKENS)
    print("PASS 1 RAW:\n ", raw)
    think, _ = _think_then(raw)

    return {"thinking": think.strip()}

# =============================
# PASS 2 — Annotate (unified-style annotation_rules + fewshots)
# =============================

def _pass2_system() -> str:
    return (
        "<role>you are an event annotator for a linux terminal session.</role>\n\n"
        "<annotation_rules>\n"
        "- Describe THIS event only, using details visible in the event\n"
        "- Do NOT include addresses/files/packages unless explicitly present\n"
        "- Output: concise action in 'verb + object' style, present tense\n"
        "- Avoid: 'user', 'typed', 'runs a command'\n"
        "</annotation_rules>\n\n"
        + (f"<examples>\n{FEWSHOTS_BLOCK}\n</examples>\n\n" if INCLUDE_FEWSHOTS else "")
        + "<instruction>Return a single short line. No quotes, no extra text.</instruction>"
    )


def pass2_annotate(event_xml: str, pass1_json: Dict) -> str:
    m_i, t_i = _load(INSTRUCT_ID)
    sys = {"role": "system", "content": _pass2_system()}
    usr = {"role": "user", "content": (
        "<inputs>\n  <event>\n" + event_xml + "\n  </event>\n  <facts>\n" + json.dumps(pass1_json, ensure_ascii=False) + "\n  </facts>\n</inputs>"
    )}
    out = _gen(m_i, t_i, [sys, usr], max_new=ANNOTATE_MAX_NEW)
    return out.strip().splitlines()[0].strip().strip('"')

# =============================
# PASS 3 — Depth (unified-style depth_rules + fewshots)
# =============================

def _allowed_depths(curr_depth: int) -> List[int]:
    max_d = max(0, -curr_depth)
    return [-1] + list(range(0, max_d + 1))


def _pass3_system(allowed: List[int], curr_depth: int) -> str:
    return (
        "<role>you determine task hierarchy depth for a terminal event.</role>\n\n"
        "<depth_rules>\n"
        "- depth represents TASK HIERARCHY\n"
        "- -1 (entering): start a focused subtask serving a larger goal\n"
        "- 0  (continuing): next step in the same workflow\n"
        "- +k (exiting): finish a contained subtask and return up k levels\n"
        f"- Allowed: {allowed}\n"
        f"- Constraint: currDepth({curr_depth}) + depth <= 0\n"
        "</depth_rules>\n\n"
        + (f"<examples>\n{FEWSHOTS_BLOCK}\n</examples>\n\n" if INCLUDE_FEWSHOTS else "")
        + "<output_format>{\"depth\": <integer from Allowed>}</output_format>\n"
        "<instruction>Return JSON only.</instruction>"
    )


def pass3_depth(event_xml: str, annotation: str, curr_depth: int) -> int:
    allowed = _allowed_depths(curr_depth)
    m_c, t_c = _load(CODER_ID)

    sys = {"role": "system", "content": _pass3_system(allowed, curr_depth)}
    usr = {"role": "user", "content": (
        "<inputs>\n  <event>\n" + event_xml + "\n  </event>\n  <annotation>\n" + annotation + "\n  </annotation>\n</inputs>"
    )}

    jtxt = _gen(m_c, t_c, [sys, usr], max_new=DEPTH_MAX_NEW)
    try:
        obj = json.loads(jtxt.strip())
    except Exception:
        m = re.search(r"\{[\s\S]*?\}", jtxt)
        if not m:
            raise ValueError("Pass3: failed to get JSON: " + jtxt[:200])
        obj = json.loads(m.group(0))
    d = int(obj.get("depth", 0))

    # enforce invariant curr + depth <= 0
    if curr_depth + d > 0:
        d = 0
    if d < -1:
        d = -1
    return d

# =============================
# RUNTIME (mirrors your prints)
# =============================

def make_flush_package(upto_idx: int) -> Dict:
    start_tgt = max(0, upto_idx)
    target_idxs = [start_tgt]
    curr_depth = compute_curr_depth_upto(start_tgt)
    target_events = [{"id": start_tgt, "xml": events[start_tgt].xml}]
    return {
        "currDepth": curr_depth,
        "neighbor_tail": [],
        "parent_xml": None,
        "target_events": target_events,
        "target_idxs": target_idxs,
    }


def print_io_table(target_idxs: List[int]) -> None:
    header = f"{'idx':>5} | {'depth':>5} | {'summary':<50} | thinking (truncated)"
    print("\n" + header)
    print("-" * len(header))
    for i in target_idxs:
        d = events[i].depth_xml if (0 <= i < len(events)) else None
        s = events[i].summary_xml if (0 <= i < len(events)) else None
        r = events[i].reasoning if (0 <= i < len(events)) else None
        d_str = "" if d is None else str(d)
        s_str = "" if s is None else s
        r_str = "" if not r else (r[:77] + "..." if len(r) > 80 else r)
        print(f"{i:>5} | {d_str:>5} | {s_str:<50} | {r_str}")


def run_flushes(evs: List[Event]) -> None:
    global events
    events = evs
    total = len(events)

    # Seeding
    if RANDOMIZE_SEED:
        actual_seed = set_seed(None)
        print(f"🎲 Using random seed: {actual_seed}")
    elif SEED is not None:
        actual_seed = set_seed(SEED)
        print(f"🔒 Using fixed seed: {actual_seed}")
    else:
        actual_seed = set_seed(42)
        print(f"🔒 Using default seed: {actual_seed}")

    print("cuda_available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("THINKING:", THINKING_ID)
    print("INSTRUCT:", INSTRUCT_ID)
    print("CODER   :", CODER_ID)

    for idx in range(total):
        pkg = make_flush_package(upto_idx=idx)
        event_xml = pkg["target_events"][0]["xml"]
        curr_depth = pkg["currDepth"]

        print("=" * 80)
        print(f"FLUSH upto event idx={idx} | currDepth(before)={curr_depth} | targets={[idx]}")

        # PASS 1
        t0 = time.time()
        p1 = pass1_extract(event_xml)
        p1_think = p1.get("thinking", "")
        t1 = time.time()
        # PASS 2
        ann = pass2_annotate(event_xml, p1.get("json", {}))
        t2 = time.time()
        # PASS 3
        depth_val = pass3_depth(event_xml, ann, curr_depth)
        t3 = time.time()

        # Save
        pred[idx] = {"depth": depth_val, "summary": ann, "reasoning": p1_think}
        events[idx].depth_xml = depth_val
        events[idx].summary_xml = ann
        events[idx].reasoning = p1_think

        # Logs
        print(f"- Pass1 (think+json): {t1 - t0:.2f}s | Pass2 (annot): {t2 - t1:.2f}s | Pass3 (depth): {t3 - t2:.2f}s")
        print(f"  annotation: {ann}")
        print(f"  depth     : {depth_val}")
        print_io_table([idx])

# =============================
# ENTRY
# =============================
if __name__ == "__main__":
    evs = load_events(XML_PATH)
    print(f"Loaded {len(evs)} usable events")
    if evs:
        print(evs[0].xml[:300] + "...\n")
    run_flushes(evs)
