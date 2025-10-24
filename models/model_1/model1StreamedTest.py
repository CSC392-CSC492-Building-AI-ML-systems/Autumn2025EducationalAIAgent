#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 streamed annotator — cleaned, robust, and stateful (v2)

Fixes in this version:
- No more chat echo: we build chat messages, let the tokenizer create input_ids,
  then slice generated tokens to only the assistant continuation.
- Multiple EOS tokens (Qwen's <|im_end|> + eos) so gen stops at the right place.
- Keeps all the previous improvements (minified XML, repetition controls,
  DONE sentinel cropping, neighbor reuse, depth clamping, etc.).
- Adds a compact I/O table printed after each flush for quick visibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
import torch
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

from formatConstraints import make_prefix_allowed_tokens_fn
import json

from logitProcessors import AsciiOnlyProcessor
# ------------------------------
# Config
# ------------------------------
XML_PATH = "../../data/model_1/inputs/1727009412_parsed.xml"
GT_PATH = "../../data/model_1/outputs/1727009412_training.txt"  # (unused here, kept for future)

# Model settings
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # instruction-tuned
USE_INT4 = True  # set False to prefer BF16/FP16 speed if you have VRAM
MAX_NEW_TOKENS = 48
SUMMARY_WORD_LIMIT = 50

# Prompt packaging
ADD_DONE_SENTINEL = True
DONE_SENTINEL = "DONE"

# Flush parameters
K_TARGET = 1  # targets per flush
N_NEIGH = 20  # preceding neighbors to include

INCLUDE_FEWSHOTS_DEFAULT = True

COLLAPSE_ECHO_INPUTS = False

# ------------------------------
# Statics
# ------------------------------
FEWSHOTS_BLOCK = f"""
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
output (two lines):
spawn shell from editor to grep syslog for errors.
-1

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
output (two lines):
view syslog content within the spawned shell.
0

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
output (two lines):
save changes and exit the editor back to the shell.
1

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
output (two lines):
initiate ssh connection to 10.0.7.138.
0

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
output (two lines):
fetch service status from http://192.168.0.5:8080/status.
0

example f — package install with apt (depth = 0)
neighbor_tail:
- id=70 depth=0  summary:initiate SSH connection to 10.0.7.138
- id=71 depth=-1 summary:authenticate and open remote shell
currDepth before target: -1
<event depth="-2">
  <system_output timestamp="0.0">user@host:~$ </system_output>
  <user_input>s</user_input>
  <system_output>s</system_output>
  <user_input>u</user_input>
  <system_output>u</system_output>
  <user_input>d</user_input>
  <system_output>d</system_output>
  <user_input>o</user_input>
  <system_output>o</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>a</user_input>
  <system_output>a</system_output>
  <user_input>p</user_input>
  <system_output>p</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>i</user_input>
  <system_output>i</system_output>
  <user_input>n</user_input>
  <system_output>n</system_output>
  <user_input>s</user_input>
  <system_output>s</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input>a</user_input>
  <system_output>a</system_output>
  <user_input>l</user_input>
  <system_output>l</system_output>
  <user_input>l</user_input>
  <system_output>l</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>h</user_input>
  <system_output>h</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input>o</user_input>
  <system_output>o</system_output>
  <user_input>p</user_input>
  <system_output>p</system_output>
</event>
output (two lines):
User installs htop package using sudo apt
0
""".strip()



# ------------------------------
# Data types
# ------------------------------
@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int]
    summary_xml: Optional[str]

# Predicted state (index -> {depth, summary})
pred: Dict[int, Dict[str, object]] = {}

# Global events (populated in __main__)
events: List[Event] = []

# ------------------------------
# XML loading & minify
# ------------------------------
REC_BLOCK_RE = re.compile(r"<recording\b[^>]*>.*?</recording>", re.DOTALL)
_MINIFY_TS_RE = re.compile(r"\s+timestamp=\"[^\"]*\"")
_MINIFY_WS_RE = re.compile(r"\s+")


def _extract_recording_block(text: str) -> str:
    m = REC_BLOCK_RE.search(text)
    if not m:
        raise ValueError("Could not find a <recording>...</recording> block in the XML file.")
    return m.group(0)

# --- terminal noise scrubber ---
_DEC_PRIV = re.compile(r"\[\?\d{1,3}[hl]")           # e.g., [?2004h or [?25l
_CSI_SEQ  = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]") # CSI like \x1b[31m or \x1b[?2004l
_OSC_SEQ  = re.compile(r"\x1B\][^\a]*\a")            # OSC ... BEL
_CTRL     = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # control chars except \t \n \r


def sanitize_term_noise(s: str) -> str:
    # remove DEC private mode markers that sometimes appear without ESC
    s = _DEC_PRIV.sub("", s)
    # remove full escape sequences (CSI/OSC)
    s = _CSI_SEQ.sub("", s)
    s = _OSC_SEQ.sub("", s)
    # drop residual control bytes
    s = _CTRL.sub("", s)
    return s


def minify_xml(xml: str) -> str:
    """Remove timestamps/extra whitespace to reduce tokens."""
    x = _MINIFY_TS_RE.sub("", xml)
    x = sanitize_term_noise(x)
    x = _MINIFY_WS_RE.sub(" ", x).strip()
    return x


def load_events(xml_path: str) -> List[Event]:
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    recording_xml = _extract_recording_block(raw)

    if COLLAPSE_ECHO_INPUTS:
        recording_xml = collapse_recording_xml(recording_xml)

    root = etree.fromstring(recording_xml.encode("utf-8"))

    evs: List[Event] = []
    for i, ev in enumerate(root.findall(".//event")):
        xml_str = etree.tostring(ev, encoding="unicode")
        evs.append(Event(idx=i, xml=minify_xml(xml_str), depth_xml=None, summary_xml=None))
    return evs


# ------------------------------
# Depth tracking
# ------------------------------
class DepthState:
    def __init__(self) -> None:
        self.curr = 0  # must remain ≤ 0

    def apply_depth(self, d: int) -> None:
        # Policy: -1 => descend; >0 => ascend by that many; 0 => stay
        if d == -1:
            self.curr -= 1
        elif d > 0:
            self.curr += d
        if self.curr > 0:
            self.curr = 0  # clamp


def compute_curr_depth_upto(idx_exclusive: int) -> int:
    ds = DepthState()
    for i in range(idx_exclusive):
        # Prefer event-backed depth if present; otherwise predicted cache
        d: Optional[int] = None
        if 0 <= i < len(events) and events[i].depth_xml is not None:
            d = events[i].depth_xml
        elif i in pred and isinstance(pred[i].get("depth"), int):
            d = pred[i]["depth"]  # type: ignore[index]
        if isinstance(d, int):
            ds.apply_depth(d)
    return ds.curr


# ------------------------------
# Flush package & prompt building (chat template)
# ------------------------------

def make_flush_package(upto_idx: int, K: int = K_TARGET, N: int = N_NEIGH) -> Dict:
    start_tgt = max(0, upto_idx - (K - 1))
    target_idxs = list(range(start_tgt, upto_idx + 1))

    neigh_end = start_tgt - 1
    neigh_start = max(0, neigh_end - (N - 1))
    neighbor_idxs = list(range(neigh_start, neigh_end + 1)) if neigh_end >= 0 else []

    curr_depth = compute_curr_depth_upto(start_tgt)

    neighbor_tail = []
    for i in neighbor_idxs:
        # Source neighbor depth/summary from the Event object **first**
        if 0 <= i < len(events) and (events[i].depth_xml is not None or events[i].summary_xml is not None):
            neighbor_tail.append({
                "id": i,
                "depth": events[i].depth_xml,
                "summary": events[i].summary_xml,
            })
        elif i in pred:
            neighbor_tail.append({
                "id": i,
                "depth": pred[i].get("depth"),
                "summary": pred[i].get("summary"),
            })

    target_events = [{"id": i, "xml": events[i].xml} for i in target_idxs]
    return {
        "currDepth": curr_depth,
        "neighbor_tail": neighbor_tail,
        "parent_xml": None,
        "target_events": target_events,
        "target_idxs": target_idxs,
    }


def build_instruction(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    neighbors_xml = "\n".join(
        [f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
         for n in pkg.get("neighbor_tail", [])]
    ) or "    <neighbor>(none)</neighbor>"

    targets_xml = "\n".join(
        [f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in pkg["target_events"]]
    )

    examples_xml = f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if use_fewshots else ""

    extra = (
        "- do not copy xml tags or attributes; no repeated phrases\n"
        "- do not mention an address that was not explicitly mentioned in the event\n"
        "- if the target event contains an <annotation> tag or depth value ignore it"
    )

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Use the <think>...</think> section to analyze what is happening in the event and assess whether the event starts a nested subtask (-1), continues at the same level (0), or exits one or more levels up (k).
- In <think>...</think>, generate a concise summary at a higher level, considering broader context.
- Use neighbors ONLY for continuity; do not invent context. Keep <think> concise.
</think_first>

<rules>
- the user’s keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent, 0 for same level, >0 to exit levels
- maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
- write action-oriented summaries; avoid “user”, “they”, “typed”, “inputs”, “enters a command
- depth is relative to the previous events and nothing else”
{extra}
</rules>{examples_xml}

<reason_rules>
explain your thought process before committing to an answer
analyze what command/action is being performed
determine if it enters a subprocess (depth=-1), continues same level (depth=0), or exits (depth=+1)
consider the context from neighbor events
be concise but show your logic clearly (1-3 sentences)
</reason_rules>

<annotation_rules>
write action-oriented summaries; avoid “user”, “they”, “typed”, “inputs”, “enters a command
limit to {SUMMARY_WORD_LIMIT} words maximum
focus on the user's action/intent, not just the raw command
use simple language, present-tense verbs
avoid special chars except: periods, commas, hyphens, underscores, forward slashes, colons
IP addresses, ports, URLs, and paths ARE allowed when diagnostic/contextual
</annotation_rules>

<depth_rules>
must be an integer from the allowed set based on currDepth
-1: entering a new subprocess or nested context
0: continuing at the same context level
+1: exiting one level (returning to parent context)
depth must satisfy: currDepth + depth <= 0 (cannot rise above root)
maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
depth is relative to the previous events and nothing else
</depth_rules>

<instruction>
for each target_event, output exactly one json with "annotation" first, then "depth".
</instruction>

<inputs>
  <curr_depth_max>0</curr_depth_max>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""
    return prompt




def build_messages(user_content: str) -> List[Dict[str, str]]:
    """Build chat messages for the Qwen3 instruct template."""
    return [
        {"role": "system", "content": "You are Model-1. Follow formatting strictly."},
        {"role": "user", "content": user_content},
    ]

# ------------------------------
# Model loading
# ------------------------------

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

    # Optional 4-bit quantization if available
    quant_config = None
    has_bnb = False
    if USE_INT4 and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb  # noqa: F401
            has_bnb = True
        except Exception:
            has_bnb = False
        if has_bnb:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    m1 = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", 
    dtype="auto", 
    device_map="auto")

    # Speed settings
    m1.config.use_cache = True
    if getattr(m1, "generation_config", None):
        m1.generation_config.use_cache = True
    m1.config.attn_implementation = "sdpa"

    # Prefer Flash + MemEfficient on Ada
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # type: ignore
        with _sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
            pass
    except Exception:
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            )
        except Exception:
            pass

    return m1, tok


# ------------------------------
# Generation & parsing
# ------------------------------
@torch.inference_mode()
def generate_pairs(m1, tok, messages, curr_depth: int) -> str:
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(m1.device)

    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != -1:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_ids = list({i for i in eos_ids if i is not None}) or None

    input_len = inputs.shape[-1]

    prefix_fn = make_prefix_allowed_tokens_fn(tok, curr_depth=curr_depth)

    always_allow = set()
    if tok.eos_token_id is not None:
        always_allow.add(tok.eos_token_id)
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != -1:
            always_allow.add(im_end_id)
    except Exception:
        pass
    ascii_gate = LogitsProcessorList([AsciiOnlyProcessor(tok, allow_controls=False, always_allow=always_allow)])

    out_ids = m1.generate(
        input_ids=inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_ids,
        prefix_allowed_tokens_fn=prefix_fn,
        logits_processor=ascii_gate,
        use_cache=True,
    )

    gen_ids = out_ids[0, input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    if ADD_DONE_SENTINEL and DONE_SENTINEL in text:
        text = text.split(DONE_SENTINEL, 1)[0].strip()
    return text



def parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]:
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    out: List[Tuple[int, str]] = []

    def add(obj):
        ann = obj.get("annotation")
        dep = obj.get("depth")
        if isinstance(ann, str) and isinstance(dep, int) and dep >= -1:
            out.append((dep, ann))

    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            j = text.find("\n", i)
            if j == -1:
                break
            i = j + 1
            continue
        i = end
        if isinstance(obj, dict):
            add(obj)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    add(it)
    return out


# ------------------------------
# Pretty I/O table helper
# ------------------------------

def print_io_table(target_idxs: List[int]) -> None:
    # Build a compact table of idx, depth, summary for quick visibility
    header = f"{'idx':>5} | {'depth':>5} | summary"
    print("\n" + header)
    print("-" * len(header))
    for i in target_idxs:
        d = events[i].depth_xml if (0 <= i < len(events)) else None
        s = events[i].summary_xml if (0 <= i < len(events)) else None
        d_str = "" if d is None else str(d)
        s_str = "" if s is None else s
        print(f"{i:>5} | {d_str:>5} | {s_str}")


# ------------------------------
# Main loop
# ------------------------------

def run_flushes(evs: List[Event]) -> None:
    global events
    events = evs  # expose to helpers

    total = len(events)
    start_idx = 0

    m1, tok = load_model_and_tokenizer()
    print("cuda_available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("MODEL:", MODEL_ID)

    # --- accumulate per-flush logs here ---
    all_flush_logs = []  # list of dicts: {"upto": int, "targets": [int], "raw": str, "pairs": [(summary, depth)]}

    for upto in range(start_idx, total):
        pkg = make_flush_package(upto_idx=upto, K=K_TARGET, N=N_NEIGH)
        instr = build_instruction(pkg)
        messages = build_messages(instr)

        print("=" * 80)
        print(
            f"FLUSH upto event idx={upto} | currDepth(before)={pkg['currDepth']} | targets={pkg['target_idxs']}"
        )
        print("- Prompt (truncated) -")
        print(instr[:1000] + ("..." if len(instr) > 1000 else ""))

        print("\n- Model output -")
        raw = generate_pairs(m1, tok, messages, curr_depth=pkg["currDepth"])
        print(raw)

        # Expect summary-first, then depth
        pairs = parse_depth_summary_pairs(raw)
        if len(pairs) != len(pkg["target_idxs"]):
            print("\n(!) Output pairs != #targets; keeping whatever parsed.")

        # Save this flush's info
        all_flush_logs.append({
            "upto": upto,
            "targets": pkg["target_idxs"],
            "raw": raw,
            "pairs": pairs,
        })

        # Apply predictions in order
        for (depth, summary), idx in zip(pairs, pkg["target_idxs"]):
            # Clamp invalid depths (< -1 → -1)
            if depth < -1:
                depth = -1
            # Enforce stack invariant locally
            live_curr = compute_curr_depth_upto(idx)
            temp_curr = live_curr
            if depth == -1:
                temp_curr -= 1
            elif depth > 0:
                temp_curr += depth
            if temp_curr > 0:
                depth = 0

            # Write to both caches: pred[] and events[] (for neighbor reuse)
            pred[idx] = {"depth": depth, "summary": summary}
            if 0 <= idx < len(events):
                events[idx].depth_xml = depth
                events[idx].summary_xml = summary

        print("\n- Recorded predictions -")
        for idx in pkg["target_idxs"]:
            v = pred.get(idx, {})
            print(f"  idx={idx}  depth={v.get('depth')}  summary={v.get('summary')}")

        # Pretty table view (current step)
        print_io_table(pkg["target_idxs"])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================
    # After-loop consolidated print
    # =========================
    print("\n" + "=" * 80)
    print("ALL MODEL OUTPUTS (raw per flush)")
    print("=" * 80)
    for log in all_flush_logs:
        print(f"\n[FLUSH upto={log['upto']} targets={log['targets']}]")
        print(log["raw"])

    # Final consolidated table for all processed targets
    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED TABLE (all processed targets)")
    print("=" * 80)
    header = f"{'idx':>5} | {'depth':>5} | summary"
    print(header)
    print("-" * len(header))
    for log in all_flush_logs:
        for idx in log["targets"]:
            d = events[idx].depth_xml if (0 <= idx < len(events)) else None
            s = events[idx].summary_xml if (0 <= idx < len(events)) else None
            d_str = "" if d is None else str(d)
            s_str = "" if s is None else s
            print(f"{idx:>5} | {d_str:>5} | {s_str}")



# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    events = load_events(XML_PATH)
    print(f"Loaded {len(events)} usable events")
    if events:
        print(events[0].xml[:300] + "...\n")
    run_flushes(events)
