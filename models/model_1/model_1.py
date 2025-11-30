#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 streamed annotator - vLLM version (CLI-friendly)

This script loads terminal events from an XML file, runs a vLLM model to
annotate each event with a (depth, summary) pair, and writes the
predictions to an output file.

Basic usage
-----------

    python model_1.py events.xml annotations.jsonl

Positional args:
    events.xml          Path to XML file containing <event> nodes.
    annotations_out     Output path for predictions.

Output format is inferred from the extension unless --output-format is provided:
    .jsonl  -> JSON Lines (default): one object per line
    .json   -> JSON array of objects
    .txt    -> Legacy format: alternating depth / summary lines

Environment overrides
---------------------
You can override some model parameters via environment variables:
    MODEL_ID           (default: "openai/gpt-oss-20b")
    GPU_UTIL           (default: 0.9)
    MAX_MODEL_LEN      (default: 131072)
    DTYPE              (default: "bfloat16")
    MAX_NEW_TOKENS     (default: 2500)
    SUMMARY_WORD_LIMIT (default: 50)
    VLLM_TP            (tensor parallel, default: 1)

"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
from vllm import LLM, SamplingParams

# ------------------------------
# Config (env-overridable)
# ------------------------------
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-20b")
GPU_UTIL = float(os.getenv("GPU_UTIL", "0.9"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "131072"))
DTYPE = os.getenv("DTYPE", "bfloat16")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2500"))
SUMMARY_WORD_LIMIT = int(os.getenv("SUMMARY_WORD_LIMIT", "50"))

# Parallelism / memory controls (env overridable)
TP_SIZE = int(os.getenv("VLLM_TP", "1"))

# Flush parameters (can be overridden via CLI)
K_TARGET = 1
N_NEIGH = 200

INCLUDE_FEWSHOTS_DEFAULT = True

# Global singleton LLM
_GLOBAL_LLM: Optional[LLM] = None

# ------------------------------
# Statics: few-shots
# ------------------------------
FEWSEP = "═══════════════════════════════════════════════════════════════════════════════"

SYSTEM_ROLE = f"""You are an expert terminal session annotator. Your goal is to identify goals/subgoals and generate concise action summaries.

Rules:
- Summaries must be ≤{SUMMARY_WORD_LIMIT} words, action-oriented (avoid "user", "typed", "enters")
- Depth represents goal transitions: -1=start subtask, 0=continue, +N=finish N levels
- Reconstruct full commands from keystroke sequences before interpreting
- Output valid JSON only
""".strip()

# Few-shot examples (same content as original script)
FEWSHOTS_BLOCK = """
EXAMPLES (for reference only)

DEPTH LOGIC:
- depth=-1: STARTING a new subtask (multi-step goal like "create backup", "run tests", "edit config")
- depth=0: CONTINUING the same subtask
- depth=+N: FINISHING N subtasks (returning to parent goal)

NOTE: XML shows keystroke-by-keystroke input. Reconstruct full commands first.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A — Starting a backup subtask (depth=-1)

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
currDepth: 0

input xml:
<event>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>-</user_input><system_output>-</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>z</user_input><system_output>z</system_output>
  <user_input>f</user_input><system_output>f</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>b</user_input><system_output>b</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>k</user_input><system_output>k</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <system_output>Creating backup.tar...</system_output>
</event>

output:
{"annotation": "Create compressed backup archive of source data", "depth": -1}

Why: Starting a new multi-step backup workflow (compress, verify, move).

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE B — Continuing backup subtask (depth=0)

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
  - id=2 depth=-1 summary="Create compressed backup archive of source data"
currDepth: -1

input xml:
<event>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>-</user_input><system_output>-</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>b</user_input><system_output>b</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>k</user_input><system_output>k</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <system_output>-rw-r--r-- 1 user staff 42M backup.tar</system_output>
</event>

output:
{"annotation": "Verify backup archive and check file size", "depth": 0}

Why: Still in backup workflow, checking result of previous step.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE C — Finishing backup subtask (depth=+1)

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
  - id=2 depth=-1 summary="Create compressed backup archive of source data"
  - id=3 depth=0  summary="Verify backup archive and check file size"
currDepth: -1

input xml:
<event>
  <user_input>m</user_input><system_output>m</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>b</user_input><system_output>b</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>k</user_input><system_output>k</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</userinput><system_output>r</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>i</userinput><system_output>i</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>e</userinput><system_output>e</system_output>
  <system_output>Moved to archive/</system_output>
</event>

output:
{"annotation": "Move backup to archive folder and complete backup task", "depth": 1}

Why: Backup workflow complete, returning to general work.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE D — Starting test/debug subtask (depth=-1)

neighbor_tail:
  - id=0 depth=0  summary="Navigate to project root"
  - id=1 depth=0  summary="Check Git branch status"
currDepth: 0

input xml:
<event>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>t</userinput><system_output>t</system_output>
  <user_input>e</userinput><system_output>e</system_output>
  <user_input>s</userinput><system_output>s</system_output>
  <user_input>t</userinput><system_output>t</system_output>
  <system_output>===== test session starts =====</system_output>
</event>

output:
{"annotation": "Start pytest test run for project", "depth": -1}

Why: Beginning focused testing/debugging workflow.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE E — Nested editor within environment setup (depth=-1)

neighbor_tail:
  - id=0 depth=0  summary="Enter project setup directory"
  - id=1 depth=-1 summary="Create and activate virtual environment"
  - id=2 depth=0  summary="Install core dependencies"
currDepth: -1

input xml:
<event>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>i</userinput><system_output>i</system_output>
  <user_input>m</userinput><system_output>m</system_output>
  <user_input> </userinput><system_output> </system_output>
  <user_input>c</userinput><system_output>c</system_output>
  <user_input>o</userinput><system_output>o</system_output>
  <user_input>n</userinput><system_output>n</system_output>
  <user_input>f</userinput><system_output>f</system_output>
  <user_input>i</userinput><system_output>i</system_output>
  <user_input>g</userinput><system_output>g</system_output>
  <user_input>.</userinput><system_output>.</system_output>
  <user_input>y</userinput><system_output>y</system_output>
  <user_input>a</userinput><system_output>a</system_output>
  <user_input>m</userinput><system_output>m</system_output>
  <user_input>l</userinput><system_output>l</system_output>
  <system_output>Opening vim...</system_output>
</event>

output:
{"annotation": "Open config file in vim during environment setup", "depth": -1}

Why: Nested subtask within the environment setup workflow.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE F — Exit editor, stay in parent task (depth=+1)

neighbor_tail:
  - id=0 depth=0  summary="Enter project setup directory"
  - id=1 depth=-1 summary="Create and activate virtual environment"
  - id=2 depth=0  summary="Install core dependencies"
  - id=3 depth=-1 summary="Open config file in vim during environment setup"
currDepth: -2

input xml:
<event>
  <user_input>:</userinput><system_output>:</system_output>
  <user_input>w</userinput><system_output>w</system_output>
  <user_input>q</userinput><system_output>q</system_output>
  <system_output>config.yaml written</system_output>
  <system_output>(venv) $</system_output>
</event>

output:
{"annotation": "Save config changes and exit vim", "depth": 1}

Why: Editor task done, back to environment setup level.
""".strip()


# ------------------------------
# Event model & global state
# ------------------------------
@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int] = None
    summary_xml: Optional[str] = None


# Global state
events: List[Event] = []
pred: Dict[int, Dict[str, object]] = {}


# ------------------------------
# XML parsing
# ------------------------------
def load_events(xml_path: str) -> List[Event]:
    tree = etree.parse(xml_path)
    root = tree.getroot()
    out: List[Event] = []

    for i, ev_el in enumerate(root.xpath("//event")):
        depth = ev_el.get("depth")
        summary = ev_el.get("summary")

        if depth is not None:
            depth = int(depth)
        if summary is not None:
            summary = summary.strip()

        xml_str = etree.tostring(ev_el, encoding="unicode")

        out.append(
            Event(
                idx=i,
                xml=xml_str,
                depth_xml=depth,
                summary_xml=summary,
            )
        )
    return out


# ------------------------------
# Depth computation
# ------------------------------
def compute_curr_depth_upto(idx: int) -> int:
    curr = 0
    for i in range(idx):
        dep = events[i].depth_xml
        if dep is None:
            continue
        if dep == -1:
            curr -= 1
        elif dep > 0:
            curr += dep
    return curr


# ------------------------------
# Packaging for prompts
# ------------------------------
def make_flush_package(upto_idx: int, K: int = 1, N: int = 20) -> Dict:
    target_idxs = list(range(max(0, upto_idx - K + 1), upto_idx + 1))
    start_neigh = max(0, target_idxs[0] - N)
    neighbor_idxs = list(range(start_neigh, target_idxs[0]))

    def get_sum(i: int) -> str:
        if 0 <= i < len(events):
            s = events[i].summary_xml
            return s if s else "???"
        return "???"

    def get_dep(i: int) -> int:
        if 0 <= i < len(events):
            d = events[i].depth_xml
            return d if d is not None else 999
        return 999

    neighbor_info = [
        f"- id={i} depth={get_dep(i)}  summary={get_sum(i)}" for i in neighbor_idxs
    ]

    target_events = [events[i].xml for i in target_idxs if 0 <= i < len(events)]
    currDepth = compute_curr_depth_upto(target_idxs[0])

    return {
        "target_idxs": target_idxs,
        "neighbor_info": neighbor_info,
        "target_events": target_events,
        "currDepth": currDepth,
    }


def _format_neighbors(pkg: dict) -> str:
    """Format neighbor information as XML."""
    neighbor_items = []
    if pkg.get("neighbor_info"):
        for line in pkg["neighbor_info"]:
            match = re.match(r"- id=(\d+) depth=(-?\d+)\s+summary=(.+)", line)
            if match:
                nid, ndepth, nsummary = match.groups()
                neighbor_items.append(
                    {"id": nid, "depth": ndepth, "summary": nsummary}
                )

    if not neighbor_items:
        return "    <neighbor>(none)</neighbor>"

    return "\n".join(
        f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
        for n in neighbor_items
    )


def _format_targets(pkg: dict) -> str:
    """Format target events as XML."""
    target_items = [
        {"id": idx, "xml": xml_str}
        for idx, xml_str in zip(pkg["target_idxs"], pkg["target_events"])
    ]
    return "\n".join(
        f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in target_items
    )


def build_instruction(pkg: dict, use_fewshots: bool = True) -> str:
    """Build the prompt with minimal but effective structure."""

    neighbors_xml = _format_neighbors(pkg)
    targets_xml = _format_targets(pkg)

    examples_section = f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>\n" if use_fewshots else ""

    return f"""<task>
Annotate terminal events by identifying goals/subgoals and depth transitions.
</task>

<output_format>
{{"annotation": "<action summary ≤{SUMMARY_WORD_LIMIT} words>", "depth": <integer ≥ -1>}}
</output_format>

<depth_logic>
Ask: Is this event...
1. STARTING a new subtask (backup, testing, editing)? → depth = -1
2. CONTINUING the current subtask? → depth = 0  
3. FINISHING one or more subtasks? → depth = +N (count levels)

Stack rule: currDepth + depth_change must stay ≤ 0
</depth_logic>

<summary_rules>
- Action-oriented: "Compile program" not "User runs make"
- Combine keystrokes into full commands before interpreting
- Include purpose when clear from context
- Avoid: "user", "typed", "inputs", "enters"
</summary_rules>
{examples_section}<instruction>
For each target event, output one JSON object.
Think briefly, then output your annotation.
</instruction>

<inputs>
  <curr_depth>{pkg.get("currDepth", 0)}</curr_depth>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""


def build_messages(instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": instruction},
    ]


# ------------------------------
# Model loading (vLLM)
# ------------------------------
def load_model() -> LLM:
    """Singleton vLLM loader with tensor parallel support."""
    global _GLOBAL_LLM

    if _GLOBAL_LLM is not None:
        return _GLOBAL_LLM

    visible = os.getenv("CUDA_VISIBLE_DEVICES", "(default)")
    print(
        "[load_model] Loading model with vLLM\n"
        f"  model                 : {MODEL_ID}\n"
        f"  CUDA_VISIBLE_DEVICES  : {visible}\n"
        f"  tensor_parallel_size  : {TP_SIZE}\n"
        f"  gpu_memory_utilization: {GPU_UTIL}\n"
        f"  max_model_len         : {MAX_MODEL_LEN}\n"
        f"  dtype                 : {DTYPE}",
        flush=True,
    )

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TP_SIZE,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype=DTYPE,
        seed=42,
    )

    print("[load_model] Model loaded successfully", flush=True)
    _GLOBAL_LLM = llm
    return llm


# ------------------------------
# Generation (vLLM)
# ------------------------------
def generate_with_thinking(llm: LLM, messages: List[Dict[str, str]]) -> Tuple[str, str, int, int]:
    """Generate with thinking model using vLLM.

    Returns: (full_output_with_thinking, extracted_json, prompt_tokens, generated_tokens)
    """
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.2,
        skip_special_tokens=True,
        seed=42,
    )

    outputs = llm.generate([prompt], sampling_params)

    req_out = outputs[0]
    first_out = req_out.outputs[0]

    full_output = first_out.text.strip()

    prompt_tokens = len(getattr(req_out, "prompt_token_ids", []))
    generated_tokens = len(getattr(first_out, "token_ids", []))

    if "</think>" in full_output:
        json_part = full_output.split("</think>", 1)[1].strip()
    else:
        json_part = full_output

    return full_output, json_part, prompt_tokens, generated_tokens


# ------------------------------
# JSON parsing for depth/summary
# ------------------------------
def parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]:
    """Robustly extract (depth, annotation) pairs from an arbitrary text blob.

    Strategy:
    - Scan for every '{' in the text.
    - At each '{', try json.JSONDecoder.raw_decode.
    - Accept dicts or lists of dicts containing "annotation" and "depth".
    - Coerce depth from string to int when possible.
    - Ignore everything else (logs, reasoning, junk).
    """
    dec = json.JSONDecoder()
    out: List[Tuple[int, str]] = []
    n = len(text)
    i = 0

    def maybe_add(obj):
        """If obj is a dict or list of dicts with annotation+depth, add to out."""

        def add_one(d):
            if not isinstance(d, dict):
                return
            ann = d.get("annotation")
            dep = d.get("depth")

            if not isinstance(ann, str):
                return

            if isinstance(dep, str):
                try:
                    dep = int(dep.strip())
                except Exception:
                    return

            if not isinstance(dep, int):
                return

            if dep < -1:
                return

            out.append((dep, ann))

        if isinstance(obj, dict):
            add_one(obj)
        elif isinstance(obj, list):
            for item in obj:
                add_one(item)

    while True:
        start = text.find("{", i)
        if start == -1:
            break

        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue

        maybe_add(obj)
        i = end

    return out


# ------------------------------
# Pretty I/O table helper
# ------------------------------
def print_io_table(target_idxs: List[int]) -> None:
    header = f"{'idx':>5} | {'depth':>5} | summary"
    print("\n" + header)
    print("-" * len(header))
    for i in target_idxs:
        if not (0 <= i < len(events)):
            continue
        d = events[i].depth_xml
        s = events[i].summary_xml
        d_str = "" if d is None else str(d)
        s_str = "" if s is None else s
        print(f"{i:>5} | {d_str:>5} | {s_str}")


# ------------------------------
# Main simple inference loop
# ------------------------------
def run_flushes(evs: List[Event]) -> Dict[int, Dict[str, object]]:
    """Run annotations over a list of events.

    Returns a dict: {idx: {"depth": int, "summary": str}}
    """
    global events, pred
    events = evs
    pred = {}

    total = len(events)
    start_idx = 0

    llm = load_model()

    print("MODEL:", MODEL_ID)
    print("Using vLLM for optimized inference")

    for upto in range(start_idx, total):
        pkg = make_flush_package(upto_idx=upto, K=K_TARGET, N=N_NEIGH)
        instr = build_instruction(pkg, use_fewshots=INCLUDE_FEWSHOTS_DEFAULT)
        messages = build_messages(instr)

        print("=" * 80)
        print(
            f"FLUSH upto event idx={upto} | currDepth(before)={pkg['currDepth']} | targets={pkg['target_idxs']}"
        )

        print("\n--- Model output (with thinking) ---")
        full_output, json_part, prompt_tokens, gen_tokens = generate_with_thinking(llm, messages)
        print(full_output)

        total_tokens = prompt_tokens + gen_tokens
        print(f"\n[Tokens] prompt={prompt_tokens} | generated={gen_tokens} | total={total_tokens}")

        pairs = parse_depth_summary_pairs(json_part)

        pairs = [
            (depth, ann)
            for (depth, ann) in pairs
            if ann is not None and ann.strip() not in ("...", '"..."') and len(ann.strip()) >= 5
        ]

        if len(pairs) > len(pkg["target_idxs"]):
            pairs = pairs[-len(pkg["target_idxs"]):]

        if len(pairs) != len(pkg["target_idxs"]):
            print("\n(!) Output pairs != #targets; keeping whatever parsed.")

        for (depth, summary), idx in zip(pairs, pkg["target_idxs"]):
            if not pkg["neighbor_info"]:
                depth = 0
            else:
                if depth < -1:
                    depth = -1

                live_curr = compute_curr_depth_upto(idx)
                temp_curr = live_curr
                if depth == -1:
                    temp_curr -= 1
                elif depth > 0:
                    temp_curr += depth

                if temp_curr > 0:
                    depth = 0

            pred[idx] = {"depth": depth, "summary": summary}
            if 0 <= idx < len(events):
                events[idx].depth_xml = depth
                events[idx].summary_xml = summary

        print("\n- Recorded predictions -")
        for idx in pkg["target_idxs"]:
            v = pred.get(idx, {})
            print(f"  idx={idx}  depth={v.get('depth')}  summary={v.get('summary')}")

        print_io_table(pkg["target_idxs"])

    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED TABLE")
    print("=" * 80)
    header = f"{'idx':>5} | {'depth':>5} | summary"
    print(header)
    print("-" * len(header))
    for i, ev in enumerate(events):
        d = ev.depth_xml
        s = ev.summary_xml
        d_str = "" if d is None else str(d)
        s_str = "" if s is None else s
        print(f"{i:>5} | {d_str:>5} | {s_str}")

    return pred


# ------------------------------
# Output helpers
# ------------------------------
def infer_output_format(path: str, explicit: Optional[str]) -> str:
    """Infer output format from explicit arg or file extension."""
    if explicit:
        return explicit

    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext == ".txt":
        return "txt"
    return "jsonl"


def write_predictions(pred: Dict[int, Dict[str, object]], path: str, fmt: str = "jsonl") -> None:
    """Write predictions to disk in the chosen format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    items = [
        {"idx": idx, "depth": v.get("depth"), "summary": v.get("summary")}
        for idx, v in sorted(pred.items(), key=lambda kv: kv[0])
    ]

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    elif fmt == "txt":
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(f"{obj['depth']}\n")
                f.write(f"{obj['summary']}\n\n")
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    print(f"[write_predictions] Wrote {len(items)} annotations to {path} (format={fmt})")


# ------------------------------
# CLI entry point
# ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate terminal events with depth + summary using a vLLM model.",
    )
    parser.add_argument("input_xml", help="Path to XML file containing <event> nodes")
    parser.add_argument("output_path", help="Output path for annotations")

    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json", "txt"],
        help="Output format (default inferred from extension)",
    )
    parser.add_argument(
        "--no-fewshots",
        action="store_true",
        help="Disable few-shot examples in the prompt",
    )
    parser.add_argument(
        "--k-target",
        type=int,
        default=K_TARGET,
        help="Number of target events per flush (default: %(default)s)",
    )
    parser.add_argument(
        "--n-neigh",
        type=int,
        default=N_NEIGH,
        help="Number of neighbor events to include as context (default: %(default)s)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="If set, only annotate the first N events (for quick tests)",
    )

    return parser.parse_args()


def main() -> None:
    global INCLUDE_FEWSHOTS_DEFAULT, K_TARGET, N_NEIGH

    args = parse_args()

    INCLUDE_FEWSHOTS_DEFAULT = not args.no_fewshots
    K_TARGET = args.k_target
    N_NEIGH = args.n_neigh

    print(f"[main] Loading events from {args.input_xml}")
    evs = load_events(args.input_xml)
    print(f"[main] Loaded {len(evs)} events")

    if args.max_events is not None:
        evs = evs[: args.max_events]
        print(f"[main] Truncated to first {len(evs)} events due to --max-events")

    if evs:
        print("[main] First event snippet:\n" + evs[0].xml[:300] + "...\n")

    preds = run_flushes(evs)

    fmt = infer_output_format(args.output_path, args.output_format)
    write_predictions(preds, args.output_path, fmt)


if __name__ == "__main__":
    main()
