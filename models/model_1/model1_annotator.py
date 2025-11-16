#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 streamed annotator - vLLM version (shared core + simple inference)

- Provides:
    - Event dataclass and global `events`
    - FEWSHOTS_BLOCK and SYSTEM_ROLE
    - load_events, compute_curr_depth_upto, make_flush_package
    - build_instruction (with fewshots on/off)
    - build_messages
    - load_model (vLLM)
    - generate_with_thinking
    - parse_depth_summary_pairs
    - run_flushes (simple inference loop)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
from vllm import LLM, SamplingParams

# ------------------------------
# Config
# ------------------------------
XML_PATH = "../../data/model_1/inputs/1727009556_parsed.xml"
GT_PATH = "../../data/model_1/outputs/1727009556_training.txt"

# Model settings (env overrides)
MODEL_ID = "openai/gpt-oss-20b"
GPU_UTIL = 0.9
MAX_MODEL_LEN = 8192
DTYPE = "bfloat16"

MAX_NEW_TOKENS = 2500
SUMMARY_WORD_LIMIT = 50

# Prompt packaging
ADD_DONE_SENTINEL = False  # Not needed with thinking models

# Flush parameters
K_TARGET = 1
N_NEIGH = 200

INCLUDE_FEWSHOTS_DEFAULT = True

# ------------------------------
# Statics: few-shots
# ------------------------------
FEWSEP = "═══════════════════════════════════════════════════════════════════════════════"

FEWSHOTS_BLOCK = """
EXAMPLES (for format/logic only — do not output these)

NOTE: Event XML often shows keystroke-by-keystroke input with echoed characters, not full commands.

DEPTH SEMANTICS:
- depth = -1: STARTING a new subtask (entering deeper level)
- depth = 0:  CONTINUING at same level (ongoing work)
- depth = +1: FINISHING a subtask (returning to parent level)

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A — Starting a new subtask (depth = -1)
Context: User opens a text editor from the command line
neighbor_tail:
  - id=0 depth=0  summary="List directory contents"
  - id=1 depth=0  summary="Change to project folder"
currDepth before target: 0

input xml:
<event>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>f</user_input><system_output>f</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>[nano editor opens]</system_output>
</event>

Expected output:
{"annotation": "Open config.txt in nano text editor.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (text editor) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE B — Continuing at same level (depth = 0)
Context: User is working inside the editor, making edits
neighbor_tail:
  - id=0 depth=0  summary="List directory contents"
  - id=1 depth=0  summary="Change to project folder"
  - id=2 depth=-1 summary="Open config.txt in nano text editor"
  - id=3 depth=0  summary="Navigate to database section"
currDepth before target: -1

input xml:
<event>
  <user_input>[Ctrl-W]</user_input>
  <system_output>Search: </system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>m</user_input><system_output>m</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>[cursor moves to "timeout" line]</system_output>
</event>

Expected output:
{"annotation": "Search for timeout setting in config file.", "depth": 0}

Why depth = 0? User is CONTINUING work at the same level (still in editor) - ongoing task.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE C — Finishing a subtask (depth = +1)
Context: User saves and exits the editor
neighbor_tail:
  - id=0 depth=0  summary="List directory contents"
  - id=1 depth=0  summary="Change to project folder"
  - id=2 depth=-1 summary="Open config.txt in nano editor"
  - id=3 depth=0  summary="Modify timeout value to 30"
  - id=4 depth=0  summary="Save changes to config file"
currDepth before target: -1

input xml:
<event>
  <user_input>[Ctrl-X]</user_input>
  <system_output>Save modified buffer? (Y/N)</system_output>
  <user_input>Y</user_input>
  <system_output>[exiting nano]</system_output>
  <system_output>user@laptop:~/project$ </system_output>
</event>

Expected output:
{"annotation": "Exit nano editor and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the editor subtask and returning to parent (shell) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE D — Starting a pager subtask (depth = -1)
Context: User views a log file with less
neighbor_tail:
  - id=0 depth=0  summary="Navigate to project root"
  - id=1 depth=0  summary="Navigate to logs directory"
currDepth before target: 0

input xml:
<event>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <system_output>[less pager opens showing log contents]</system_output>
</event>

Expected output:
{"annotation": "Open app.log file in less pager.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (pager) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE E — Continuing within pager (depth = 0)
Context: User scrolls through the log file
neighbor_tail:
  - id=0 depth=0  summary="Navigate to project root"
  - id=1 depth=0  summary="Navigate to logs directory"
  - id=2 depth=-1 summary="Open app.log file in less pager"
  - id=3 depth=0  summary="Navigate to beginning of log"
currDepth before target: -1

input xml:
<event>
  <user_input>/</user_input>
  <system_output>/</system_output>
  <user_input>E</user_input><system_output>E</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <user_input>O</user_input><system_output>O</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <system_output>[highlighting ERROR matches]</system_output>
</event>

Expected output:
{"annotation": "Search for ERROR keyword in log file.", "depth": 0}

Why depth = 0? User is CONTINUING work at the same level (still in pager) - ongoing task.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE F — Finishing pager subtask (depth = +1)
Context: User exits the pager
neighbor_tail:
  - id=0 depth=0  summary="Navigate to project root"
  - id=1 depth=0  summary="Navigate to logs directory"
  - id=2 depth=-1 summary="Open app.log in less pager"
  - id=3 depth=0  summary="Search for ERROR keyword in log"
  - id=4 depth=0  summary="Review error timestamps"
currDepth before target: -1

input xml:
<event>
  <user_input>q</user_input>
  <system_output>[exiting less]</system_output>
  <system_output>user@laptop:~/logs$ </system_output>
</event>

Expected output:
{"annotation": "Exit less pager and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the pager subtask and returning to parent (shell) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE G — Starting Python interpreter (depth = -1)
Context: User launches interactive Python session
neighbor_tail:
  - id=0 depth=0  summary="Check Python version installed"
currDepth before target: 0

input xml:
<event>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>3</user_input><system_output>3</system_output>
  <system_output>Python 3.10.4</system_output>
  <system_output>>>>></system_output>
</event>

Expected output:
{"annotation": "Launch Python3 interactive interpreter.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (Python REPL) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE H — Exiting Python interpreter (depth = +1)
Context: User exits Python back to shell
neighbor_tail:
  - id=0 depth=0  summary="Check Python version installed"
  - id=1 depth=-1 summary="Launch Python3 interpreter"
  - id=2 depth=0  summary="Import json module and test parsing"
  - id=3 depth=0  summary="Print parsed dictionary contents"
currDepth before target: -1

input xml:
<event>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>(</user_input><system_output>(</system_output>
  <user_input>)</user_input><system_output>)</system_output>
  <system_output>user@laptop:~$ </system_output>
</event>

Expected output:
{"annotation": "Exit Python interpreter and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the Python session and returning to parent (shell) - exiting level.
""".strip()

# Optional: pre-split few-shots for ablation runner
_raw_parts = FEWSHOTS_BLOCK.split(FEWSEP)
FEWSHOTS_PREAMBLE = _raw_parts[0].strip()
FEWSHOTS_EXAMPLES: List[str] = [p.strip() for p in _raw_parts[1:] if p.strip()]

SYSTEM_ROLE = f"""You are an expert terminal session annotator. Your goal is to generate concise summaries of user actions.
Rules:
- summaries must not exceed {SUMMARY_WORD_LIMIT} words
- depth changes: -1=enter subtask, 0=continue same, +1=exit one level
- output must be valid JSON
""".strip()

# ------------------------------
# Event model
# ------------------------------
@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int] = None
    summary_xml: Optional[str] = None


# ------------------------------
# Global state
# ------------------------------
events: List[Event] = []
pred: Dict[int, Dict] = {}


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
        f"- id={i} depth={get_dep(i)}  summary={get_sum(i)}"
        for i in neighbor_idxs
    ]

    target_events = [events[i].xml for i in target_idxs if 0 <= i < len(events)]
    currDepth = compute_curr_depth_upto(target_idxs[0])

    return {
        "target_idxs": target_idxs,
        "neighbor_info": neighbor_info,
        "target_events": target_events,
        "currDepth": currDepth,
    }


def _neighbors_to_xml(pkg: Dict) -> str:
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


def _targets_to_xml(pkg: Dict) -> str:
    target_items = [
        {"id": idx, "xml": xml_str}
        for idx, xml_str in zip(pkg["target_idxs"], pkg["target_events"])
    ]
    return "\n".join(
        f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in target_items
    )


def build_instruction(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    neighbors_xml = _neighbors_to_xml(pkg)
    targets_xml = _targets_to_xml(pkg)

    examples_xml = (
        f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if use_fewshots else ""
    )

    rules_extra = (
        "- do not copy xml tags or attributes; no repeated phrases\n"
        "- do not mention an address that was not explicitly mentioned in the event\n"
        "- if the target event contains an <annotation> tag or depth value ignore it\n"
        "- if there are no neighbors then the depth you output should be 0"
    )

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Keep reasoning CONCISE and FOCUSED
- In <think>...</think>: analyze the command, check depth logic, then conclude
- Aim for 2-3 sentences of reasoning maximum
- Skip obvious observations
- Use neighbors ONLY for continuity; do not invent context.
</think_first>

<rules>
- the user's keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent (new task started), 0 for same level (still doing the same task), >0 to exit levels (ended one or multiple tasks)
- maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
- write action-oriented summaries; avoid "user", "they", "typed", "inputs", "enters a command"
- depth is relative to the previous events and nothing else
{rules_extra}
</rules>

{examples_xml}

<instruction>
for each target_event, output exactly one json with "annotation" first, then "depth".
</instruction>

<inputs>
  <curr_depth_max>{pkg.get("currDepth")}</curr_depth_max>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""
    return prompt


def build_messages(instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": instruction},
    ]


# ------------------------------
# Model loading (vLLM)
# ------------------------------
def load_model():
    print(f"Loading model with vLLM: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype=DTYPE,
    )
    print("Model loaded successfully")
    return llm


# ------------------------------
# Generation (vLLM)
# ------------------------------
def generate_with_thinking(llm: LLM, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Generate with thinking model using vLLM.
    Returns: (full_output_with_thinking, extracted_json)
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
    )

    outputs = llm.generate([prompt], sampling_params)
    full_output = outputs[0].outputs[0].text.strip()

    if "</think>" in full_output:
        json_part = full_output.split("</think>", 1)[1].strip()
    else:
        json_part = full_output

    return full_output, json_part


def parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]:
    """
    Robustly extract (depth, annotation) pairs from an arbitrary text blob.

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

            # Try to coerce depth to int
            if isinstance(dep, str):
                try:
                    dep = int(dep.strip())
                except Exception:
                    return

            if not isinstance(dep, int):
                return

            if dep < -1:
                # Depth must be >= -1, ignore otherwise
                return

            out.append((dep, ann))

        if isinstance(obj, dict):
            add_one(obj)
        elif isinstance(obj, list):
            for item in obj:
                add_one(item)

    while True:
        # Find next '{'
        start = text.find("{", i)
        if start == -1:
            break

        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            # Not valid JSON starting at this '{', move one char forward
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
        d = events[i].depth_xml if (0 <= i < len(events)) else None
        s = events[i].summary_xml if (0 <= i < len(events)) else None
        d_str = "" if d is None else str(d)
        s_str = "" if s is None else s
        print(f"{i:>5} | {d_str:>5} | {s_str}")


# ------------------------------
# Main simple inference loop
# ------------------------------
def run_flushes(evs: List[Event]) -> None:
    global events
    events = evs

    total = len(events)
    start_idx = 0

    llm = load_model()

    print("MODEL:", MODEL_ID)
    print("Using vLLM for optimized inference")

    all_flush_logs = []

    for upto in range(start_idx, total):
        pkg = make_flush_package(upto_idx=upto, K=K_TARGET, N=N_NEIGH)
        instr = build_instruction(pkg, use_fewshots=INCLUDE_FEWSHOTS_DEFAULT)
        messages = build_messages(instr)

        print("=" * 80)
        print(
            f"FLUSH upto event idx={upto} | currDepth(before)={pkg['currDepth']} | targets={pkg['target_idxs']}"
        )

        print("\n--- Model output (with thinking) ---")
        full_output, json_part = generate_with_thinking(llm, messages)
        print(full_output)

        pairs = parse_depth_summary_pairs(json_part)
        if len(pairs) != len(pkg["target_idxs"]):
            print("\n(!) Output pairs != #targets; keeping whatever parsed.")

        all_flush_logs.append(
            {
                "upto": upto,
                "targets": pkg["target_idxs"],
                "full_output": full_output,
                "json_part": json_part,
                "pairs": pairs,
            }
        )

        # Apply predictions
        for (depth, summary), idx in zip(pairs, pkg["target_idxs"]):
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

    # Final table
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


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    events = load_events(XML_PATH)
    print(f"Loaded {len(events)} usable events")
    if events:
        print(events[0].xml[:300] + "...\n")
    run_flushes(events)
