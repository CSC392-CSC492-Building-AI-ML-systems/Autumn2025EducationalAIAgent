#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Model-1 streamed annotator

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import json

# ------------------------------
# Config
# ------------------------------
XML_PATH = "../../data/model_1/inputs/1727009412_parsed.xml"
GT_PATH = "../../data/model_1/outputs/1727009412_training.txt"

# Model settings
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # ONE reasoning model for everything
USE_INT4 = True
MAX_NEW_TOKENS = 8000  # Let it think as much as needed
SUMMARY_WORD_LIMIT = 50

# Prompt packaging
ADD_DONE_SENTINEL = False  # Not needed with thinking models

# Flush parameters
K_TARGET = 1
N_NEIGH = 20

INCLUDE_FEWSHOTS_DEFAULT = True
COLLAPSE_ECHO_INPUTS = False

# ------------------------------
# Statics
# ------------------------------
FEWSHOTS_BLOCK = """
EXAMPLES (for format/logic only — do not output these)

NOTE: Event XML often shows keystroke-by-keystroke input with echoed characters, not full commands.

DEPTH SEMANTICS:
- depth = -1: STARTING a new subtask (entering deeper level)
- depth = 0:  CONTINUING at same level (ongoing work)
- depth = +1: FINISHING a subtask (returning to parent level)

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A — Starting a new subtask (depth = -1)
Context: User is editing a config file and decides to spawn a shell to check logs
neighbor_tail:
  - id=10 depth=0  summary="Edit network monitoring config in vim"
  - id=11 depth=0  summary="Navigate to logging section of config"
currDepth before target: 0

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
  <system_output>[shell spawned from vim]</system_output>
</event>

Expected output:
{"annotation": "Spawn shell from vim editor to grep syslog for errors.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (shell within editor) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE B — Continuing at same level (depth = 0)
Context: User is working within the spawned shell, continuing their investigation
neighbor_tail:
  - id=20 depth=-1 summary="Spawn shell from vim to investigate logs"
  - id=21 depth=0  summary="Search syslog for error patterns"
currDepth before target: -1

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
  <system_output>-rw-r----- 1 syslog  adm  2.4M Oct 26 15:32 syslog</system_output>
</event>

Expected output:
{"annotation": "List log directory to verify syslog file exists.", "depth": 0}

Why depth = 0? User is CONTINUING work at the same level (still in spawned shell) - ongoing task.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE C — Finishing a subtask (depth = +1)
Context: User completes their work in the spawned shell and exits back to vim
neighbor_tail:
  - id=30 depth=-1 summary="Spawn shell from vim to check logs"
  - id=31 depth=0  summary="Examine syslog entries for errors"
  - id=32 depth=0  summary="Identify network timeout pattern in logs"
currDepth before target: -1

input xml:
<event>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>[returning to vim]</system_output>
  <system_output>demo@host:/etc/config.conf (modified)</system_output>
</event>

Expected output:
{"annotation": "Exit shell and return to vim editor.", "depth": 1}

Why depth = +1? User is FINISHING the shell subtask and returning to parent (vim) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE D — Continuing at top level (depth = 0)
Context: User is working in their main shell, typing a command
neighbor_tail:
  - id=40 depth=0  summary="Interactive bash shell at home directory"
  - id=41 depth=0  summary="List files in current directory"
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

Expected output:
{"annotation": "Initiate SSH connection to 10.0.7.138.", "depth": 0}

Why depth = 0? User is CONTINUING work at the main shell level - not entering or exiting.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE E — Starting SSH session subtask (depth = -1)
Context: After typing SSH command, user now authenticates and enters remote session
neighbor_tail:
  - id=50 depth=0  summary="Initiate SSH connection to 10.0.7.138"
currDepth before target: 0

input xml:
<event>
  <system_output>demo@10.0.7.138's password: </system_output>
  <user_input>[password entered]</user_input>
  <system_output>Welcome to Ubuntu 22.04.3 LTS</system_output>
  <system_output>Last login: Fri Oct 25 14:23:11 2025</system_output>
  <system_output>demo@remote-server:~$ </system_output>
</event>

Expected output:
{"annotation": "Authenticate and log into remote server via SSH.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (remote SSH session) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE F — Finishing SSH session (depth = +1)
Context: User completes work on remote server and exits SSH session
neighbor_tail:
  - id=60 depth=-1 summary="Log into remote server via SSH"
  - id=61 depth=0  summary="Install monitoring package on remote server"
  - id=62 depth=0  summary="Verify package installation completed"
currDepth before target: -1

input xml:
<event>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>logout</system_output>
  <system_output>Connection to 10.0.7.138 closed.</system_output>
  <system_output>demo@boxtop:~$ </system_output>
</event>

Expected output:
{"annotation": "Log out and close SSH connection to remote server.", "depth": 1}

Why depth = +1? User is FINISHING the SSH session subtask and returning to local shell - exiting level.

═══════════════════════════════════════════════════════════════════════════════

KEY PRINCIPLES:
1. Use depth = -1 when ENTERING a new tool/context (vim, SSH, subshell, pager, etc.)
2. Use depth = 0 when CONTINUING work in the current context
3. Use depth = +1 when EXITING/CLOSING a tool/context back to parent
4. Think of depth as task nesting: -1 = push onto stack, 0 = work at current level, +1 = pop from stack
""".strip()

SYSTEM_ROLE = """You are an expert terminal session annotator. Your goal is to generate concise summaries of user actions.
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
def collapse_echo(xml_str: str) -> str:
    if not COLLAPSE_ECHO_INPUTS:
        return xml_str

    root = etree.fromstring(xml_str)
    children = list(root)
    collapsed: List[etree.Element] = []
    i = 0

    while i < len(children):
        if i + 1 < len(children):
            c1, c2 = children[i], children[i + 1]
            if (
                c1.tag == "user_input"
                and c2.tag == "system_output"
                and (c1.text or "") == (c2.text or "")
            ):
                merged = etree.Element("echo")
                merged.text = c1.text
                collapsed.append(merged)
                i += 2
                continue
        collapsed.append(children[i])
        i += 1

    new_root = etree.Element(root.tag, attrib=root.attrib)
    new_root.extend(collapsed)
    return etree.tostring(new_root, encoding="unicode")


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
        xml_str = collapse_echo(xml_str)

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

    neighbor_info = []
    for i in neighbor_idxs:
        neighbor_info.append(f"- id={i} depth={get_dep(i)}  summary={get_sum(i)}")

    target_events = []
    for i in target_idxs:
        if 0 <= i < len(events):
            target_events.append(events[i].xml)

    currDepth = compute_curr_depth_upto(target_idxs[0])

    return {
        "target_idxs": target_idxs,
        "neighbor_info": neighbor_info,
        "target_events": target_events,
        "currDepth": currDepth,
    }


def build_instruction(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    # Build neighbor XML
    neighbor_items = []
    if pkg.get("neighbor_info"):
        for line in pkg["neighbor_info"]:
            match = re.match(r"- id=(\d+) depth=(-?\d+)\s+summary=(.+)", line)
            if match:
                nid, ndepth, nsummary = match.groups()
                neighbor_items.append({"id": nid, "depth": ndepth, "summary": nsummary})
    
    neighbors_xml = "\n".join(
        [f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
         for n in neighbor_items]
    ) or "    <neighbor>(none)</neighbor>"

    # Build target events XML
    target_items = []
    for idx, xml_str in zip(pkg["target_idxs"], pkg["target_events"]):
        target_items.append({"id": idx, "xml": xml_str})
    
    targets_xml = "\n".join(
        [f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in target_items]
    )

    examples_xml = f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if use_fewshots else ""

    extra = (
        "- do not copy xml tags or attributes; no repeated phrases\n"
        "- do not mention an address that was not explicitly mentioned in the event\n"
        "- if the target event contains an <annotation> tag or depth value ignore it\n"
        "- if there are no neighbors then the depth should be 0"
    )

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Use the <think>...</think> section to analyze what is happening in the event and assess whether the event starts a nested subtask (-1), continues at the same level (0), or exits one or more levels up (k).
- In <think>...</think>, generate a concise summary at a higher level, considering broader context.
- Use neighbors ONLY for continuity; do not invent context.
- Think carefully about BOTH the annotation AND the depth together.
</think_first>

<rules>
- the user's keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent (new task started), 0 for same level (still doing the same task), >0 to exit levels (ended one or multiple tasks)
- maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
- write action-oriented summaries; avoid "user", "they", "typed", "inputs", "enters a command
- depth is relative to the previous events and nothing else"
{extra}
</rules>{examples_xml}

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


def build_messages(instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": instruction},
    ]


# ------------------------------
# Model loading
# ------------------------------
def load_model_and_tokenizer():
    print(f"Loading model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if USE_INT4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    m.eval()
    return m, tok


# ------------------------------
# Generation
# ------------------------------
def generate_with_thinking(model, tok, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Generate with thinking model - NO format constraints.
    Returns: (full_output_with_thinking, extracted_json)
    """
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    # Get EOS tokens
    eos_ids = [tok.eos_token_id]
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != -1:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_ids = list({i for i in eos_ids if i is not None}) or None

    input_len = inputs.input_ids.shape[-1]

    # NO constraints - let it think freely
    out_ids = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_ids,
        use_cache=True,
        temperature=0.4,        # was 0.7
        top_p=0.7,             # was 0.85
        repetition_penalty=1.2, # was 1.15
        early_stopping=True,    
    )

    gen_ids = out_ids[0, input_len:]
    full_output = tok.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Extract JSON from output (after </think> if present)
    if "</think>" in full_output:
        json_part = full_output.split("</think>", 1)[1].strip()
    else:
        json_part = full_output
    
    return full_output, json_part


def parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]:
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    out: List[Tuple[int, str]] = []

    def add(obj):
        ann = obj.get("annotation")
        dep = obj.get("depth")
        
        if isinstance(ann, str):
            # Convert string depth to int if needed
            if isinstance(dep, str):
                try:
                    dep = int(dep)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert depth '{dep}' to int")
                    return
            
            if isinstance(dep, int) and dep >= -1:
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
    events = evs

    total = len(events)
    start_idx = 0

    model, tok = load_model_and_tokenizer()
    
    print("cuda_available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("MODEL:", MODEL_ID)
    print("NO FORMAT CONSTRAINTS - letting model think freely")

    all_flush_logs = []

    for upto in range(start_idx, total):
        pkg = make_flush_package(upto_idx=upto, K=K_TARGET, N=N_NEIGH)
        instr = build_instruction(pkg, use_fewshots=INCLUDE_FEWSHOTS_DEFAULT)
        messages = build_messages(instr)

        print("=" * 80)
        print(
            f"FLUSH upto event idx={upto} | currDepth(before)={pkg['currDepth']} | targets={pkg['target_idxs']}"
        )

        # Generate with thinking
        print("\n--- Model output (with thinking) ---")
        full_output, json_part = generate_with_thinking(model, tok, messages)
        print(full_output)

        # Parse JSON
        pairs = parse_depth_summary_pairs(json_part)
        if len(pairs) != len(pkg["target_idxs"]):
            print("\n(!) Output pairs != #targets; keeping whatever parsed.")

        all_flush_logs.append({
            "upto": upto,
            "targets": pkg["target_idxs"],
            "full_output": full_output,
            "json_part": json_part,
            "pairs": pairs,
        })

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # After-loop print
    print("\n" + "=" * 80)
    print("ALL MODEL OUTPUTS")
    print("=" * 80)
    for log in all_flush_logs:
        print(f"\n[FLUSH upto={log['upto']} targets={log['targets']}]")
        print(log["full_output"])

    # Final table
    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED TABLE")
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