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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

# ------------------------------
# Config
# ------------------------------
XML_PATH = "../../data/model_1/inputs/first_julia_rec_parsed.xml"
GT_PATH = "../../data/model_1/outputs/first_julia_rec_training.txt"  # (unused here, kept for future)

# Model settings
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # instruction-tuned
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

# ------------------------------
# Statics
# ------------------------------
EXAMPLE_NEIGHBORS_TEXT = """- id=20 depth=-1 summary=Enter editing /etc/ntopng/ntopng.conf to tweak capture settings.
- id=21 depth=0  summary=Scroll within the editor; reviewing options.
- id=22 depth=-1 summary=Open a shell from editor to list /var/log for recent errors.
- id=23 depth=1  summary=Return from the shell to the editor context.
- id=24 depth=0  summary=Save the config and keep editing."""

FEWSHOTS_BLOCK = f"""
EXAMPLES (FOR FORMAT/LOGIC ONLY — DO NOT OUTPUT THESE IN YOUR ANSWER)

Example neighbor_tail (dense):
{EXAMPLE_NEIGHBORS_TEXT}

Example A — NEW nested subtask (depth = -1)
Example currDepth before target: -1
Example INPUT XML:
<event><user_input>:!grep -i error /var/log/syslog</user_input><system_output>[shell] matching lines...</system_output></event>
Example OUTPUT (two lines):
Spawn shell from editor to grep syslog for errors.
-1
EXAMPLE RATIONALE (do not output):
Action starts a nested tool (shell) inside the editor workflow; that descends one level → depth = -1.

Example B — Same-level continuation (depth = 0)
Example currDepth before target: -2
Example INPUT XML:
<event><user_input>less /var/log/syslog</user_input><system_output>--- syslog ---</system_output></event>
Example OUTPUT (two lines):
View syslog content within the spawned shell.
0
EXAMPLE RATIONALE (do not output):
Still operating in the same nested shell context (no new subtask, no exit); remain at current level → depth = 0.

Example C — Exit one level (depth = +1)
Example currDepth before target: -1
Example INPUT XML:
<event><user_input>:wq</user_input><system_output>[wrote config] demo@host:/etc/ntopng$ </system_output></event>
Example OUTPUT (two lines):
Save changes and exit the editor back to the shell.
1
EXAMPLE RATIONALE (do not output):
Exiting the editor returns to the parent shell context, popping one level → depth = +1.

Example D — Same-level command (depth = 0)
Example currDepth before target: 0
Example INPUT XML:
<event>  <system_output timestamp="0.096022">[?2004h]0;demo@boxtop: ~demo@boxtop:~$ </system_output>  <user_input timestamp="9.163614">s</user_input>  <system_output timestamp="9.164051">s</system_output>  <user_input timestamp="9.365744">s</user_input>  <system_output timestamp="9.366263">s</system_output>  <user_input timestamp="9.589844">h</user_input>  <system_output timestamp="9.59026">h</system_output>  <user_input timestamp="9.708352"> </user_input>  <system_output timestamp="9.708844"> </system_output>  <user_input timestamp="10.1118">1</user_input>  <system_output timestamp="10.112236">1</system_output>  <user_input timestamp="10.270878">0</user_input>  <system_output timestamp="10.271223">0</system_output>  <user_input timestamp="10.471565">.</user_input>  <system_output timestamp="10.471898">.</system_output>  <user_input timestamp="10.594981">0</user_input>  <system_output timestamp="10.595383">0</system_output>  <user_input timestamp="10.757499">.</user_input>  <system_output timestamp="10.757882">.</system_output>  <user_input timestamp="11.140897">7</user_input>  <system_output timestamp="11.14119">7</system_output>  <user_input timestamp="11.603706">.</user_input>  <system_output timestamp="11.604019">.</system_output>  <user_input timestamp="12.330584">1</user_input>  <system_output timestamp="12.331455">1</system_output>  <user_input timestamp="12.632256">3</user_input>  <system_output timestamp="12.633323">3</system_output>  <user_input timestamp="13.446626">8</user_input>  <system_output timestamp="13.447562">8</system_output>  <user_input timestamp="14.510021"></user_input>  <system_output timestamp="14.511984">[?2004l</system_output></event>
Example OUTPUT (two lines):
User initiates SSH connection to server 10.0.7.138
0
EXAMPLE RATIONALE (do not output):
Typing an ssh command within the same shell does not enter a nested tool yet; it stays at the current level → depth = 0.
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
    # Neighbors summary (real context)
    neigh_lines = []
    for n in pkg["neighbor_tail"]:
        neigh_lines.append(f"- id={n['id']} depth={n['depth']} summary={n['summary']}")
    neighbors_text = "\n".join(neigh_lines) if neigh_lines else "(none)"

    # Targets block
    targets_block = "\n".join(
        [f"\n<BEGIN_TARGET id={t['id']}>\n{t['xml']}\n<END_TARGET>\n" for t in pkg["target_events"]]
    )

    # Output constraints
    extra = (
        f"\n- Output EXACTLY TWO LINES per target. "
        f"- Line 1 (summary) MUST be ≤ {SUMMARY_WORD_LIMIT} words. Be concise and factual.\n"
        "- Line 2 MUST be a single integer depth (−1, 0, or >0). No extra text.\n"
        "- Do NOT copy XML tags/attributes. No repeated phrases.\n"
    )

    # Core instructions (without examples)
    core = f"""
You are Model-1 (annotator).

Given:
- currDepth (≤ 0): {pkg['currDepth']}
- neighbor_tail (already annotated, for context; may be empty):
{neighbors_text}
- TARGET_EVENT: A single event's XML.

THINK FIRST (hidden):
- Think inside <think>...</think> about what is happening in the event and whether the event starts a nested subtask (→ -1),
  continues at the same level (→ 0), or exits up (→ k).
- Think inside <think>...</think> to understand what is happening at a higher level to generate a summary
- Use neighbors ONLY for continuity; do not invent. Keep the <think> section compact.

Then OUTPUT (exactly two lines; order matters):
1) one-sentence annotation of what happened in the target event (≤ {SUMMARY_WORD_LIMIT} words)
2) a single integer 'depth' (use -1 for subevent, 0 for same level, >0 to exit levels)
{extra}

Rules:
- Output ONLY the two lines (and the final sentinel if required). No numbering, no JSON, no prose.
- Respect the stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth.
- Never let the running currDepth become > 0.
- Do not simply say what the user is typing or that they are typing something, we want to know what they are doing and the reason behind it.
- Write action-oriented summaries. Do NOT mention “user”, “they”, “typed/typing”, “by typing”, “inputs”, or “enters a command”.
- Start with a verb that describes the action’s intent (e.g., “List…”, “Open…”, “Initiate…”, “Install…”, “Exit…”).
""".strip()

    # Conditionally include examples
    examples_part = f"\n\n{FEWSHOTS_BLOCK}\n" if use_fewshots else ""

    instructions = (
        core + examples_part + "\n\nNow produce the pairs for the targets below:\n" + targets_block
    )
    return instructions




def build_messages(user_content: str) -> List[Dict[str, str]]:
    """Build chat messages for the Qwen3 instruct template."""
    return [
        {"role": "system", "content": "You are Model-1. Follow formatting strictly."},
        {"role": "user", "content": user_content},
    ]

# ------------------------------
# Logit Processors
# ------------------------------

class TwoLineFormatProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        import torch
        self.tok = tokenizer
        self.state = "line1"
        # IDs for special chars
        self.id_nl = self.tok.convert_tokens_to_ids("\n")
        self.id_minus = self.tok.convert_tokens_to_ids("-")
        # digits might be multiple tokens; we collect all that encode as digits
        self.digit_ids = set(self.tok.encode("0123456789", add_special_tokens=False))
        self.torch = torch  # avoid re-import in __call__

    def _mask_to(self, scores, allowed_ids):
        mask = self.torch.full_like(scores, float("-inf"))
        if allowed_ids:
            idxs = list(allowed_ids)
            mask[:, idxs] = scores[:, idxs]
        return mask

    def __call__(self, input_ids, scores):
        last = input_ids[0, -1].item()

        # if we just saw a newline, move from line1 -> waiting_linebreak
        if self.state == "line1":
            if last == self.id_nl:
                self.state = "waiting_linebreak"
            return scores

        # first token of line 2: must be '-' or digit (or newline to end early)
        if self.state == "waiting_linebreak":
            allowed = set([self.id_nl]) | self.digit_ids
            if isinstance(self.id_minus, int) and self.id_minus != -1:
                allowed.add(self.id_minus)
            self.state = "line2"
            return self._mask_to(scores, allowed)

        # subsequent tokens of line 2: allow '-', digits, newline
        if self.state == "line2":
            allowed = set([self.id_nl]) | self.digit_ids
            if isinstance(self.id_minus, int) and self.id_minus != -1:
                allowed.add(self.id_minus)
            return self._mask_to(scores, allowed)

        return scores

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

    m1 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )

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
def generate_pairs(m1: AutoModelForCausalLM, tok: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """Generate assistant-only text using chat template and slice off the prompt tokens.
    Fixes the echoing of 'system/user' you observed.
    """
    # Build input ids directly from chat template
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(m1.device)

    # Handle multiple EOS tokens (Qwen uses <|im_end|> in addition to eos)
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != -1:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_ids = list({i for i in eos_ids if i is not None}) or None

    input_len = inputs.shape[-1]

    processors = LogitsProcessorList([TwoLineFormatProcessor(tok)])
    out_ids = m1.generate(
        input_ids=inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,               # greedy → stable formatting
        num_beams=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_ids,
        logits_processor=processors,
        use_cache=True,
    )

    # Slice to only new tokens and decode
    gen_ids = out_ids[0, input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if ADD_DONE_SENTINEL and DONE_SENTINEL in text:
        text = text.split(DONE_SENTINEL, 1)[0].strip()
    return text



def parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]:
    """Parse alternating lines: depth (int), then summary (string). Forgiving about blank lines."""
    lines = [ln.strip() for ln in text.splitlines()]
    pairs: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        # non-empty line is the summary
        while i < len(lines) and lines[i] == "":
            i += 1
        if i >= len(lines):
            break
        summary = lines[i]

        # seek next integer line
        while i < len(lines) and not re.fullmatch(r"-?\d+", lines[i]):
            i += 1
        if i >= len(lines):
            break
        depth = int(lines[i]); i += 1

        pairs.append((depth, summary))
        i += 1
    return pairs


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
        raw = generate_pairs(m1, tok, messages)
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
