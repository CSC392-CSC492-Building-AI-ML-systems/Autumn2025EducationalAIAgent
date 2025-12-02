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

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
from vllm import LLM, SamplingParams

from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------
# Config
# ------------------------------
XML_PATH = "../../data/model_1/inputs/renee_rec2_parsed.xml"
GT_PATH = "../../data/model_1/outputs/renee_rec2_training.txt"

# Model settings (env overrides)
MODEL_ID = "openai/gpt-oss-20b"
GPU_UTIL = 0.9
MAX_MODEL_LEN = 131072
DTYPE = "bfloat16"

MAX_NEW_TOKENS = 2500
SUMMARY_WORD_LIMIT = 50

# Parallelism / memory controls (env overridable)
TP_SIZE = int(os.getenv("VLLM_TP", "1"))


# Global singleton LLM
_GLOBAL_LLM: Optional[LLM] = None


# Flush parameters
K_TARGET = 1
N_NEIGH = 200

INCLUDE_FEWSHOTS_DEFAULT = True

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

# ------------------------------
# Few-shot examples - streamlined
# ------------------------------
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
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>e</user_input><system_output>e</system_output>
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
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
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
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>m</user_input><system_output>m</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>f</user_input><system_output>f</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>m</user_input><system_output>m</system_output>
  <user_input>l</user_input><system_output>l</system_output>
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
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>w</user_input><system_output>w</system_output>
  <user_input>q</user_input><system_output>q</system_output>
  <system_output>config.yaml written</system_output>
  <system_output>(venv) $</system_output>
</event>

output:
{"annotation": "Save config changes and exit vim", "depth": 1}

Why: Editor task done, back to environment setup level.
""".strip()

# Pre-split for ablation if needed
_raw_parts = FEWSHOTS_BLOCK.split(FEWSEP)
FEWSHOTS_PREAMBLE = _raw_parts[0].strip()
FEWSHOTS_EXAMPLES = [p.strip() for p in _raw_parts[1:] if p.strip()]

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
# Ground-truth loading
# ------------------------------
def load_gt_annotations(gt_path: str) -> Dict[int, Dict[str, object]]:
    """
    Load GT (depth, summary) pairs from a text file of the form:

        0
        User connects ...
        -1
        User attempts ...
        0
        User tries ...

    i.e. depth on one line, summary on the next, repeated.
    Returns: {idx: {"depth": int, "summary": str}}
    """
    gt: Dict[int, Dict[str, object]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    buf_depth: Optional[int] = None
    idx = 0
    for line in lines:
        if not line.strip():
            # skip empty lines
            continue

        if buf_depth is None:
            # expecting a depth
            try:
                buf_depth = int(line.strip())
            except ValueError:
                raise ValueError(f"Expected depth int, got: {line!r}")
        else:
            # this line is the summary corresponding to buf_depth
            summary = line.strip()
            gt[idx] = {"depth": buf_depth, "summary": summary}
            buf_depth = None
            idx += 1

    if buf_depth is not None:
        print("[WARN] GT file ended with a depth but no summary; ignoring last depth.")

    return gt


# ------------------------------
# Embedding + ROUGE-L + Cross-Encoder + BERTScore scoring
# ------------------------------
def score_annotations_with_embeddings(
    pred: Dict[int, Dict[str, object]],
    gt_path: str,
    bi_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_name: str = "cross-encoder/stsb-roberta-base",
) -> None:
    """
    Compute multiple similarity metrics between GT and model summaries:

    - Bi-encoder cosine similarity (sentence embeddings)
    - ROUGE-L F1 overlap
    - Cross-encoder STS similarity (option A)
    - BERTScore F1 (option B)

    Args:
        pred: {idx: {"depth": int, "summary": str}}
        gt_path: path to GT .txt file in (depth, summary) alternating format
        bi_encoder_name: sentence-transformers bi-encoder model for cosine sim
        cross_encoder_name: sentence-transformers cross-encoder STS model
    """
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rouge_score import rouge_scorer
    from bert_score import score as bertscore_score
    import numpy as np

    gt = load_gt_annotations(gt_path)

    # Intersection of indices that have both GT and pred summaries
    common_idxs = sorted(set(gt.keys()) & set(pred.keys()))
    if not common_idxs:
        print("[SIM] No overlapping indices between GT and predictions; cannot score.")
        return

    gt_summaries = []
    pred_summaries = []
    for idx in common_idxs:
        gt_sum = str(gt[idx]["summary"])
        pred_sum = str(pred[idx].get("summary", "") or "")
        gt_summaries.append(gt_sum)
        pred_summaries.append(pred_sum)

    print(f"[SIM] Using {len(common_idxs)} matched events for similarity scoring.")

    # ---------------- Bi-encoder cosine similarity ----------------
    bi_model = SentenceTransformer(bi_encoder_name)
    gt_emb = bi_model.encode(
        gt_summaries,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    pred_emb = bi_model.encode(
        pred_summaries,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    sims = (gt_emb * pred_emb).sum(dim=-1)  # shape: (N,)
    sims_np = sims.detach().cpu().numpy()

    mean_sim = float(sims_np.mean())
    median_sim = float(np.median(sims_np))
    p25_sim, p75_sim = np.percentile(sims_np, [25, 75])

    print("\n[Cosine] Bi-encoder annotation similarity (cosine)")
    print(f"  mean   : {mean_sim:.4f}")
    print(f"  median : {median_sim:.4f}")
    print(f"  p25    : {p25_sim:.4f}")
    print(f"  p75    : {p75_sim:.4f}")

    # ---------------- ROUGE-L F1 ----------------
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []
    for ref, hyp in zip(gt_summaries, pred_summaries):
        score = rouge.score(ref, hyp)["rougeL"].fmeasure
        rouge_l_scores.append(score)

    rouge_l_scores = np.array(rouge_l_scores, dtype=float)
    mean_rouge = float(rouge_l_scores.mean())
    median_rouge = float(np.median(rouge_l_scores))
    p25_rouge, p75_rouge = np.percentile(rouge_l_scores, [25, 75])

    print("\n[ROUGE-L] Annotation overlap (ROUGE-L F1)")
    print(f"  mean   : {mean_rouge:.4f}")
    print(f"  median : {median_rouge:.4f}")
    print(f"  p25    : {p25_rouge:.4f}")
    print(f"  p75    : {p75_rouge:.4f}")

    # ---------------- Option A: Cross-encoder STS similarity ----------------
    # CrossEncoder takes pairs and outputs a similarity score (typically ~0–1)
    cross_model = CrossEncoder(cross_encoder_name)
    pair_inputs = list(zip(gt_summaries, pred_summaries))
    cross_scores = cross_model.predict(pair_inputs)
    cross_scores_np = np.array(cross_scores, dtype=float)

    mean_cross = float(cross_scores_np.mean())
    median_cross = float(np.median(cross_scores_np))
    p25_cross, p75_cross = np.percentile(cross_scores_np, [25, 75])

    print("\n[Cross-Encoder] STS similarity")
    print(f"  mean   : {mean_cross:.4f}")
    print(f"  median : {median_cross:.4f}")
    print(f"  p25    : {p25_cross:.4f}")
    print(f"  p75    : {p75_cross:.4f}")

    # ---------------- Option B: BERTScore F1 ----------------
    # Note: order is (cands, refs) = (pred, gt)
    P, R, F1 = bertscore_score(
        pred_summaries,
        gt_summaries,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )
    bert_f1_np = F1.detach().cpu().numpy()

    mean_bert = float(bert_f1_np.mean())
    median_bert = float(np.median(bert_f1_np))
    p25_bert, p75_bert = np.percentile(bert_f1_np, [25, 75])

    print("\n[BERTScore] Annotation similarity (F1)")
    print(f"  mean   : {mean_bert:.4f}")
    print(f"  median : {median_bert:.4f}")
    print(f"  p25    : {p25_bert:.4f}")
    print(f"  p75    : {p75_bert:.4f}")

    # ---------------- Sample table ----------------
    print("\n[SIM] Sample per-event scores (first 10):")
    header = (
        f"{'idx':>5} | {'cos':>6} | {'rougeL':>7} | "
        f"{'cross':>6} | {'bertF1':>7} | {'gt_summary':<40} | model_summary"
    )
    print(header)
    print("-" * len(header))
    for i, idx in enumerate(common_idxs[:10]):
        cos_val = sims_np[i]
        rouge_val = rouge_l_scores[i]
        cross_val = cross_scores_np[i]
        bert_val = bert_f1_np[i]
        gt_sum = gt_summaries[i].replace("\n", " ")[:40]
        pred_sum = pred_summaries[i].replace("\n", " ")[:60]
        print(
            f"{idx:5d} | {cos_val:6.4f} | {rouge_val:7.4f} | "
            f"{cross_val:6.4f} | {bert_val:7.4f} | {gt_sum:<40} | {pred_sum}"
        )

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


def build_instruction(pkg: dict, use_fewshots: bool = True) -> str:
    """Build the prompt with minimal but effective structure."""
    
    # Format neighbors
    neighbors_xml = _format_neighbors(pkg)
    
    # Format targets
    targets_xml = _format_targets(pkg)
    
    # Include examples if requested
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
{examples_section}
<instruction>
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


def _format_neighbors(pkg: dict) -> str:
    """Format neighbor information as XML."""
    import re
    
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
        f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' 
        for t in target_items
    )

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
        tensor_parallel_size=TP_SIZE,        # <--- key line: shard across GPUs
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
    """
    Generate with thinking model using vLLM.
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

    # vLLM's RequestOutput
    req_out = outputs[0]
    first_out = req_out.outputs[0]

    full_output = first_out.text.strip()

    # Token counts
    prompt_tokens = len(req_out.prompt_token_ids) if hasattr(req_out, "prompt_token_ids") else 0
    generated_tokens = len(first_out.token_ids) if hasattr(first_out, "token_ids") else 0

    if "</think>" in full_output:
        json_part = full_output.split("</think>", 1)[1].strip()
    else:
        json_part = full_output

    return full_output, json_part, prompt_tokens, generated_tokens



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
        full_output, json_part, prompt_tokens, gen_tokens = generate_with_thinking(llm, messages)
        print(full_output)

        total_tokens = prompt_tokens + gen_tokens
        print(f"\n[Tokens] prompt={prompt_tokens} | generated={gen_tokens} | total={total_tokens}")

        pairs = parse_depth_summary_pairs(json_part)


        # Drop obvious placeholder annotations
        pairs = [
            (depth, ann)
            for (depth, ann) in pairs
            if ann is not None and ann.strip() not in ("...", '"..."') and len(ann.strip()) >= 5
        ]

        if len(pairs) > len(pkg["target_idxs"]):
            pairs = pairs[-len(pkg["target_idxs"]):]

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
            # If there are no neighbors, force depth = 0 regardless of what the model said
            if not pkg["neighbor_info"]:
                depth = 0
            else:
                # Normal depth constraints
                if depth < -1:
                    depth = -1

                live_curr = compute_curr_depth_upto(idx)
                temp_curr = live_curr
                if depth == -1:
                    temp_curr -= 1
                elif depth > 0:
                    temp_curr += depth

                # Enforce stack invariant: currDepth must never go above 0
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
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Model 1 inference with optional ground truth evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths from script
  python model1_annotator.py

  # Override XML path only
  python model1_annotator.py --xml-path path/to/input.xml

  # Override both XML and GT paths
  python model1_annotator.py --xml-path path/to/input.xml --gt-path path/to/gt.txt

  # Skip GT evaluation even if GT path exists
  python model1_annotator.py --xml-path path/to/input.xml --no-eval
        """
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Path to XML file containing <event> nodes",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Path to ground truth .txt file for evaluation",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip ground truth evaluation even if GT path is provided",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override paths if provided via command line
    xml_path = args.xml_path if args.xml_path is not None else XML_PATH
    gt_path = args.gt_path if args.gt_path is not None else GT_PATH
    
    # Update module-level constants for compatibility with other scripts
    # (No global keyword needed since we're at module scope)
    XML_PATH = xml_path
    GT_PATH = gt_path
    
    events = load_events(XML_PATH)
    print(f"Loaded {len(events)} usable events")
    if events:
        print(events[0].xml[:300] + "...\n")
    run_flushes(events)

    # Evaluate against ground truth if GT path exists and evaluation is not disabled
    if not args.no_eval and os.path.exists(GT_PATH):
        print("\n" + "=" * 80)
        print("EMBEDDING-BASED SIMILARITY BETWEEN GT AND MODEL ANNOTATIONS")
        print("=" * 80)
        score_annotations_with_embeddings(pred, GT_PATH)
    elif args.gt_path and not os.path.exists(GT_PATH):
        print(f"[WARN] GT_PATH does not exist: {GT_PATH}")
    elif not args.no_eval:
        print(f"[INFO] No ground truth evaluation (GT_PATH not set or does not exist)")
