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
- depth = -1: STARTING a new subtask (a multi-step goal, often spanning several events)
- depth = 0:  CONTINUING at the same level (still inside the current subtask)
- depth > 0:  FINISHING one or more subtasks and returning toward the parent level

A “subtask” is not only an editor or tool; it can also be a logical unit of work
like “create a backup”, “run and debug tests”, or “set up a data pipeline”.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A — Starting a backup subtask (depth = -1)
Context: User has been exploring a project folder and decides to create a backup archive.

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
currDepth before target: 0

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
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>d</user_input><system_output>d</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <system_output>Compressing files into backup.tar...</system_output>
</event>

Expected output:
{"annotation": "Create a compressed backup archive of the source data.", "depth": -1}

Why depth = -1? The user is beginning a multi-step backup subtask that will likely
include verifying and possibly moving the archive.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE B — Continuing the backup subtask (depth = 0)
Context: After creating the archive, the user verifies the file and checks its size.

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
  - id=2 depth=-1 summary="Create a compressed backup archive of the source data"
currDepth before target: -1

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
  <system_output>-rw-r--r--  1 user  staff  42M backup.tar</system_output>
</event>

Expected output:
{"annotation": "Verify the newly created backup archive and inspect its size.", "depth": 0}

Why depth = 0? The user is still in the same backup subtask, checking the result
of the archive they just created.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE C — Finishing the backup subtask (depth = +1)
Context: The user moves the backup to a separate folder and marks the operation as done.

neighbor_tail:
  - id=0 depth=0  summary="List project directory contents"
  - id=1 depth=0  summary="Inspect size of source and data folders"
  - id=2 depth=-1 summary="Create a compressed backup archive of the source data"
  - id=3 depth=0  summary="Verify the newly created backup archive and inspect its size"
currDepth before target: -1

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
  <system_output>backup.tar -> archive/backup.tar</system_output>
  <system_output>Backup completed.</system_output>
</event>

Expected output:
{"annotation": "Move the backup archive to an archive folder and finish the backup task.", "depth": 1}

Why depth = +1? The user is wrapping up the backup workflow and returning to the
previous level of work.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE D — Starting a test/debug subtask (depth = -1)
Context: After normal shell navigation, the user begins a focused testing session.

neighbor_tail:
  - id=0 depth=0  summary="Open project root directory"
  - id=1 depth=0  summary="View current Git branch and status"
currDepth before target: 0

input xml:
<event>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <system_output>================= test session starts =================</system_output>
</event>

Expected output:
{"annotation": "Start a focused test run for the project using pytest.", "depth": -1}

Why depth = -1? Running the test suite begins a dedicated testing/debugging subtask.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE E — Continuing the test/debug subtask (depth = 0)
Context: The user reruns tests after editing code, still within the same testing workflow.

neighbor_tail:
  - id=0 depth=0  summary="Open project root directory"
  - id=1 depth=0  summary="View current Git branch and status"
  - id=2 depth=-1 summary="Start a focused test run for the project using pytest"
  - id=3 depth=0  summary="Inspect failing test output and error trace"
currDepth before target: -1

input xml:
<event>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>-</user_input><system_output>-</system_output>
  <user_input>k</user_input><system_output>k</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>f</user_input><system_output>f</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>d</user_input><system_output>d</system_output>
  <system_output>Re-running failed tests...</system_output>
</event>

Expected output:
{"annotation": "Re-run only the previously failing tests to check the fix.", "depth": 0}

Why depth = 0? The user remains in the same testing subtask, iterating on failures.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE F — Finishing the test/debug subtask (depth = +1)
Context: All tests pass and the user leaves the focused testing workflow.

neighbor_tail:
  - id=0 depth=0  summary="Open project root directory"
  - id=1 depth=0  summary="View current Git branch and status"
  - id=2 depth=-1 summary="Start a focused test run for the project using pytest"
  - id=3 depth=0  summary="Inspect failing test output and error trace"
  - id=4 depth=0  summary="Re-run only the previously failing tests to check the fix"
currDepth before target: -1

input xml:
<event>
  <system_output>================= 20 passed in 3.21s =================</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input></user_input><system_output>$ </system_output>
</event>

Expected output:
{"annotation": "Finish the test run after all tests pass and return to regular shell work.", "depth": 1}

Why depth = +1? The dedicated testing subtask is complete; depth returns to the
parent level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE G — Starting a nested editor subtask (depth = -1)
Context: Inside an environment-setup subtask, the user opens a config file in an editor.

neighbor_tail:
  - id=0 depth=0  summary="Enter project environment setup directory"
  - id=1 depth=-1 summary="Create and activate a virtual environment for the project"
  - id=2 depth=0  summary="Install core dependencies into the virtual environment"
currDepth before target: -1

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
  <system_output>Opening config.yaml in vim...</system_output>
</event>

Expected output:
{"annotation": "Open the project configuration file in vim while setting up the environment.", "depth": -1}

Why depth = -1? Editing the config is a nested subtask inside the broader
environment-setup workflow.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE H — Exiting the nested editor subtask but staying in the parent subtask (depth = +1)
Context: The user saves the config file and returns to the environment-setup work.

neighbor_tail:
  - id=0 depth=0  summary="Enter project environment setup directory"
  - id=1 depth=-1 summary="Create and activate a virtual environment for the project"
  - id=2 depth=0  summary="Install core dependencies into the virtual environment"
  - id=3 depth=-1 summary="Open the project configuration file in vim while setting up the environment"
currDepth before target: -2

input xml:
<event>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>w</user_input><system_output>w</system_output>
  <user_input>q</user_input><system_output>q</system_output>
  <user_input></user_input>
  <system_output>config.yaml written</system_output>
  <system_output>(back to shell inside virtual environment)</system_output>
</event>

Expected output:
{"annotation": "Save changes in vim and return to the environment-setup shell.", "depth": 1}

Why depth = +1? The nested editor subtask ends, but the user is still inside the
higher-level environment-setup subtask.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE I — Exiting multiple nested subtasks at once (depth = +2)
Context: After finishing environment setup, the user deactivates the environment
and returns to regular shell work.

neighbor_tail:
  - id=0 depth=0  summary="Enter project environment setup directory"
  - id=1 depth=-1 summary="Create and activate a virtual environment for the project"
  - id=2 depth=0  summary="Install core dependencies into the virtual environment"
  - id=3 depth=-1 summary="Open the project configuration file in vim while setting up the environment"
  - id=4 depth=1  summary="Save changes in vim and return to the environment-setup shell."
currDepth before target: -1

input xml:
<event>
  <user_input>d</user_input><system_output>d</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <system_output>(virtual environment deactivated)</system_output>
  <system_output>$ </system_output>
</event>

Expected output:
{"annotation": "Deactivate the virtual environment and return to general shell work.", "depth": 1}

Why depth = +1 (and not +2)? Only the environment-setup subtask is being closed
here; the editor was already exited in the previous event. If the user had left
both a nested tool and its parent subtask in a single step, depth could be 2.
Here, only one level is closed.
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


def build_instruction(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    neighbors_xml = _neighbors_to_xml(pkg)
    targets_xml = _targets_to_xml(pkg)

    examples_xml = (
        f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if use_fewshots else ""
    )

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Keep reasoning CONCISE and FOCUSED
- In <think>...</think>: analyze the command, check depth logic, then conclude
</think_first>

<rules>
- the user's keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent (new task started), 0 for same level (still doing the same task), >0 to exit levels (ended one or multiple tasks)
- maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
- write action-oriented summaries; avoid "user", "they", "typed", "inputs", "enters a command"
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
        seed=42,
    )
    print("Model loaded successfully")
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

    if os.path.exists(GT_PATH):
        print("\n" + "=" * 80)
        print("EMBEDDING-BASED SIMILARITY BETWEEN GT AND MODEL ANNOTATIONS")
        print("=" * 80)
        score_annotations_with_embeddings(pred, GT_PATH)
    else:
        print(f"[WARN] GT_PATH does not exist: {GT_PATH}")
