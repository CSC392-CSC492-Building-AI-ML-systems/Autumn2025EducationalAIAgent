#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 prompt ablation runner.

Depends on `model1_annotator` which defines:
- XML_PATH, MODEL_ID, SUMMARY_WORD_LIMIT, GT_PATH
- FEWSHOTS_PREAMBLE, FEWSHOTS_EXAMPLES, FEWSEP
- SYSTEM_ROLE
- Event dataclass
- load_events(xml_path: str) -> List[Event]
- make_flush_package(upto_idx: int, K: int, N: int) -> Dict
- load_model() -> vLLM LLM
- generate_with_thinking(llm, messages) -> (full_output, json_tail)
- parse_depth_summary_pairs(text: str) -> List[Tuple[int, str]]
- K_TARGET, N_NEIGH

Usage:
  python model1_ablation_runner.py ablate_big   > big_ablation.json
  python model1_ablation_runner.py ablate_few   > fewshot_ablation.json
  python model1_ablation_runner.py ablate_rules > rules_ablation.json
  python model1_ablation_runner.py ablate_think > think_ablation.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import model1_annotator as m1

# ------------------------------
# Event model (alias)
# ------------------------------
Event = m1.Event

# -----------------------------------------------------------------------------
# Prompt configuration
# -----------------------------------------------------------------------------
@dataclass
class PromptConfig:
    """Controls which high-level blocks are included in the prompt.

    Depth and neighbors are *always* included in the prompt and are not
    ablated in this script (per design choice).
    """

    include_system_role: bool = True
    include_fewshots: bool = True
    include_think_first: bool = True
    include_rules: bool = True

    # kept for compatibility / labeling
    include_stack_invariant_rule: bool = True


# Few-shots: use structured data from annotator
FEWSEP = m1.FEWSEP
FEWSHOTS_PREAMBLE: str = m1.FEWSHOTS_PREAMBLE
FEWSHOTS_EXAMPLES: List[str] = list(m1.FEWSHOTS_EXAMPLES)

# Rules broken into individual lines for within-block ablation
BASE_RULES: List[str] = [
    "the user's keystrokes appear separately; combine them to form the full command before interpreting it",
    "depth is an integer (≥ -1); -1 for subevent (new task started), 0 for same level (still doing the same task), >0 to exit levels (ended one or multiple tasks)",
    "maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth",
    "write action-oriented summaries; avoid \"user\", \"they\", \"typed\", \"inputs\", \"enters a command\"",
    "depth is relative to the previous events and nothing else",
    "do not copy xml tags or attributes; no repeated phrases",
    "do not mention an address that was not explicitly mentioned in the event",
    "if the target event contains an <annotation> tag or depth value ignore it",
    "if there are no neighbors then the depth you output should be 0",
]

# Think-first bullet points as separate items
THINK_LINES: List[str] = [
    "- Keep reasoning CONCISE and FOCUSED",
    "- In <think>...</think>: analyze the command, check depth logic, then conclude",
    "- Aim for 2-3 sentences of reasoning maximum",
    "- Skip obvious observations",
    "- Use neighbors ONLY for continuity; do not invent context.",
]

# -----------------------------------------------------------------------------
# GT loader + metrics helpers
# -----------------------------------------------------------------------------
def _load_gt_annotations(gt_path: str) -> Dict[int, Dict[str, object]]:
    """
    Load GT annotations from a (depth, summary) alternating .txt file.

    Returns:
        {idx: {"depth": int, "summary": str}}
    """
    gt: Dict[int, Dict[str, object]] = {}
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except FileNotFoundError:
        print(f"[GT] Could not open GT file: {gt_path}", file=sys.stderr)
        return gt

    idx = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            depth = int(line)
        except ValueError:
            # Malformed depth line; skip
            i += 1
            continue

        if i + 1 >= n:
            break
        summary = lines[i + 1].strip()
        gt[idx] = {"depth": depth, "summary": summary}
        idx += 1
        i += 2

    return gt


def _build_pred_from_flushes(flushes: List[Dict]) -> Dict[int, Dict[str, object]]:
    """
    Reconstruct {idx: {"depth": int, "summary": str}} from a list of flush entries.

    Each flush entry must contain:
      - "target_idxs": List[int]
      - "model_output_json_tail": str
    """
    pred: Dict[int, Dict[str, object]] = {}

    for fl in flushes:
        target_idxs = fl.get("target_idxs") or []
        json_tail = fl.get("model_output_json_tail") or ""

        pairs = m1.parse_depth_summary_pairs(json_tail)
        # Drop placeholder / trivial annotations (mirror m1.run_flushes logic)
        cleaned_pairs = [
            (depth, ann)
            for (depth, ann) in pairs
            if ann is not None
            and ann.strip() not in ("...", '"..."')
            and len(ann.strip()) >= 5
        ]

        # If model produced more pairs than targets, keep the last ones
        if len(cleaned_pairs) > len(target_idxs):
            cleaned_pairs = cleaned_pairs[-len(target_idxs) :]

        for (depth, summary), idx in zip(cleaned_pairs, target_idxs):
            pred[idx] = {"depth": depth, "summary": summary}

    return pred


def _compute_annotation_metrics(
    pred: Dict[int, Dict[str, object]],
    gt_path: str,
    bi_model,
    cross_model,
    rouge_scorer_obj,
    bert_lang: str = "en",
) -> Dict[str, object]:
    """
    Compute:
      - bi-encoder cosine similarity
      - ROUGE-L F1
      - cross-encoder STS similarity
      - BERTScore F1

    Returns a dict of aggregate stats.
    """
    import numpy as np
    from bert_score import score as bertscore_score

    gt = _load_gt_annotations(gt_path)
    if not gt:
        return {"num_pairs": 0}

    common_idxs = sorted(set(gt.keys()) & set(pred.keys()))
    if not common_idxs:
        return {"num_pairs": 0}

    gt_summaries: List[str] = []
    pred_summaries: List[str] = []
    for idx in common_idxs:
        gt_sum = str(gt[idx]["summary"])
        pred_sum = str(pred[idx].get("summary", "") or "")
        gt_summaries.append(gt_sum)
        pred_summaries.append(pred_sum)

    # ---------- Bi-encoder cosine ----------
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
    sims = (gt_emb * pred_emb).sum(dim=-1)
    sims_np = sims.detach().cpu().numpy()

    mean_sim = float(sims_np.mean())
    median_sim = float(np.median(sims_np))
    p25_sim, p75_sim = np.percentile(sims_np, [25, 75])

    # ---------- ROUGE-L ----------
    rouge_scores = []
    for ref, hyp in zip(gt_summaries, pred_summaries):
        score = rouge_scorer_obj.score(ref, hyp)["rougeL"].fmeasure
        rouge_scores.append(score)
    rouge_np = np.array(rouge_scores, dtype=float)

    mean_rouge = float(rouge_np.mean())
    median_rouge = float(np.median(rouge_np))
    p25_rouge, p75_rouge = np.percentile(rouge_np, [25, 75])

    # ---------- Cross-encoder STS ----------
    pair_inputs = list(zip(gt_summaries, pred_summaries))
    cross_scores = cross_model.predict(pair_inputs)
    cross_np = np.array(cross_scores, dtype=float)

    mean_cross = float(cross_np.mean())
    median_cross = float(np.median(cross_np))
    p25_cross, p75_cross = np.percentile(cross_np, [25, 75])

    # ---------- BERTScore F1 ----------
    P, R, F1 = bertscore_score(
        pred_summaries,
        gt_summaries,
        lang=bert_lang,
        rescale_with_baseline=False,
        verbose=False,
    )
    bert_f1_np = F1.detach().cpu().numpy()

    mean_bert = float(bert_f1_np.mean())
    median_bert = float(np.median(bert_f1_np))
    p25_bert, p75_bert = np.percentile(bert_f1_np, [25, 75])

    return {
        "num_pairs": len(common_idxs),

        "cosine_mean": mean_sim,
        "cosine_median": median_sim,
        "cosine_p25": p25_sim,
        "cosine_p75": p75_sim,

        "rougeL_mean": mean_rouge,
        "rougeL_median": median_rouge,
        "rougeL_p25": p25_rouge,
        "rougeL_p75": p75_rouge,

        "cross_mean": mean_cross,
        "cross_median": median_cross,
        "cross_p25": p25_cross,
        "cross_p75": p75_cross,

        "bertF1_mean": mean_bert,
        "bertF1_median": median_bert,
        "bertF1_p25": p25_bert,
        "bertF1_p75": p75_bert,
    }


def _print_metrics_for_ablation(name: str, metrics: Dict[str, object]) -> None:
    """Print a compact metrics line for an ablation (to stderr)."""
    if not metrics or metrics.get("num_pairs", 0) == 0:
        print(f"[METRICS] ablation={name}: no overlapping GT/pred pairs", file=sys.stderr)
        return

    msg = (
        f"[METRICS] ablation={name} "
        f"pairs={metrics['num_pairs']} "
        f"cos_mean={metrics['cosine_mean']:.4f} "
        f"rougeL_mean={metrics['rougeL_mean']:.4f} "
        f"cross_mean={metrics['cross_mean']:.4f} "
        f"bertF1_mean={metrics['bertF1_mean']:.4f}"
    )
    print(msg, file=sys.stderr)


# -----------------------------------------------------------------------------
# Blocks: rules, examples, think-first
# -----------------------------------------------------------------------------
def build_rules_block(cfg: PromptConfig, rule_indices: Optional[List[int]] = None) -> str:
    if not cfg.include_rules:
        return ""

    if rule_indices is None:
        rule_indices = list(range(len(BASE_RULES)))

    rules = [BASE_RULES[i] for i in rule_indices]
    rules_str = "\n".join(f"- {r}" for r in rules)
    return f"<rules>\n{rules_str}\n</rules>"


def build_examples_block(
    cfg: PromptConfig, fewshot_indices: Optional[List[int]] = None
) -> str:
    if not cfg.include_fewshots:
        return ""

    if fewshot_indices is None:
        chunks = FEWSHOTS_EXAMPLES
    else:
        chunks = [FEWSHOTS_EXAMPLES[i] for i in fewshot_indices]

    if not chunks:
        body = FEWSHOTS_PREAMBLE
    else:
        examples_body = ("\n\n" + FEWSEP + "\n\n").join(chunks)
        body = f"{FEWSHOTS_PREAMBLE}\n\n{FEWSEP}\n\n{examples_body}"

    return f"\n<examples>\n{body}\n</examples>"


def build_think_block(
    cfg: PromptConfig, think_indices: Optional[List[int]] = None
) -> str:
    if not cfg.include_think_first:
        return ""

    if think_indices is None:
        lines = THINK_LINES
    else:
        lines = [THINK_LINES[i] for i in think_indices]

    body = "\n".join(lines)
    return f"<think_first>\n{body}\n</think_first>"


# -----------------------------------------------------------------------------
# Configurable prompt builder (neighbors + currDepth always included)
# -----------------------------------------------------------------------------
def build_instruction_cfg(
    pkg: Dict,
    cfg: PromptConfig,
    fewshot_indices: Optional[List[int]] = None,
    rule_indices: Optional[List[int]] = None,
    think_indices: Optional[List[int]] = None,
) -> str:
    # Neighbors
    neighbor_items: List[Dict[str, str]] = []
    if pkg.get("neighbor_info"):
        for line in pkg["neighbor_info"]:
            match = re.match(r"- id=(\d+) depth=(-?\d+)\s+summary=(.+)", line)
            if match:
                nid, ndepth, nsummary = match.groups()
                neighbor_items.append(
                    {"id": nid, "depth": ndepth, "summary": nsummary}
                )

    neighbors_xml = (
        "\n".join(
            f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
            for n in neighbor_items
        )
        or "    <neighbor>(none)</neighbor>"
    )

    # Targets
    target_items: List[Dict[str, str]] = [
        {"id": idx, "xml": xml_str}
        for idx, xml_str in zip(pkg["target_idxs"], pkg["target_events"])
    ]
    targets_xml = "\n".join(
        f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in target_items
    )

    # Examples / rules / think-first
    examples_xml = build_examples_block(cfg, fewshot_indices=fewshot_indices)
    rules_block = build_rules_block(cfg, rule_indices=rule_indices)
    think_block = build_think_block(cfg, think_indices=think_indices)

    curr_depth_xml = f"<curr_depth_max>{pkg.get('currDepth')}</curr_depth_max>"

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {m1.SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

{think_block}

{rules_block}

{examples_xml}

<instruction>
for each target_event, output exactly one json with "annotation" first, then "depth".
</instruction>

<inputs>
  {curr_depth_xml}
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""
    return prompt


def build_messages_cfg(instruction: str, cfg: PromptConfig) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if cfg.include_system_role:
        msgs.append({"role": "system", "content": m1.SYSTEM_ROLE})
    msgs.append({"role": "user", "content": instruction})
    return msgs


# -----------------------------------------------------------------------------
# Ablation runners (all reuse m1.load_model & m1.generate_with_thinking)
# -----------------------------------------------------------------------------
def run_bigblock_ablation(evs) -> Dict:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rouge_score import rouge_scorer

    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()

    # Evaluation models (shared across ablations in this run)
    bi_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    configs = {
        "full": PromptConfig(),
        "no_fewshots": PromptConfig(include_fewshots=False),
        "no_rules": PromptConfig(include_rules=False),
        "no_think_first": PromptConfig(include_think_first=False),
        "no_system_role": PromptConfig(include_system_role=False),
        "no_stack_invariant_rule": PromptConfig(include_stack_invariant_rule=False),
    }

    all_results: Dict[str, object] = {
        "xml_path": m1.XML_PATH,
        "model_id": m1.MODEL_ID,
        "ablations": [],
    }

    total = len(evs)
    start_idx = 0

    for name, cfg in configs.items():
        ablation_entry = {
            "name": name,
            "config": asdict(cfg),
            "flushes": [],
        }

        total_time = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(
                upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH
            )
            instr = build_instruction_cfg(pkg, cfg)
            messages = build_messages_cfg(instr, cfg)

            start = time.perf_counter()
            full_output, json_part, prompt_tokens, gen_tokens = m1.generate_with_thinking(llm, messages)
            elapsed = time.perf_counter() - start

            prompt_tok_count = prompt_tokens
            output_tok_count = gen_tokens

            total_time += elapsed
            total_prompt_tokens += prompt_tok_count
            total_output_tokens += output_tok_count
            num_prompts += 1

            ablation_entry["flushes"].append(
                {
                    "upto": upto,
                    "target_idxs": pkg["target_idxs"],
                    "currDepth_before": pkg["currDepth"],
                    "prompt": instr,
                    "model_output_raw": full_output,
                    "model_output_json_tail": json_part,
                }
            )

        if num_prompts > 0 and total_time > 0:
            avg_sec = total_time / num_prompts
            prompt_tps = total_prompt_tokens / total_time
            output_tps = total_output_tokens / total_time
            combined_tps = (total_prompt_tokens + total_output_tokens) / total_time
        else:
            avg_sec = prompt_tps = output_tps = combined_tps = None

        ablation_entry["timing"] = {
            "num_prompts": num_prompts,
            "total_wall_time_sec": total_time,
            "avg_sec_per_prompt": avg_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "prompt_tokens_per_sec": prompt_tps,
            "output_tokens_per_sec": output_tps,
            "combined_tokens_per_sec": combined_tps,
        }

        # ---- compute metrics for this ablation ----
        pred = _build_pred_from_flushes(ablation_entry["flushes"])
        metrics = _compute_annotation_metrics(
            pred,
            m1.GT_PATH,
            bi_model=bi_model,
            cross_model=cross_model,
            rouge_scorer_obj=rouge,
        )
        ablation_entry["metrics"] = metrics
        _print_metrics_for_ablation(name, metrics)

        all_results["ablations"].append(ablation_entry)

    return all_results


def run_fewshots_ablation(evs) -> Dict:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rouge_score import rouge_scorer

    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_fewshots=True)

    # Evaluation models
    bi_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    n = len(FEWSHOTS_EXAMPLES)
    all_results: Dict[str, object] = {
        "xml_path": m1.XML_PATH,
        "model_id": m1.MODEL_ID,
        "fewshots_ablations": [],
    }

    total = len(evs)
    start_idx = 0

    def run_with_indices(indices: List[int], label: str) -> Dict:
        entry = {
            "name": label,
            "indices": indices,
            "config": asdict(cfg),
            "flushes": [],
        }

        total_time = total_prompt_tokens = total_output_tokens = 0.0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(
                upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH
            )
            instr = build_instruction_cfg(pkg, cfg, fewshot_indices=indices)
            messages = build_messages_cfg(instr, cfg)

            start = time.perf_counter()
            full_output, json_part, prompt_tokens, gen_tokens = m1.generate_with_thinking(llm, messages)
            elapsed = time.perf_counter() - start

            prompt_tok_count = prompt_tokens
            output_tok_count = gen_tokens

            total_time += elapsed
            total_prompt_tokens += prompt_tok_count
            total_output_tokens += output_tok_count
            num_prompts += 1

            entry["flushes"].append(
                {
                    "upto": upto,
                    "target_idxs": pkg["target_idxs"],
                    "currDepth_before": pkg["currDepth"],
                    "prompt": instr,
                    "model_output_raw": full_output,
                    "model_output_json_tail": json_part,
                }
            )

        if num_prompts > 0 and total_time > 0:
            avg_sec = total_time / num_prompts
            prompt_tps = total_prompt_tokens / total_time
            output_tps = total_output_tokens / total_time
            combined_tps = (total_prompt_tokens + total_output_tokens) / total_time
        else:
            avg_sec = prompt_tps = output_tps = combined_tps = None

        entry["timing"] = {
            "num_prompts": num_prompts,
            "total_wall_time_sec": total_time,
            "avg_sec_per_prompt": avg_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "prompt_tokens_per_sec": prompt_tps,
            "output_tokens_per_sec": output_tps,
            "combined_tokens_per_sec": combined_tps,
        }

        pred = _build_pred_from_flushes(entry["flushes"])
        metrics = _compute_annotation_metrics(
            pred,
            m1.GT_PATH,
            bi_model=bi_model,
            cross_model=cross_model,
            rouge_scorer_obj=rouge,
        )
        entry["metrics"] = metrics
        _print_metrics_for_ablation(label, metrics)

        return entry

    # Baseline
    all_indices = list(range(n))
    all_results["fewshots_ablations"].append(
        run_with_indices(all_indices, "all_fewshots")
    )

    # Leave-one-out
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_example_{i}"
        all_results["fewshots_ablations"].append(run_with_indices(indices, label))

    return all_results


def run_rules_ablation(evs) -> Dict:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rouge_score import rouge_scorer

    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_rules=True)

    # Evaluation models
    bi_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    n = len(BASE_RULES)
    all_results: Dict[str, object] = {
        "xml_path": m1.XML_PATH,
        "model_id": m1.MODEL_ID,
        "rules_ablations": [],
    }

    total = len(evs)
    start_idx = 0

    def run_with_rule_indices(indices: List[int], label: str) -> Dict:
        entry = {
            "name": label,
            "rule_indices": indices,
            "config": asdict(cfg),
            "flushes": [],
        }

        total_time = total_prompt_tokens = total_output_tokens = 0.0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(
                upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH
            )
            instr = build_instruction_cfg(pkg, cfg, rule_indices=indices)
            messages = build_messages_cfg(instr, cfg)

            start = time.perf_counter()
            full_output, json_part = m1.generate_with_thinking(llm, messages)
            elapsed = time.perf_counter() - start

            prompt_tok_count = len(tokenizer.encode(instr))
            output_tok_count = len(tokenizer.encode(full_output))

            total_time += elapsed
            total_prompt_tokens += prompt_tok_count
            total_output_tokens += output_tok_count
            num_prompts += 1

            entry["flushes"].append(
                {
                    "upto": upto,
                    "target_idxs": pkg["target_idxs"],
                    "currDepth_before": pkg["currDepth"],
                    "prompt": instr,
                    "model_output_raw": full_output,
                    "model_output_json_tail": json_part,
                }
            )

        if num_prompts > 0 and total_time > 0:
            avg_sec = total_time / num_prompts
            prompt_tps = total_prompt_tokens / total_time
            output_tps = total_output_tokens / total_time
            combined_tps = (total_prompt_tokens + total_output_tokens) / total_time
        else:
            avg_sec = prompt_tps = output_tps = combined_tps = None

        entry["timing"] = {
            "num_prompts": num_prompts,
            "total_wall_time_sec": total_time,
            "avg_sec_per_prompt": avg_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "prompt_tokens_per_sec": prompt_tps,
            "output_tokens_per_sec": output_tps,
            "combined_tokens_per_sec": combined_tps,
        }

        pred = _build_pred_from_flushes(entry["flushes"])
        metrics = _compute_annotation_metrics(
            pred,
            m1.GT_PATH,
            bi_model=bi_model,
            cross_model=cross_model,
            rouge_scorer_obj=rouge,
        )
        entry["metrics"] = metrics
        _print_metrics_for_ablation(label, metrics)

        return entry

    # Baseline
    all_indices = list(range(n))
    all_results["rules_ablations"].append(
        run_with_rule_indices(all_indices, "all_rules")
    )

    # Leave-one-out
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_rule_{i}"
        all_results["rules_ablations"].append(
            run_with_rule_indices(indices, label)
        )

    return all_results


def run_think_ablation(evs) -> Dict:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rouge_score import rouge_scorer

    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_think_first=True)

    # Evaluation models
    bi_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    n = len(THINK_LINES)
    all_results: Dict[str, object] = {
        "xml_path": m1.XML_PATH,
        "model_id": m1.MODEL_ID,
        "think_ablations": [],
    }

    total = len(evs)
    start_idx = 0

    def run_with_think_indices(indices: List[int], label: str) -> Dict:
        entry = {
            "name": label,
            "think_indices": indices,
            "config": asdict(cfg),
            "flushes": [],
        }

        total_time = total_prompt_tokens = total_output_tokens = 0.0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(
                upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH
            )
            instr = build_instruction_cfg(pkg, cfg, think_indices=indices)
            messages = build_messages_cfg(instr, cfg)

            start = time.perf_counter()
            full_output, json_part = m1.generate_with_thinking(llm, messages)
            elapsed = time.perf_counter() - start

            prompt_tok_count = len(tokenizer.encode(instr))
            output_tok_count = len(tokenizer.encode(full_output))

            total_time += elapsed
            total_prompt_tokens += prompt_tok_count
            total_output_tokens += output_tok_count
            num_prompts += 1

            entry["flushes"].append(
                {
                    "upto": upto,
                    "target_idxs": pkg["target_idxs"],
                    "currDepth_before": pkg["currDepth"],
                    "prompt": instr,
                    "model_output_raw": full_output,
                    "model_output_json_tail": json_part,
                }
            )

        if num_prompts > 0 and total_time > 0:
            avg_sec = total_time / num_prompts
            prompt_tps = total_prompt_tokens / total_time
            output_tps = total_output_tokens / total_time
            combined_tps = (total_prompt_tokens + total_output_tokens) / total_time
        else:
            avg_sec = prompt_tps = output_tps = combined_tps = None

        entry["timing"] = {
            "num_prompts": num_prompts,
            "total_wall_time_sec": total_time,
            "avg_sec_per_prompt": avg_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "prompt_tokens_per_sec": prompt_tps,
            "output_tokens_per_sec": output_tps,
            "combined_tokens_per_sec": combined_tps,
        }

        pred = _build_pred_from_flushes(entry["flushes"])
        metrics = _compute_annotation_metrics(
            pred,
            m1.GT_PATH,
            bi_model=bi_model,
            cross_model=cross_model,
            rouge_scorer_obj=rouge,
        )
        entry["metrics"] = metrics
        _print_metrics_for_ablation(label, metrics)

        return entry

    # Baseline
    all_indices = list(range(n))
    all_results["think_ablations"].append(
        run_with_think_indices(all_indices, "all_think_lines")
    )

    # Leave-one-out
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_think_{i}"
        all_results["think_ablations"].append(
            run_with_think_indices(indices, label)
        )

    return all_results


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    events = m1.load_events(m1.XML_PATH)
    if not events:
        raise SystemExit("No events loaded from XML_PATH")

    mode = sys.argv[1] if len(sys.argv) > 1 else "ablate_big"

    if mode == "ablate_big":
        results = run_bigblock_ablation(events)
    elif mode == "ablate_few":
        results = run_fewshots_ablation(events)
    elif mode == "ablate_rules":
        results = run_rules_ablation(events)
    elif mode == "ablate_think":
        results = run_think_ablation(events)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    print(json.dumps(results, ensure_ascii=False, indent=2))
