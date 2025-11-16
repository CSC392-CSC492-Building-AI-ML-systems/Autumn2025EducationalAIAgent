#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 prompt ablation runner (separate script).

This script performs prompt ablation experiments on the Model-1 streamed
annotator. It depends on the main annotator module (imported as
`model1_annotator`) which must define:

- XML_PATH, MODEL_ID, SUMMARY_WORD_LIMIT
- FEWSHOTS_BLOCK, SYSTEM_ROLE
- Event dataclass
- load_events(xml_path: str) -> List[Event]
- make_flush_package(upto_idx: int, K: int, N: int) -> Dict
- load_model() -> vLLM LLM
- generate_with_thinking(llm, messages) -> (full_output, json_tail)
- K_TARGET, N_NEIGH

Usage examples:

  python model1_ablation_runner.py ablate_big   > big_ablation.json
  python model1_ablation_runner.py ablate_few   > fewshot_ablation.json
  python model1_ablation_runner.py ablate_rules > rules_ablation.json
  python model1_ablation_runner.py ablate_think > think_ablation.json

Each JSON file contains, for every ablation config and every event:
  - the config used
  - the exact prompt sent
  - the model's full output
  - the JSON tail (everything after </think>)
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
# Event model
# ------------------------------
@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int] = None
    summary_xml: Optional[str] = None

# Type alias for convenience (we actually use the main module's Event)
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

    # Big blocks
    include_system_role: bool = True
    include_fewshots: bool = True
    include_think_first: bool = True
    include_rules: bool = True

    # Example granular toggle inside rules (kept for compatibility / labeling)
    include_stack_invariant_rule: bool = True


# Split FEWSHOTS into chunks for within-block ablation
FEWSEP = "═══════════════════════════════════════════════════════════════════════════════"
FEWSHOTS_EXAMPLES: List[str] = m1.FEWSHOTS_BLOCK.split(FEWSEP)

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

# Think-first bullet points as separate items for within-block ablation
THINK_LINES: List[str] = [
    "- Keep reasoning CONCISE and FOCUSED",
    "- In <think>...</think>: analyze the command, check depth logic, then conclude",
    "- Aim for 2-3 sentences of reasoning maximum",
    "- Skip obvious observations",
    "- Use neighbors ONLY for continuity; do not invent context.",
]


# -----------------------------------------------------------------------------
# Blocks: rules, examples, think-first
# -----------------------------------------------------------------------------

def build_rules_block(cfg: PromptConfig, rule_indices: Optional[List[int]] = None) -> str:
    """Build the <rules> block, optionally using only a subset of BASE_RULES.

    rule_indices: list of indices into BASE_RULES to keep. If None, keep all.
    """
    if not cfg.include_rules:
        return ""

    if rule_indices is None:
        rule_indices = list(range(len(BASE_RULES)))

    rules = [BASE_RULES[i] for i in rule_indices]
    rules_str = "\n".join(f"- {r}" for r in rules)
    return f"<rules>\n{rules_str}\n</rules>"


def build_examples_block(cfg: PromptConfig, fewshot_indices: Optional[List[int]] = None) -> str:
    """Build the <examples> block, optionally keeping only some few-shot chunks."""
    if not cfg.include_fewshots:
        return ""

    if fewshot_indices is None:
        chunks = FEWSHOTS_EXAMPLES
    else:
        chunks = [FEWSHOTS_EXAMPLES[i] for i in fewshot_indices]

    body = FEWSEP.join(chunks)
    return f"\n<examples>\n{body}\n</examples>"


def build_think_block(cfg: PromptConfig, think_indices: Optional[List[int]] = None) -> str:
    """Build the <think_first> block as a list of bullet lines.

    think_indices: list of indices into THINK_LINES to include. If None, include all.
    """
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
    """Build the instruction prompt using a PromptConfig and optional ablations.

    - neighbors + currDepth are always included (no ablation).
    - fewshot_indices, rule_indices, think_indices let us ablate within blocks.
    """
    # Neighbors (always included if present)
    neighbor_items: List[Dict[str, str]] = []
    if pkg.get("neighbor_info"):
        for line in pkg["neighbor_info"]:
            match = re.match(r"- id=(\d+) depth=(-?\d+)\s+summary=(.+)", line)
            if match:
                nid, ndepth, nsummary = match.groups()
                neighbor_items.append({"id": nid, "depth": ndepth, "summary": nsummary})

    neighbors_xml = "\n".join(
        [
            f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
            for n in neighbor_items
        ]
    ) or "    <neighbor>(none)</neighbor>"

    # Targets (always needed)
    target_items: List[Dict[str, str]] = []
    for idx, xml_str in zip(pkg["target_idxs"], pkg["target_events"]):
        target_items.append({"id": idx, "xml": xml_str})

    targets_xml = "\n".join(
        [f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in target_items]
    )

    # Examples (few-shots)
    examples_xml = build_examples_block(cfg, fewshot_indices=fewshot_indices)

    # Rules
    rules_block = build_rules_block(cfg, rule_indices=rule_indices)

    # Think-first block
    think_block = build_think_block(cfg, think_indices=think_indices)

    # currDepth (always present)
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
# Ablation runners
# -----------------------------------------------------------------------------

def run_bigblock_ablation(evs: List[Event]) -> Dict:
    """Big-block ablation over high-level prompt components.

    Returns a JSON-serializable dict with, for each config:
      - the config used
      - per-event prompt
      - full model output and JSON tail (assuming JSON is at the end)
      - timing/resource summary per ablation config
    """
    # Ensure the annotator module sees the correct events list for depth tracking
    m1.events = evs

    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()

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
    start_idx = 0  # can clamp for smaller runs

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
            pkg = m1.make_flush_package(upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH)
            instr = build_instruction_cfg(pkg, cfg)
            messages = build_messages_cfg(instr, cfg)

            start = time.perf_counter()
            full_output, json_part = m1.generate_with_thinking(llm, messages)
            elapsed = time.perf_counter() - start

            # Token-based resource metrics
            prompt_tok_count = len(tokenizer.encode(instr))
            output_tok_count = len(tokenizer.encode(full_output))

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

        # Aggregate timing/resource metrics per ablation config
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

        all_results["ablations"].append(ablation_entry)

    return all_results


def run_fewshots_ablation(evs: List[Event]) -> Dict:
    """Within-FEWSHOTS ablation: all examples, then leave-one-out over each example."""
    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_fewshots=True)

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

        total_time = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH)
            instr = build_instruction_cfg(pkg, cfg, fewshot_indices=indices)
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

        return entry

    # Baseline: all examples
    all_indices = list(range(n))
    all_results["fewshots_ablations"].append(run_with_indices(all_indices, "all_fewshots"))

    # Leave-one-out for each example
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_example_{i}"
        all_results["fewshots_ablations"].append(run_with_indices(indices, label))

    return all_results


def run_rules_ablation(evs: List[Event]) -> Dict:
    """Within-RULES ablation: all rules, then drop each rule one at a time."""
    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_rules=True)

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

        total_time = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH)
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

        return entry

    # Baseline: all rules
    all_indices = list(range(n))
    all_results["rules_ablations"].append(run_with_rule_indices(all_indices, "all_rules"))

    # Leave-one-out per rule
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_rule_{i}"
        all_results["rules_ablations"].append(run_with_rule_indices(indices, label))

    return all_results


def run_think_ablation(evs: List[Event]) -> Dict:
    """Within-think-first ablation: all think-lines, then drop each bullet once."""
    m1.events = evs
    llm = m1.load_model()
    tokenizer = llm.get_tokenizer()
    cfg = PromptConfig(include_think_first=True)

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

        total_time = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        num_prompts = 0

        for upto in range(start_idx, total):
            pkg = m1.make_flush_package(upto_idx=upto, K=m1.K_TARGET, N=m1.N_NEIGH)
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

        return entry

    # Baseline: all think-lines
    all_indices = list(range(n))
    all_results["think_ablations"].append(
        run_with_think_indices(all_indices, "all_think_lines")
    )

    # Leave-one-out per think-line
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        label = f"drop_think_{i}"
        all_results["think_ablations"].append(run_with_think_indices(indices, label))

    return all_results


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    events: List[Event] = m1.load_events(m1.XML_PATH)
    if not events:
        raise SystemExit("No events loaded from XML_PATH")

    mode = sys.argv[1] if len(sys.argv) > 1 else "ablate_big"

    if mode == "ablate_big":
        results = run_bigblock_ablation(events)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif mode == "ablate_few":
        results = run_fewshots_ablation(events)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif mode == "ablate_rules":
        results = run_rules_ablation(events)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif mode == "ablate_think":
        results = run_think_ablation(events)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        raise SystemExit(f"Unknown mode: {mode}")
