#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-1 benchmark runner.

Goal: benchmark a *single* prompt configuration (the one defined in model1_annotator)
using the same instruction/message construction code, and extract key metrics:

- Time:
    - total_wall_time_sec
    - avg_sec_per_flush
    - prompt_tokens_per_sec
    - gen_tokens_per_sec
    - combined_tokens_per_sec

- Tokens:
    - total_prompt_tokens
    - total_gen_tokens
    - total_tokens
    - total_think_tokens   (tokens inside <think>...</think>)
    - avg_think_tokens_per_flush
    - think_token_share_of_gen

- Resources (best-effort; optional deps):
    - cpu_rss_mb_start / end / peak       (via psutil, if installed)
    - gpu_mem_mb_start / end / peak       (via pynvml, if installed)
    - gpu_info: {"name": ..., "total_mem_mb": ...} for GPU 0 if available

Usage (single XML, using m1.XML_PATH):
    python model1_benchmark_runner.py > benchmark.json

Explicit XML:
    python model1_benchmark_runner.py path/to/file.xml > benchmark.json

Directory of XMLs:
    python model1_benchmark_runner.py path/to/xml_dir > benchmark_all.json

This script intentionally delegates *all* prompt construction to model1_annotator:
    - m1.make_flush_package
    - m1.build_instruction
    - m1.build_messages
    - m1.generate_with_thinking
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import model1_annotator as m1

# Optional deps for resource tracking
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None

# ------------------------------
# Globals (shared model + tokenizer)
# ------------------------------
_GLOBAL_LLM = None
_GLOBAL_TOKENIZER = None

# ------------------------------
# Resource helpers
# ------------------------------


def _cpu_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if psutil unavailable."""
    if psutil is None:
        return None
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


class GPUMonitor:
    """Lightweight wrapper around NVML for GPU-0 memory stats."""

    def __init__(self) -> None:
        self.available = False
        self.handle = None
        self.name = None
        self.total_mb = None

        if pynvml is None:
            return

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_mb = info.total / (1024 * 1024)
            self.name = pynvml.nvmlDeviceGetName(self.handle).decode("utf-8")
            self.available = True
        except Exception:
            self.available = False

    def get_used_mb(self) -> Optional[float]:
        if not self.available or self.handle is None:
            return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / (1024 * 1024)
        except Exception:
            return None

    def get_info(self) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        return {"name": self.name, "total_mem_mb": self.total_mb}

    def shutdown(self) -> None:
        if not self.available:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# ------------------------------
# Token helpers
# ------------------------------


def _ensure_llm_and_tokenizer():
    """Lazy-init shared LLM + HF tokenizer from vLLM."""
    global _GLOBAL_LLM, _GLOBAL_TOKENIZER
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = m1.load_model()
    if _GLOBAL_TOKENIZER is None:
        # vLLM LLM exposes underlying HF tokenizer
        _GLOBAL_TOKENIZER = _GLOBAL_LLM.get_tokenizer()
    return _GLOBAL_LLM, _GLOBAL_TOKENIZER


def _count_tokens(tokenizer, texts: List[str]) -> int:
    """Count tokens for a list of strings using HF tokenizer (approx)."""
    if not texts:
        return 0

    try:
        encoded = tokenizer(
            texts,
            return_tensors=None,
            add_special_tokens=False,
            padding=False,
        )
    except TypeError:
        # Some tokenizers don't support add_special_tokens / padding kwargs
        encoded = tokenizer(texts, return_tensors=None)

    # HF can return dict with "input_ids" = List[List[int]] or similar
    if isinstance(encoded, dict) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, list):
            # batched: List[List[int]]
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                return sum(len(seq) for seq in input_ids)
            # flat: List[int]
            return len(input_ids)
    # Fallback: treat as sequence-like
    if hasattr(encoded, "__len__"):
        return len(encoded)

    return 0


def _count_think_tokens(tokenizer, full_output: str) -> int:
    """Extract <think>...</think> blocks and count tokens inside."""
    think_blocks = re.findall(r"<think>(.*?)</think>", full_output, flags=re.DOTALL)
    if not think_blocks:
        return 0
    return _count_tokens(tokenizer, think_blocks)


# ------------------------------
# Benchmark core
# ------------------------------


@dataclass
class FlushRecord:
    upto: int
    target_idxs: List[int]
    wall_time_sec: float
    prompt_tokens: int
    gen_tokens: int
    think_tokens: int
    cpu_rss_mb_before: Optional[float]
    cpu_rss_mb_after: Optional[float]
    gpu_mem_mb_before: Optional[float]
    gpu_mem_mb_after: Optional[float]


def benchmark_single_xml(xml_path: str) -> Dict[str, Any]:
    """Run benchmark for a single XML trace.

    Uses model1_annotator as the single source of truth:
      - m1.XML_PATH = xml_path
      - events = m1.load_events
      - for each upto:
          pkg = m1.make_flush_package(...)
          instr = m1.build_instruction(pkg, ...)
          messages = m1.build_messages(instr)
          full_output, json_tail, prompt_tokens, gen_tokens = m1.generate_with_thinking(...)
    """
    m1.XML_PATH = xml_path
    events = m1.load_events(m1.XML_PATH)
    if not events:
        raise SystemExit(f"No events loaded from XML: {xml_path}")

    # Attach events so annotator helpers (if any) can see them
    m1.events = events

    llm, tokenizer = _ensure_llm_and_tokenizer()
    gpu_mon = GPUMonitor()

    num_events = len(events)
    flush_records: List[FlushRecord] = []

    # Resource baselines
    cpu_start = _cpu_rss_mb()
    gpu_start = gpu_mon.get_used_mb()

    cpu_peak = cpu_start or 0.0
    gpu_peak = gpu_start or 0.0

    total_wall = 0.0
    total_prompt_toks = 0
    total_gen_toks = 0
    total_think_toks = 0

    for upto in range(num_events):
        pkg = m1.make_flush_package(
            upto_idx=upto,
            K=m1.K_TARGET,
            N=m1.N_NEIGH,
        )

        # ---- Build prompt via model1_annotator (no duplication here) ----
        # If build_instruction has additional kwargs (e.g. include_fewshots),
        # you can tweak this call, but the idea is to centralize all prompt
        # construction there.
        try:
            instruction = m1.build_instruction(pkg)
        except TypeError:
            # Backwards-compatible fallback if build_instruction requires flags
            instruction = m1.build_instruction(pkg, include_fewshots=True)

        messages = m1.build_messages(instruction)

        cpu_before = _cpu_rss_mb()
        gpu_before = gpu_mon.get_used_mb()

        t0 = time.perf_counter()
        full_output, json_tail, prompt_toks, gen_toks = m1.generate_with_thinking(
            llm, messages
        )
        dt = time.perf_counter() - t0

        cpu_after = _cpu_rss_mb()
        gpu_after = gpu_mon.get_used_mb()

        think_toks = _count_think_tokens(tokenizer, full_output)

        # Update aggregates
        total_wall += dt
        total_prompt_toks += int(prompt_toks or 0)
        total_gen_toks += int(gen_toks or 0)
        total_think_toks += int(think_toks or 0)

        # Update peaks
        for v in (cpu_before, cpu_after):
            if v is not None:
                cpu_peak = max(cpu_peak, v)
        for v in (gpu_before, gpu_after):
            if v is not None:
                gpu_peak = max(gpu_peak, v)

        flush_records.append(
            FlushRecord(
                upto=upto,
                target_idxs=list(pkg.get("target_idxs") or []),
                wall_time_sec=dt,
                prompt_tokens=int(prompt_toks or 0),
                gen_tokens=int(gen_toks or 0),
                think_tokens=int(think_toks or 0),
                cpu_rss_mb_before=cpu_before,
                cpu_rss_mb_after=cpu_after,
                gpu_mem_mb_before=gpu_before,
                gpu_mem_mb_after=gpu_after,
            )
        )

    num_flushes = len(flush_records)
    total_tokens = total_prompt_toks + total_gen_toks

    if total_wall > 0:
        prompt_tps = total_prompt_toks / total_wall
        gen_tps = total_gen_toks / total_wall
        combined_tps = total_tokens / total_wall
        avg_sec_per_flush = total_wall / max(1, num_flushes)
    else:
        prompt_tps = gen_tps = combined_tps = None
        avg_sec_per_flush = None

    avg_think_tokens_per_flush = (
        total_think_toks / num_flushes if num_flushes > 0 else None
    )
    think_share = (
        total_think_toks / total_gen_toks if total_gen_toks > 0 else None
    )

    cpu_end = _cpu_rss_mb()
    gpu_end = gpu_mon.get_used_mb()
    gpu_info = gpu_mon.get_info()
    gpu_mon.shutdown()

    result: Dict[str, Any] = {
        "xml_path": xml_path,
        "model_id": getattr(m1, "MODEL_ID", "<unknown>"),
        "num_events": num_events,
        "num_flushes": num_flushes,
        "timing": {
            "total_wall_time_sec": total_wall,
            "avg_sec_per_flush": avg_sec_per_flush,
            "prompt_tokens_per_sec": prompt_tps,
            "gen_tokens_per_sec": gen_tps,
            "combined_tokens_per_sec": combined_tps,
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_toks,
            "total_gen_tokens": total_gen_toks,
            "total_tokens": total_tokens,
            "total_think_tokens": total_think_toks,
            "avg_think_tokens_per_flush": avg_think_tokens_per_flush,
            "think_token_share_of_gen": think_share,
        },
        "resources": {
            "cpu_rss_mb_start": cpu_start,
            "cpu_rss_mb_end": cpu_end,
            "cpu_rss_mb_peak": cpu_peak if cpu_peak != 0.0 else None,
            "gpu_mem_mb_start": gpu_start,
            "gpu_mem_mb_end": gpu_end,
            "gpu_mem_mb_peak": gpu_peak if gpu_peak != 0.0 else None,
            "gpu_info": gpu_info,
        },
        "flushes": [asdict(fr) for fr in flush_records],
    }

    return result


# ------------------------------
# Entry point
# ------------------------------


def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Single-XML mode (backwards compatible): use m1.XML_PATH
    if path_arg is None:
        xml_path = getattr(m1, "XML_PATH", None)
        if xml_path is None:
            raise SystemExit("No path provided and m1.XML_PATH is not set.")
        xml_path = str(xml_path)
        res = benchmark_single_xml(xml_path)
        out = {
            "mode": "benchmark",
            "multi_xml": False,
            "result": res,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    root = Path(path_arg)

    # If user passes a single XML file
    if root.is_file() and root.suffix.lower() == ".xml":
        res = benchmark_single_xml(str(root))
        out = {
            "mode": "benchmark",
            "multi_xml": False,
            "result": res,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # If user passes a directory, run over all *.xml
    if root.is_dir():
        xml_paths = sorted(root.glob("*.xml"))
        if not xml_paths:
            raise SystemExit(f"No .xml files found in directory: {path_arg}")

        per_xml_results: List[Dict[str, Any]] = []
        overall = {
            "num_xml": 0,
            "total_wall_time_sec": 0.0,
            "total_prompt_tokens": 0,
            "total_gen_tokens": 0,
            "total_think_tokens": 0,
            "total_tokens": 0,
            "total_flushes": 0,
        }

        for xp in xml_paths:
            print(f"[INFO] Benchmarking {xp.name}", file=sys.stderr)
            res = benchmark_single_xml(str(xp))
            per_xml_results.append(res)

            overall["num_xml"] += 1
            overall["total_wall_time_sec"] += res["timing"]["total_wall_time_sec"] or 0.0
            overall["total_prompt_tokens"] += res["tokens"]["total_prompt_tokens"] or 0
            overall["total_gen_tokens"] += res["tokens"]["total_gen_tokens"] or 0
            overall["total_think_tokens"] += res["tokens"]["total_think_tokens"] or 0
            overall["total_tokens"] += res["tokens"]["total_tokens"] or 0
            overall["total_flushes"] += res["num_flushes"] or 0

        # Derived overall rates
        total_wall = overall["total_wall_time_sec"]
        total_prompt = overall["total_prompt_tokens"]
        total_gen = overall["total_gen_tokens"]
        total_tok = overall["total_tokens"]
        total_think = overall["total_think_tokens"]
        total_flushes = overall["total_flushes"]

        if total_wall > 0:
            prompt_tps = total_prompt / total_wall
            gen_tps = total_gen / total_wall
            combined_tps = total_tok / total_wall
            avg_sec_per_flush = total_wall / max(1, total_flushes)
        else:
            prompt_tps = gen_tps = combined_tps = None
            avg_sec_per_flush = None

        if total_flushes > 0:
            avg_think_per_flush = total_think / total_flushes
        else:
            avg_think_per_flush = None

        if total_gen > 0:
            think_share = total_think / total_gen
        else:
            think_share = None

        overall_summary = {
            "num_xml": overall["num_xml"],
            "total_wall_time_sec": total_wall,
            "total_flushes": total_flushes,
            "timing": {
                "avg_sec_per_flush": avg_sec_per_flush,
                "prompt_tokens_per_sec": prompt_tps,
                "gen_tokens_per_sec": gen_tps,
                "combined_tokens_per_sec": combined_tps,
            },
            "tokens": {
                "total_prompt_tokens": total_prompt,
                "total_gen_tokens": total_gen,
                "total_tokens": total_tok,
                "total_think_tokens": total_think,
                "avg_think_tokens_per_flush": avg_think_per_flush,
                "think_token_share_of_gen": think_share,
            },
        }

        out = {
            "mode": "benchmark",
            "multi_xml": True,
            "xml_root": str(root),
            "overall": overall_summary,
            "per_xml_results": per_xml_results,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Provided path is neither an XML file nor a directory: {path_arg}")


if __name__ == "__main__":
    main()
