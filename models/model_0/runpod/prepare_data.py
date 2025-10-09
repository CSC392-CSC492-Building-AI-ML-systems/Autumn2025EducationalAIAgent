#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Iterator
from bisect import bisect_right
from tqdm import tqdm
from transformers import AutoTokenizer

MARGIN = 60  # target_tokens - MARGIN is the cap

CHUNK_RE = re.compile(
    r"<(?:user_input|system_output)\b[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
    flags=re.DOTALL,
)

def get_base(p: str) -> str:
    return os.path.basename(p).rsplit(".", 2)[0]

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def read_lines(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return f.readlines()

def chunk_input(xml_path: str) -> List[str]:
    return CHUNK_RE.findall(read_text(xml_path))

def build_groups(output_path: str, chunks: List[str]) -> List[int]:
    out_lines = read_lines(output_path)
    groups, current_group, curr_line = [], 0, 2
    for ch in chunks:
        curr_line += len(ch.splitlines())
        groups.append(current_group)
        if curr_line < len(out_lines) and out_lines[curr_line].strip() == "0":
            current_group += 1
    return groups

def tok_len(tok, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))

def pretokenize_chunks(tok, chunks: List[str], groups: List[int]):
    grouped_txt = []
    grouped_len = []
    sortme_txt = []
    sortme_len = []

    for ch, g in zip(chunks, groups):
        gtxt = ch.replace(">", f' group="{g}">', 1)
        stxt = ch.replace(">", ' sortme="True">', 1)
        grouped_txt.append(gtxt)
        sortme_txt.append(stxt)
        grouped_len.append(tok_len(tok, gtxt) + 1)
        sortme_len.append(tok_len(tok, stxt))

    rev_grouped_len = list(reversed(grouped_len))
    rev_cumsum = [0] * len(rev_grouped_len)
    running = 0
    for i, L in enumerate(rev_grouped_len):
        running += L
        rev_cumsum[i] = running

    return grouped_txt, grouped_len, sortme_txt, sortme_len, rev_cumsum

def find_max_prior(rev_cumsum: List[int], budget_for_prior: int) -> int:
    idx = bisect_right(rev_cumsum, budget_for_prior)  # position to insert
    return idx  # k = number of prior events to include

def stream_examples_fast(
    tokenizer,
    xml_path: str,
    out_path: str,
    sys_prompt: str,
    target_tokens: int,
) -> Iterator[dict]:
    chunks = chunk_input(xml_path)
    groups = build_groups(out_path, chunks)

    instruction = sys_prompt.strip() + "\n\n"
    instr_len = tok_len(tokenizer, instruction)

    # Pre-tokenize
    grouped_txt, grouped_len, sortme_txt, sortme_len, rev_cumsum = pretokenize_chunks(
        tokenizer, chunks, groups
    )

    n = len(chunks)
    for i in tqdm(range(n), total=n, desc=f"{os.path.basename(xml_path)} chunks", unit="ev", leave=False):
        current_txt = sortme_txt[i]
        current_toks = sortme_len[i]

        if i == 0:
            target = "Answer: NEW"
        else:
            target = f"Answer: {groups[i]}" if groups[i] == groups[i-1] else "Answer: NEW"

        budget_prior = target_tokens - MARGIN - instr_len - current_toks
        if budget_prior <= 0 or i == 0:
            input_text = current_txt
        else:
            lo, hi = 0, i  # k in [0..i]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if rev_cumsum[mid - 1] <= budget_prior:
                    lo = mid
                else:
                    hi = mid - 1
            k = lo  # number of prior events we can include

            if k == 0:
                input_text = current_txt
            else:
                # Take the last k prior events = grouped_txt[i-k : i]
                prior_slice = grouped_txt[i-k:i]
                input_text = "\n".join(prior_slice + [current_txt])

        yield {
            "instruction": instruction,
            "input": input_text,
            "output": target,
        }

def write_per_file_streaming(
    tokenizer,
    xml_path: str,
    out_path: str,
    sys_prompt: str,
    out_dir: str,
    target_tokens: int,
    flush_every: int = 10_000,
) -> int:
    base = get_base(xml_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    final_path = os.path.join(out_dir, f"{base}.jsonl")

    rows = 0
    with open(final_path, "w", encoding="utf-8") as fh:
        for ex in stream_examples_fast(
            tokenizer=tokenizer,
            xml_path=xml_path,
            out_path=out_path,
            sys_prompt=sys_prompt,
            target_tokens=target_tokens,
        ):
            fh.write(json.dumps(ex, ensure_ascii=False))
            fh.write("\n")
            rows += 1
            if (rows % flush_every) == 0:
                fh.flush()
                os.fsync(fh.fileno())
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Folder with XML input files")
    ap.add_argument("--outputs", required=True, help="Folder with marker files (*.xml.txt)")
    ap.add_argument("--system_prompt", required=True, help="Path to system_prompt.txt")
    ap.add_argument("--out_dir", required=True, help="Output folder for per-file .jsonl")
    ap.add_argument("--tokenizer_model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="HF model id or local path for tokenizer")
    ap.add_argument("--target_tokens", type=int, default=2000,
                    help="Token cap for instruction + input (MARGIN reserved)")
    args = ap.parse_args()

    sys_prompt = read_text(args.system_prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)

    outputs_map = { get_base(f): os.path.join(args.outputs, f) for f in os.listdir(args.outputs) }
    print(outputs_map)

    input_files = sorted(os.listdir(args.inputs))
    total_rows = 0

    for fname in tqdm(input_files, desc="Processing files", unit="file", mininterval=0.2):
        base = get_base(fname)
        if base not in outputs_map:
            tqdm.write(f"[WARN] No matching output file for {fname}; skipping.")
            continue

        xml_path = os.path.join(args.inputs, fname)
        out_path = outputs_map[base]

        try:
            rows = write_per_file_streaming(
                tokenizer=tokenizer,
                xml_path=xml_path,
                out_path=out_path,
                sys_prompt=sys_prompt,
                out_dir=args.out_dir,
                target_tokens=args.target_tokens,
                flush_every=10_000,
            )
            total_rows += rows
            tqdm.write(f"[OK] {fname} -> {rows} rows -> {os.path.join(args.out_dir, base + '.jsonl')}")
        except Exception as e:
            tqdm.write(f"[ERROR] {fname}: {e}")

    print(f"[DONE] Wrote {total_rows} rows across {args.out_dir}")

if __name__ == "__main__":
    main()
