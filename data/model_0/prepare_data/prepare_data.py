#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Iterator
from tqdm import tqdm
from transformers import AutoTokenizer

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


def pretokenize_chunks(tokenizer, chunks: List[str]) -> List[int]:
    return [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks]

def alpaca_len_pretokenized(
    template_overhead: int,
    instruction_tokens: int,
    input_tokens: int,
    output_tokens: int
) -> int:
    return template_overhead + instruction_tokens + input_tokens + output_tokens

def stream_examples_simple(
    tokenizer,
    xml_path: str,
    out_path: str,
    sys_prompt: str,
    target_tokens: int,
) -> Iterator[dict]:
    chunks = chunk_input(xml_path)
    groups = build_groups(out_path, chunks)
    n = len(chunks)

    chunk_token_lens = pretokenize_chunks(tokenizer, chunks)

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    template_parts = f"""{system_prompt}

### Instruction:


### Input:


### Response:
"""
    template_overhead_tokens = len(tokenizer.encode(template_parts, add_special_tokens=True)) + 1

    instruction_tokens = len(tokenizer.encode(sys_prompt, add_special_tokens=False))

    output_new_tokens = len(tokenizer.encode("Answer: NEW", add_special_tokens=False))

    newline_tokens = len(tokenizer.encode("\n", add_special_tokens=False))

    for i in tqdm(range(n), total=n, desc=f"{os.path.basename(xml_path)} chunks", unit="ev", leave=False):
        current_txt = chunks[i].replace(">", ' sortme="True">', 1)
        current_tokens = len(tokenizer.encode(current_txt, add_special_tokens=False))

        if i == 0:
            target = "Answer: NEW"
            output_tokens = output_new_tokens
        else:
            if groups[i] == groups[i-1]:
                target = f"Answer: {groups[i]}"
                output_tokens = len(tokenizer.encode(target, add_special_tokens=False))
            else:
                target = "Answer: NEW"
                output_tokens = output_new_tokens

        prior_indices = []
        prior_token_count = current_tokens

        for j in range(i - 1, -1, -1):
            chunk_with_group_tokens = chunk_token_lens[j] + 15  # rough overhead for group attribute
            trial_tokens = prior_token_count + chunk_with_group_tokens + newline_tokens

            total_tokens = alpaca_len_pretokenized(
                template_overhead_tokens,
                instruction_tokens,
                trial_tokens,
                output_tokens
            )

            if total_tokens <= target_tokens:
                prior_indices.insert(0, j)
                prior_token_count = trial_tokens
            else:
                break

        prior = [chunks[j].replace(">", f' group="{groups[j]}">', 1) for j in prior_indices]
        input_text = "\n".join(prior + [current_txt]) if prior else current_txt

        input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
        while alpaca_len_pretokenized(template_overhead_tokens, instruction_tokens, input_tokens, output_tokens) > target_tokens:
            if "\n" not in input_text:
                input_text = current_txt
                break
            parts = input_text.split("\n")
            if len(parts) <= 1:
                input_text = current_txt
                break
            input_text = "\n".join(parts[1:])
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))

        yield {
            "instruction": sys_prompt,
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
        for ex in stream_examples_simple(
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
    ap.add_argument("--target_tokens", type=int, default=2048,
                    help="Cap for full Alpaca prompt")
    args = ap.parse_args()

    sys_prompt = read_text(args.system_prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)

    outputs_map = {get_base(f): os.path.join(args.outputs, f)
                   for f in os.listdir(args.outputs)}

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
