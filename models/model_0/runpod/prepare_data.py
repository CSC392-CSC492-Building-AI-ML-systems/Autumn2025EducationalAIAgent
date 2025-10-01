#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Iterator
from tqdm import tqdm

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

def stream_examples_all_prior(
    xml_path: str,
    out_path: str,
    sys_prompt: str,
) -> Iterator[dict]:
    chunks = chunk_input(xml_path)
    groups = build_groups(out_path, chunks)

    instruction = sys_prompt.strip() + "\n\n"
    prior_text = ""

    iterator = enumerate(zip(chunks, groups))
    iterator = tqdm(iterator, total=len(chunks), desc=f"{os.path.basename(xml_path)} chunks", unit="ev", leave=False)

    for i, (chunk_xml, g) in iterator:
        current = chunk_xml.replace(">", ' sortme="True">', 1)
        user_xml = prior_text + current

        if i == 0:
            target = "Answer: NEW"
        else:
            prev_g = groups[i - 1]
            target = f"Answer: {g}" if g == prev_g else "Answer: NEW"

        yield {
            "instruction": instruction,
            "input": user_xml,
            "response": target,
        }

        prior_text += chunk_xml.replace(">", f' group="{g}">', 1) + "\n"

def write_per_file_streaming(
    xml_path: str,
    out_path: str,
    sys_prompt: str,
    out_dir: str,
    flush_every: int = 10_000,
) -> int:
    base = get_base(xml_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    final_path = os.path.join(out_dir, f"{base}.jsonl")

    rows = 0
    with open(final_path, "w", encoding="utf-8") as fh:
        for ex in stream_examples_all_prior(
            xml_path=xml_path,
            out_path=out_path,
            sys_prompt=sys_prompt,
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
    args = ap.parse_args()

    sys_prompt = read_text(args.system_prompt)

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
                xml_path=xml_path,
                out_path=out_path,
                sys_prompt=sys_prompt,
                out_dir=args.out_dir,
                flush_every=10_000,
            )
            total_rows += rows
            tqdm.write(f"[OK] {fname} -> {rows} rows -> {os.path.join(args.out_dir, base + '.jsonl')}")
        except Exception as e:
            tqdm.write(f"[ERROR] {fname}: {e}")

    print(f"[DONE] Wrote {total_rows} rows across {args.out_dir}")

if __name__ == "__main__":
    main()
