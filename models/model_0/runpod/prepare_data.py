#!/usr/bin/env python3
import os, re, json, gzip, argparse
from pathlib import Path
from typing import List
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

def make_examples_for_pair(
    xml_path: str,
    out_path: str,
    sys_prompt: str,
    K: int,
    stride: int,
) -> List[dict]:
    chunks = chunk_input(xml_path)
    groups = build_groups(out_path, chunks)

    accumulated: List[str] = []
    examples: List[dict] = []

    INSTRUCTION = sys_prompt.strip() + "\n\n"

    iterator = zip(chunks, groups)
    desc_name = f"{os.path.basename(xml_path)} chunks"
    iterator = tqdm(list(iterator), desc=desc_name, unit="ev", leave=False)

    for i, (chunk_xml, g) in enumerate(iterator):
        prior = accumulated[-(K-1):] if K > 0 else accumulated[:]
        current = chunk_xml.replace(">", ' sortme="True">', 1)
        user_xml = "\n".join(prior + [current])

        if i == 0:
            target = "Answer: NEW"
        else:
            prev_group = groups[i - 1]
            target = f"Answer: {g}" if g == prev_group else "Answer: NEW"

        examples.append({
            "instruction": INSTRUCTION,
            "input": user_xml,
            "response": target,
        })

        accumulated.append(chunk_xml.replace(">", f' group="{g}">', 1))

    if stride > 1:
        examples = [ex for idx, ex in enumerate(examples) if idx % stride == 0]
    return examples

def write_per_file_gz(examples: List[dict], out_dir: str, base: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}.jsonl.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Folder with XML files")
    ap.add_argument("--outputs", required=True, help="Folder with marker files")
    ap.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    ap.add_argument("--out_dir", required=True, help="Output folder for per-file .jsonl.gz")
    ap.add_argument("--k", type=int, default=0, help="Sliding window size. Use 0 to include all prior events.")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth example")
    args = ap.parse_args()

    sys_prompt = read_text(args.system_prompt)

    outputs_map = { get_base(f): os.path.join(args.outputs, f) for f in os.listdir(args.outputs) }
    print(outputs_map)

    total_rows = 0
    input_files = sorted(os.listdir(args.inputs))

    for fname in tqdm(input_files, desc="Processing files", unit="file"):
        tqdm.write(fname)

        base = get_base(fname)
        if base not in outputs_map:
            tqdm.write(f"[WARN] No matching output file for {fname}; skipping.")
            continue

        xml_path = os.path.join(args.inputs, fname)
        out_path = outputs_map[base]

        exs = make_examples_for_pair(
            xml_path, out_path,
            sys_prompt=sys_prompt,
            K=args.k,
            stride=args.stride,
        )
        out_file = write_per_file_gz(exs, args.out_dir, base)
        total_rows += len(exs)
        tqdm.write(f"[OK] {fname} -> {len(exs)} rows -> {out_file}")

    print(f"[DONE] Wrote {total_rows} rows across {args.out_dir}")

if __name__ == "__main__":
    main()
