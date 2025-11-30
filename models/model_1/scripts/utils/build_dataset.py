#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from lxml import etree

REC_BLOCK_RE = re.compile(r"<recording\b[^>]*>.*?</recording>", re.DOTALL)
_MINIFY_WS_RE = re.compile(r"\s+")

# Terminal noise scrubbers (safe to keep)
_DEC_PRIV = re.compile(r"\[\?\d{1,3}[hl]")           # e.g., [?2004h or [?25l
_CSI_SEQ  = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]") # CSI like \x1b[31m or \x1b[?2004l
_OSC_SEQ  = re.compile(r"\x1B\][^\a]*\a")            # OSC ... BEL
_CTRL     = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # control chars except \t \n \r

@dataclass
class Event:
    idx: int
    xml: str

def sanitize_term_noise(s: str) -> str:
    s = _DEC_PRIV.sub("", s)
    s = _CSI_SEQ.sub("", s)
    s = _OSC_SEQ.sub("", s)
    s = _CTRL.sub("", s)
    return s

def minify_xml(xml: str) -> str:
    """
    Preserve timestamp="..." attributes; just scrub terminal noise and compact whitespace.
    """
    x = sanitize_term_noise(xml)
    x = _MINIFY_WS_RE.sub(" ", x).strip()
    return x

def extract_recording_block(text: str) -> str:
    m = REC_BLOCK_RE.search(text)
    if not m:
        raise ValueError("No <recording>...</recording> block found")
    return m.group(0)

def load_events(xml_path: str) -> List[Event]:
    raw = open(xml_path, "r", encoding="utf-8", errors="ignore").read()
    rec = extract_recording_block(raw)
    root = etree.fromstring(rec.encode("utf-8"))
    evs: List[Event] = []
    for i, ev in enumerate(root.findall(".//event")):
        xml_str = etree.tostring(ev, encoding="unicode")
        evs.append(Event(idx=i, xml=minify_xml(xml_str)))
    return evs

def parse_training_txt(txt_path: str) -> List[Tuple[int, str]]:
    """
    training file is alternating lines: depth (int), then annotation (string).
    Forgiving about blanks.
    """
    lines = [ln.strip() for ln in open(txt_path, "r", encoding="utf-8", errors="ignore").read().splitlines()]
    pairs: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        while i < len(lines) and lines[i] == "":
            i += 1
        if i >= len(lines): break
        if not re.fullmatch(r"-?\d+", lines[i]):
            i += 1
            continue
        depth = int(lines[i]); i += 1
        ann = ""
        if i < len(lines) and not re.fullmatch(r"-?\d+", lines[i]):
            ann = lines[i].strip()
            i += 1
        pairs.append((depth, ann))
    return pairs

def compute_curr_depths(depths: List[int]) -> List[int]:
    """
    currDepth[t] is stack depth BEFORE applying depths[t].
    Start at 0; -1 descends; >0 ascends; clamp to never > 0.
    """
    currs: List[int] = []
    curr = 0
    for d in depths:
        currs.append(curr)
        if d == -1:
            curr -= 1
        elif d > 0:
            curr += d
        if curr > 0:
            curr = 0
    return currs

def make_neighbors(idx: int, anns: List[Optional[str]], deps: List[Optional[int]]) -> List[Dict]:
    return [{"id": i, "depth": deps[i], "annotation": anns[i]} for i in range(0, idx)]

def key_from_filename(path: str) -> str:
    """
    Shared key = filename (without extension) up to the LAST underscore.
    e.g.,
      1728501686_parsed.xml     → 1728501686
      sessionA_part1_parsed.xml → sessionA_part1
      1728501686_training.txt   → 1728501686
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    # split from the right once; if no underscore, use the whole stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_rows(xml_path: str, txt_path: str, file_id: Optional[str] = None) -> List[Dict]:
    evs = load_events(xml_path)
    pairs = parse_training_txt(txt_path)

    if len(pairs) != len(evs):
        print(f"[warn] {os.path.basename(xml_path)}: events={len(evs)} training_pairs={len(pairs)} (aligning to min length)")

    n = min(len(evs), len(pairs))
    depths = [pairs[i][0] for i in range(n)]
    annots = [pairs[i][1] for i in range(n)]
    currs = compute_curr_depths(depths)

    fid = file_id or os.path.splitext(os.path.basename(xml_path))[0]
    rows: List[Dict] = []
    for i in range(n):
        rows.append({
            "file_id": fid,
            "target_idx": i,
            "currDepth": currs[i],
            "neighbors": make_neighbors(i, annots, depths),
            "target_xml": evs[i].xml,        # timestamps preserved
            "annotation": annots[i],
            "depth": depths[i],
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="directory containing *_parsed.xml")
    ap.add_argument("--gt_dir",  required=True, help="directory containing *_training.txt")
    ap.add_argument("--out",     required=True, help="output JSONL path")
    args = ap.parse_args()

    xml_paths = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    if not xml_paths:
        raise SystemExit(f"No XML files found in {args.xml_dir}")

    # Build map from key → gt path
    gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, "*.txt")))
    gt_map: Dict[str, str] = {key_from_filename(p): p for p in gt_paths}

    total = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for xp in xml_paths:
            k = key_from_filename(xp)
            tp = gt_map.get(k)
            if not tp:
                print(f"[skip] No matching GT for XML {os.path.basename(xp)} (key={k}) in {args.gt_dir}")
                continue

            rows = build_rows(xp, tp, file_id=k)
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} rows → {args.out}")

if __name__ == "__main__":
    main()