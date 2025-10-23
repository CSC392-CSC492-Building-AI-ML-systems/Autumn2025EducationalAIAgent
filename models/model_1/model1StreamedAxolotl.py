"""
Model-1 preprocessing script for Axolotl (JSON-output training)

- Loads HF dataset: redaMM/educational-ai-agent-small-annotation-depth (split=train)
- Uses formatting_func() to build `messages` with the *JSON* output style
- Writes ./prepared_train.jsonl that Axolotl can ingest with type: chat_template
- Run:  python model1StreamedAxolotl_json.py
"""

import json
from typing import Dict, List
from datasets import load_dataset

# ==============================
# Parameters
# ==============================
DATASET_ID = "redaMM/educational-ai-agent-small-annotation-depth"
DATASET_SPLIT = "train"
OUTPUT_JSONL = "prepared_train.jsonl"

# Prompt knobs (mirrors your runtime annotator)
INCLUDE_FEWSHOTS_DEFAULT = True
SUMMARY_WORD_LIMIT = 50
MAX_NEIGH = 16  # cap neighbor_tail length
DEBUG = True
# ------------------------------
# Statics
# ------------------------------
FEWSHOTS_BLOCK = f"""
examples (for format/logic only — do not output these)

note: event xml often shows keystroke-by-keystroke input with echoed characters, not a full command.

example a — new nested subtask (depth = -1)
neighbor_tail:
- id=10 depth=0  summary=open editor on /etc/ntopng/ntopng.conf
- id=11 depth=0  summary=navigate within config to log options
currDepth before target: -1
input xml:
<event>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>!</user_input><system_output>!</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>-</user_input><system_output>-</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
</event>
output (two lines):
spawn shell from editor to grep syslog for errors.
-1

example b — same-level continuation (depth = 0)
neighbor_tail:
- id=20 depth=-1 summary:spawn shell from editor
- id=21 depth=0  summary:view /var/log/syslog in pager
currDepth before target: -2
input xml:
<event>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>v</user_input><system_output>v</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <system_output>--- syslog ---</system_output>
</event>
output (two lines):
view syslog content within the spawned shell.
0

example c — exit one level (depth = +1)
neighbor_tail:
- id=30 depth=-1 summary:open file in editor from shell
- id=31 depth=0  summary:edit configuration options
currDepth before target: -1
input xml:
<event>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>w</user_input><system_output>w</system_output>
  <user_input>q</user_input><system_output>q</system_output>
  <system_output>[wrote config] demo@host:/etc/ntopng$ </system_output>
</event>
output (two lines):
save changes and exit the editor back to the shell.
1

example d — same-level keystroke sequence (depth = 0)
neighbor_tail:
- id=40 depth=0  summary:interactive shell at home directory
- id=41 depth=0  summary:prepare to connect to a remote host
currDepth before target: 0
input xml:
<event>
  <system_output>demo@boxtop:~$ </system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>7</user_input><system_output>7</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>3</user_input><system_output>3</system_output>
  <user_input>8</user_input><system_output>8</system_output>
</event>
output (two lines):
initiate ssh connection to 10.0.7.138.
0

example e — explicit address allowed (depth = 0)
neighbor_tail:
- id=50 depth=0  summary:running curl to query service endpoint
- id=51 depth=0  summary:checking api health for troubleshooting
currDepth before target: 0
input xml:
<event>
  <system_output>demo@host:~$ </system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>r</user_input><system_output>r</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>9</user_input><system_output>9</system_output>
  <user_input>2</user_input><system_output>2</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>1</user_input><system_output>1</system_output>
  <user_input>6</user_input><system_output>6</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>5</user_input><system_output>5</system_output>
  <user_input>:</user_input><system_output>:</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>8</user_input><system_output>8</system_output>
  <user_input>0</user_input><system_output>0</system_output>
  <user_input>/</user_input><system_output>/</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>s</user_input><system_output>s</system_output>
</event>
output (two lines):
fetch service status from http://192.168.0.5:8080/status.
0

example f — package install with apt (depth = 0)
neighbor_tail:
- id=70 depth=0  summary:initiate SSH connection to 10.0.7.138
- id=71 depth=-1 summary:authenticate and open remote shell
currDepth before target: -1
<event depth="-2">
  <system_output timestamp="0.0">user@host:~$ </system_output>
  <user_input>s</user_input>
  <system_output>s</system_output>
  <user_input>u</user_input>
  <system_output>u</system_output>
  <user_input>d</user_input>
  <system_output>d</system_output>
  <user_input>o</user_input>
  <system_output>o</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>a</user_input>
  <system_output>a</system_output>
  <user_input>p</user_input>
  <system_output>p</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>i</user_input>
  <system_output>i</system_output>
  <user_input>n</user_input>
  <system_output>n</system_output>
  <user_input>s</user_input>
  <system_output>s</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input>a</user_input>
  <system_output>a</system_output>
  <user_input>l</user_input>
  <system_output>l</system_output>
  <user_input>l</user_input>
  <system_output>l</system_output>
  <user_input> </user_input>
  <system_output> </system_output>
  <user_input>h</user_input>
  <system_output>h</system_output>
  <user_input>t</user_input>
  <system_output>t</system_output>
  <user_input>o</user_input>
  <system_output>o</system_output>
  <user_input>p</user_input>
  <system_output>p</system_output>
</event>
output (two lines):
User installs htop package using sudo apt
0
""".strip()


# ------------------------------
# Helpers
# ------------------------------
def _normalize_pkg(ex: Dict) -> Dict:
    """Normalize dataset example to the prompt package."""
    curr_depth = int(ex["currDepth"])
    if curr_depth > 0:
        # Your data shouldn’t have positive currDepth; keep the guard.
        raise ValueError(f"currDepth should be <= 0, got {curr_depth}")

    neighbor_tail = []
    for n in ex.get("neighbors", []):
        neighbor_tail.append({
            "id": int(n["id"]),
            "depth": int(n["depth"]),
            "summary": n["annotation"],
        })

    target_id = int(ex["target_idx"])
    target_xml = ex["target_xml"]
    return {
        "currDepth": curr_depth,
        "neighbor_tail": neighbor_tail,
        "target_events": [{"id": target_id, "xml": target_xml}],
        "target_idxs": [target_id],
    }

def build_instruction_json(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    """Prompt that matches the runtime annotator’s JSON-output spec."""
    # Neighbors
    neigh_lines: List[str] = []
    for n in pkg["neighbor_tail"]:
        neigh_lines.append(f"- id={n['id']} depth={n['depth']} summary={n['summary']}")
    neighbors_text = "\n".join(neigh_lines) if neigh_lines else "    <neighbor>(none)</neighbor>"

    # Targets
    targets_xml = "\n".join(
        [f'  <target id="{t["id"]}">\n{t["xml"]}\n  </target>' for t in pkg["target_events"]]
    )

    # Extra instructions (mirror your runtime “v2” style)
    extra_rules = (
        "- do not copy xml tags or attributes; no repeated phrases\n"
        "- do not mention an address that was not explicitly mentioned in the event\n"
        "- if the target event contains an <annotation> tag or depth value ignore it"
    )

    examples = f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if use_fewshots else ""

    prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {SUMMARY_WORD_LIMIT} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Use the <think>...</think> section to analyze what is happening in the event and assess whether the event starts a nested subtask (-1), continues at the same level (0), or exits one or more levels up (k).
- In <think>...</think>, generate a concise summary at a higher level, considering broader context.
- Use neighbors ONLY for continuity; do not invent context. Keep <think> concise.
</think_first>

<rules>
- the user’s keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent, 0 for same level, >0 to exit levels
- maintain stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth
- write action-oriented summaries; avoid “user”, “they”, “typed”, “inputs”, “enters a command
- depth is relative to the previous events and nothing else”
{extra_rules}
</rules>{examples}

<instruction>
for each target_event, output exactly one json with "annotation" first, then "depth".
</instruction>

<inputs>
  <curr_depth_max>0</curr_depth_max>
  <neighbors>
{neighbors_text}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""
    return prompt

def build_answer_json(ex: Dict) -> str:
    """Gold assistant message: single JSON object."""
    ann = ex["annotation"].strip()
    dep = int(ex["depth"])
    return json.dumps({"annotation": ann, "depth": dep}, ensure_ascii=False)

def formatting_func(ex: Dict):
    # Trim long neighbor tails to keep tokens under control
    if len(ex.get("neighbors", [])) > MAX_NEIGH:
        ex = dict(ex)
        ex["neighbors"] = ex["neighbors"][-MAX_NEIGH:]

    pkg = _normalize_pkg(ex)
    user_prompt = build_instruction_json(pkg, use_fewshots=False)  # you can switch to True if you want the fewshots in-train

    return {
        "messages": [
            {"role": "system", "content": "You are Model-1. Follow formatting strictly."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": build_answer_json(ex)},
        ]
    }

# ------------------------------
# Pre-materialization runner
# ------------------------------
def _write_prepared_jsonl():
    print(f"[pre] Loading dataset: {DATASET_ID} (split={DATASET_SPLIT})")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    print(f"rows: {ds.num_rows}")
    first = ds[0]
    print(f"headers: {first.keys()}")
    print(f"schema: {ds.features}")

    total, skipped = 0, 0
    print(f"[pre] Writing: {OUTPUT_JSONL}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ex in ds:
            try:
                rec = formatting_func(ex)
                if not rec or not isinstance(rec.get("messages"), list) or not rec["messages"]:
                    skipped += 1
                    continue
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
                total += 1
            except Exception:
                skipped += 1
                continue

    print(f"[pre] Done. Wrote {total} records to {OUTPUT_JSONL}. Skipped {skipped} rows.")

if __name__ == "__main__":
    _write_prepared_jsonl()
