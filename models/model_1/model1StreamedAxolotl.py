"""
Model-1 preprocessing script for Axolotl (no CLI args)

- Loads HF dataset: redaMM/educational-ai-agent-small-annotation-depth (split=train)
- Uses formatting_func() to build `messages`
- Writes ./prepared_train.jsonl that Axolotl can ingest with type: chat_template
- Run:  PYTHONPATH=. python model1StreamedAxolotl.py
"""

import json
from typing import Dict, List

# ==============================
# Parameters (edit here only)
# ==============================
DATASET_ID = "redaMM/educational-ai-agent-small-annotation-depth"
DATASET_SPLIT = "train"
OUTPUT_JSONL = "prepared_train.jsonl"
INCLUDE_FEWSHOTS_DEFAULT = True
SUMMARY_WORD_LIMIT = 50
MAX_NEIGH = 16  # cap neighbor_tail length

# ------------------------------
# Static Vars
# ------------------------------

EXAMPLE_NEIGHBORS_TEXT = """- id=20 depth=-1 summary=Enter editing /etc/ntopng/ntopng.conf to tweak capture settings.
- id=21 depth=0  summary=Scroll within the editor; reviewing options.
- id=22 depth=-1 summary=Open a shell from editor to list /var/log for recent errors.
- id=23 depth=1  summary=Return from the shell to the editor context.
- id=24 depth=0  summary=Save the config and keep editing."""

FEWSHOTS_BLOCK = f"""
EXAMPLES (FOR FORMAT/LOGIC ONLY — DO NOT OUTPUT THESE IN YOUR ANSWER)

Example neighbor_tail (dense):
{EXAMPLE_NEIGHBORS_TEXT}

Example A — NEW nested subtask (depth = -1)
Example currDepth before target: -1
Example INPUT XML:
<event><user_input>:!grep -i error /var/log/syslog</user_input><system_output>[shell] matching lines...</system_output></event>
Example OUTPUT (two lines):
Spawn shell from editor to grep syslog for errors.
-1
EXAMPLE RATIONALE (do not output):
Action starts a nested tool (shell) inside the editor workflow; that descends one level → depth = -1.

Example B — Same-level continuation (depth = 0)
Example currDepth before target: -2
Example INPUT XML:
<event><user_input>less /var/log/syslog</user_input><system_output>--- syslog ---</system_output></event>
Example OUTPUT (two lines):
View syslog content within the spawned shell.
0
EXAMPLE RATIONALE (do not output):
Still operating in the same nested shell context (no new subtask, no exit); remain at current level → depth = 0.

Example C — Exit one level (depth = +1)
Example currDepth before target: -1
Example INPUT XML:
<event><user_input>:wq</user_input><system_output>[wrote config] demo@host:/etc/ntopng$ </system_output></event>
Example OUTPUT (two lines):
Save changes and exit the editor back to the shell.
1
EXAMPLE RATIONALE (do not output):
Exiting the editor returns to the parent shell context, popping one level → depth = +1.

Example D — Same-level command (depth = 0)
Example currDepth before target: 0
Example INPUT XML:
<event>  <system_output timestamp="0.096022">[?2004h]0;demo@boxtop: ~demo@boxtop:~$ </system_output>  <user_input timestamp="9.163614">s</user_input>  <system_output timestamp="9.164051">s</system_output>  <user_input timestamp="9.365744">s</user_input>  <system_output timestamp="9.366263">s</system_output>  <user_input timestamp="9.589844">h</user_input>  <system_output timestamp="9.59026">h</system_output>  <user_input timestamp="9.708352"> </user_input>  <system_output timestamp="9.708844"> </system_output>  <user_input timestamp="10.1118">1</user_input>  <system_output timestamp="10.112236">1</system_output>  <user_input timestamp="10.270878">0</user_input>  <system_output timestamp="10.271223">0</system_output>  <user_input timestamp="10.471565">.</user_input>  <system_output timestamp="10.471898">.</system_output>  <user_input timestamp="10.594981">0</user_input>  <system_output timestamp="10.595383">0</system_output>  <user_input timestamp="10.757499">.</user_input>  <system_output timestamp="10.757882">.</system_output>  <user_input timestamp="11.140897">7</user_input>  <system_output timestamp="11.14119">7</system_output>  <user_input timestamp="11.603706">.</user_input>  <system_output timestamp="11.604019">.</system_output>  <user_input timestamp="12.330584">1</user_input>  <system_output timestamp="12.331455">1</system_output>  <user_input timestamp="12.632256">3</user_input>  <system_output timestamp="12.633323">3</system_output>  <user_input timestamp="13.446626">8</user_input>  <system_output timestamp="13.447562">8</system_output>  <user_input timestamp="14.510021"></user_input>  <system_output timestamp="14.511984">[?2004l</system_output></event>
Example OUTPUT (two lines):
User initiates SSH connection to server 10.0.7.138
0
EXAMPLE RATIONALE (do not output):
Typing an ssh command within the same shell does not enter a nested tool yet; it stays at the current level → depth = 0.
""".strip()


# ------------------------------
# Helpers
# ------------------------------

def make_pkg_from_example(ex: Dict) -> Dict:
    return {
        "currDepth": int(ex["currDepth"]),
        "neighbor_tail": [
            {"id": int(n["id"]), "depth": int(n["depth"]), "summary": n["annotation"]}
            for n in ex["neighbors"]
        ],
        "parent_xml": None,
        "target_events": [{"id": int(ex["target_idx"]), "xml": ex["target_xml"]}],
        "target_idxs": [int(ex["target_idx"])],
    }

def build_instruction(pkg: Dict, use_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT) -> str:
    # Neighbors summary (real context)
    neigh_lines = []
    for n in pkg["neighbor_tail"]:
        neigh_lines.append(f"- id={n['id']} depth={n['depth']} summary={n['summary']}")
    neighbors_text = "\n".join(neigh_lines) if neigh_lines else "(none)"

    # Targets block
    targets_block = "\n".join(
        [f"\n<BEGIN_TARGET id={t['id']}>\n{t['xml']}\n<END_TARGET>\n" for t in pkg["target_events"]]
    )

    # Output constraints
    extra = (
        f"\n- Output EXACTLY TWO LINES per target. "
        f"- Line 1 (summary) MUST be ≤ {SUMMARY_WORD_LIMIT} words. Be concise and factual.\n"
        "- Line 2 MUST be a single integer depth (−1, 0, or >0). No extra text.\n"
        "- Do NOT copy XML tags/attributes. No repeated phrases.\n"
    )

    # Core instructions (without examples)
    core = f"""
You are Model-1 (annotator).

Given:
- currDepth (≤ 0): {pkg['currDepth']}
- neighbor_tail (already annotated, for context; may be empty):
{neighbors_text}
- TARGET_EVENT: A single event's XML.

THINK FIRST (hidden):
- Think inside <think>...</think> about what is happening in the event and whether the event starts a nested subtask (→ -1),
  continues at the same level (→ 0), or exits up (→ k).
- Think inside <think>...</think> to understand what is happening at a higher level to generate a summary
- Use neighbors ONLY for continuity; do not invent. Keep the <think> section compact.

Then OUTPUT (exactly two lines; order matters):
1) one-sentence annotation of what happened in the target event (≤ {SUMMARY_WORD_LIMIT} words)
2) a single integer 'depth' (use -1 for subevent, 0 for same level, >0 to exit levels)
{extra}

Rules:
- Output ONLY the two lines (and the final sentinel if required). No numbering, no JSON, no prose.
- Respect the stack invariant: currDepth ≤ 0; if depth == -1 then currDepth -= 1; if depth > 0 then currDepth += depth.
- Never let the running currDepth become > 0.
- Do not simply say what the user is typing or that they are typing something, we want to know what they are doing and the reason behind it.
- Write action-oriented summaries. Do NOT mention “user”, “they”, “typed/typing”, “by typing”, “inputs”, or “enters a command”.
- Start with a verb that describes the action’s intent (e.g., “List…”, “Open…”, “Initiate…”, “Install…”, “Exit…”).
""".strip()

    # Conditionally include examples
    examples_part = f"\n\n{FEWSHOTS_BLOCK}\n" if use_fewshots else ""

    instructions = (
        core + examples_part + "\n\nNow produce the pairs for the targets below:\n" + targets_block
    )
    return instructions

def build_answer(ex: Dict) -> str:
    return ex["annotation"].strip() + "\n" + str(int(ex["depth"])) + "\n"

# Axolotl-style formatter we’ll reuse for pre-materialization
def formatting_func(example_batch: Dict):
    outs = []
    for ex in example_batch["data"]:
        if len(ex.get("neighbors", [])) > MAX_NEIGH:
            ex = dict(ex)
            ex["neighbors"] = ex["neighbors"][-MAX_NEIGH:]

        pkg = make_pkg_from_example(ex)
        user_prompt = build_instruction(pkg, use_fewshots=False)

        outs.append({
            "messages": [
                {"role": "system", "content": "You are Model-1. Follow formatting strictly."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": build_answer(ex)},
            ]
        })
    return outs

# ------------------------------
# Pre-materialization runner
# ------------------------------

def _write_prepared_jsonl():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("`datasets` library not installed. pip install datasets") from e

    print(f"[pre] Loading dataset: {DATASET_ID} (split={DATASET_SPLIT})")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

    total, skipped = 0, 0
    print(f"[pre] Writing: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w") as f:
        for ex in ds:
            try:
                recs = formatting_func({"data": [ex]})
                if not recs:
                    skipped += 1
                    continue
                for r in recs:
                    msgs = r.get("messages")
                    if not isinstance(msgs, list) or not msgs:
                        skipped += 1
                        continue
                    json.dump(r, f)
                    f.write("\n")
                    total += 1
            except Exception:
                skipped += 1
                continue

    print(f"[pre] Done. Wrote {total} records to {OUTPUT_JSONL}. Skipped {skipped} rows.")

# Auto-run when invoked as a script
if __name__ == "__main__":
    _write_prepared_jsonl()
