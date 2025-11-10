import argparse
import re
from typing import List, Tuple, Dict, Any
import os               # [ADD]
import json             # [ADD]
import requests         # [ADD]
import torch
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset(
    "patea4/educational-ai-agent-small",
    data_files="1721655754.jsonl"
)


class Model0:
    def __init__(self):
        print("Loading model...")

        adapter_id = "patea4/deepseek-r1-educational-lora-tuned"
        base_id = "unsloth/DeepSeek-R1-Distill-Llama-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id,
            device_map="cuda:0",
        )

        print("Model loaded!")

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True,
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()


model = None

output = None
m1_log = None          # [ADD] log Model-1 responses
SYSTEM_PROMPT_M1 = ""  # [ADD] read in main
TAIL_K = 8             # [ADD] neighbors window for Model-1
m1_history: List[Dict[str, Any]] = []  # [ADD] rolling history for neighbors

# [ADD] RunPod Model-1 endpoint/env
RUNPOD_MODEL1_URL = "https://api.runpod.ai/v2/fia5ucxjg7u84k/runsync"
RUNPOD_API_KEY = "REPLACE WITH API KEY"

def rp_call_model1(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    print(f"[DEBUG][M1] POST {RUNPOD_MODEL1_URL}  messages={len(messages)}")
    resp = requests.post(RUNPOD_MODEL1_URL, headers=headers, json={"input": {"messages": messages}})
    print(f"[DEBUG][M1] HTTP {resp.status_code}")
    print(f"[DEBUG][M1] HTTP response: {resp}")
    resp.raise_for_status()
    data = resp.json()
    # RunPod returns {"output": {...}} for /run; extract it if present
    out = data.get("output") or data
    print(f"[DEBUG][M1] Keys in output: {list(out.keys())}")
    return out

def compute_curr_depth_from_history(history: List[Dict[str, Any]]) -> int:
    curr = 0
    for h in history:
        d = h.get("depth", None)
        if d is None:
            continue
        if d == -1:
            curr -= 1
        elif isinstance(d, int) and d > 0:
            curr += d
    if curr > 0:
        curr = 0
    return curr

def call_model1_for_group(group_id: int, group_xml_inner: str) -> Tuple[str, Any, Dict[str, Any]]:
    """Build the exact <inputs> block and call RunPod Model-1. Update history and log.
       Returns (annotation, depth, full_response_dict)."""
    # neighbors xml from tail of history
    neighbors_tail = m1_history[-TAIL_K:]
    neighbors_xml = "\n".join(
        [
            f'        <neighbor id="{t["gid"]}" depth="{t.get("depth", 0)}">{t.get("annotation","")}</neighbor>'
            for t in neighbors_tail
        ]
    ) or "        <neighbor>(none)</neighbor>"

    targets_xml = f'        <target id="{group_id}">\n          <event>\n{group_xml_inner}\n          </event>\n        </target>'

    curr_depth_val = compute_curr_depth_from_history(m1_history)

    user_block = f"""<inputs>
  <curr_depth_max>{curr_depth_val}</curr_depth_max>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>""".strip()

    # ---- DEBUG PRINTS ----
    print(f"[DEBUG][M1] ---- Building prompt for gid={group_id} ----")
    print(f"[DEBUG][M1] neighbors_count={len(neighbors_tail)} tail_ids={[t['gid'] for t in neighbors_tail]}")
    print(f"[DEBUG][M1] curr_depth_max={curr_depth_val}")
    print(f"[DEBUG][M1] group_xml_len(chars)={len(group_xml_inner)} lines={group_xml_inner.count('\\n')+1}")
    # Uncomment to see full block (verbose):
    # print(f"[DEBUG][M1] user_block:\n{user_block}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_M1},
        {"role": "user", "content": user_block},
    ]
    out = rp_call_model1(messages)

    # Expecting {"json": {"annotation":..., "depth":...}, "thinking": "...", "text": "..."}
    js = out.get("json", {}) or {}
    thinking = out.get("thinking", "") or ""
    text = out.get("text", "") or ""

    ann = js.get("annotation", "")
    dep = js.get("depth", None)

    # ---- DEBUG PRINTS on response ----
    print(f"[DEBUG][M1] json_has_annotation={bool(ann)} json_has_depth={dep is not None}")
    print(f"[DEBUG][M1] thinking_len={len(thinking)} text_head={text[:120].replace('\\n',' ')}...")

    if dep is None:
        print(f"[WARN][M1] Depth missing in response JSON for gid={group_id}. JSON={js}")

    # update rolling history
    m1_history.append({"gid": group_id, "annotation": ann, "depth": dep})

    # optional log to .jsonl
    if m1_log is not None:
        m1_log.write(json.dumps({
            "group_id": group_id,
            "annotation": ann,
            "depth": dep,
            "thinking": thinking,
            "text": text
        }, ensure_ascii=False) + "\n")
        m1_log.flush()

    # concise console msg
    print(f"[Model-1] gid={group_id} depth={dep} annotation={ann}")
    return ann, dep, out


sortme_pattern = re.compile(
    r"<(?:user_input|system_output)\b[^>]*?sortme=\"True\"[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
    flags=re.DOTALL,
)

def extract_clean_event(event_with_attrs: str) -> str:
    return re.sub(r'\s+(group|sortme)="[^"]*"', '', event_with_attrs)

def parse_model0_response(text: str) -> Tuple[str, int]:
    match = re.search(r"Answer:\s*(NEW|\d+)", text, re.IGNORECASE)
    if not match:
        return "UNKNOWN", -1
    answer = match.group(1)
    if answer.upper() == "NEW":
        return "NEW", -1
    else:
        return "EXISTING", int(answer)

def send_grouped_event_model1(hlc_events: List[str], group_id: int):
    """(Changed) Only write the Model-1 annotation + depth for this group."""
    global output
    print(f"Finalizing group {group_id} with {len(hlc_events)} low-level events.")

    group_xml_inner = "\n".join(hlc_events)

    # Call Model-1 first (so we can write only annotation + depth)
    ann, dep, _ = call_model1_for_group(group_id, group_xml_inner)

    # ---- WRITE ONLY ANNOTATION + DEPTH (as requested) ----
    if output is not None:
        # Write one line per group (JSON keeps it clean to parse)
        output.write(json.dumps({
            "group_id": group_id,
            "depth": dep,
            "annotation": ann
        }, ensure_ascii=False) + "\n")
        output.flush()


def process_dataset_examples(system_prompt: str):
    global output

    print(f"Processing {len(dataset['train'])} examples from dataset")

    current_hlc_events = []
    current_group_id = 0

    for idx, example in enumerate(dataset['train']):
        input_xml = example['input']
        expected_output = example['output']

        print(f"Processing example {idx + 1}/{len(dataset['train'])}")

        prompt = f"{system_prompt}\n\n{input_xml}"

        # response = model.generate(prompt)
        # print(f"Expected: {expected_output}")
        # print(f"Got: {response}")

        sortme_match = sortme_pattern.search(input_xml)

        if sortme_match:
            sortme_event = sortme_match.group(0)
            clean_event = extract_clean_event(sortme_event)

            predicton_type, _ = parse_model0_response(expected_output)

            if predicton_type == "NEW":
                if current_hlc_events:
                    send_grouped_event_model1(current_hlc_events, current_group_id)
                current_group_id += 1
                current_hlc_events = [clean_event]
            else:
                current_hlc_events.append(clean_event)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(dataset['train'])} examples")

    if current_hlc_events:
        send_grouped_event_model1(current_hlc_events, current_group_id)


def main():
    global output, m1_log, SYSTEM_PROMPT_M1

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--m1-log", default=None, help="Optional path to write Model-1 jsonl logs")  # [ADD]
    parser.add_argument("--m1-system-prompt", default="model_1/system_prompt.txt", help="Path to Model-1 system prompt")  # [ADD]
    args = parser.parse_args()

    system_prompt = ""
    SYSTEM_PROMPT_M1 = open(args.m1_system_prompt).read()  # [ADD]

    # sanity for runpod config
    if not RUNPOD_MODEL1_URL or not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_MODEL1_URL and RUNPOD_API_KEY must be set in the environment.")

    output = open(args.output, "w", encoding="utf-8")
    if args.m1_log:
        m1_log = open(args.m1_log, "w", encoding="utf-8")

    try:
        process_dataset_examples(system_prompt)
    finally:
        output.close()
        if m1_log:
            m1_log.close()


if __name__ == "__main__":
    main()
