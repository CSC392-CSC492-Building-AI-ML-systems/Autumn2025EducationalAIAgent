import argparse
import re
from typing import List, Tuple
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


class Model0:
    def __init__(self):
        print("Loading model...")

        adapter_id = "patea4/deepseek-r1-educational-lora"
        base_id = "unsloth/DeepSeek-R1-Distill-Llama-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
        self.tokenizer.truncation_side = "left"

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id,
            load_in_4bit=True,
            device_map="cuda:0",
        )

        print("Model loaded!")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.5,
                do_sample=True,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in full_output:
            return full_output.split("### Response:")[-1].strip()
        return full_output


model = Model0()

output = None

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
    """

def load_xml_chunks(xml_path: str) -> List[str]:
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"<(?:user_input|system_output)\b[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
        flags=re.DOTALL,
    )
    return pattern.findall(content)




def build_model0_prompt(context_events: List[Tuple[str, int]], current_event: str, system_prompt: str, max_context: int = 20) -> str:
    input_parts = []

    recent_events = context_events[-max_context:] if len(context_events) > max_context else context_events

    for prev_event_xml, group_id in recent_events:
        xml_with_group = prev_event_xml.replace(">", f' group="{group_id}">', 1)
        input_parts.append(xml_with_group)

    current_with_sortme = current_event.replace(">", ' sortme="True">', 1)
    input_parts.append(current_with_sortme)
    input_xml = "\n".join(input_parts)

    prompt = ALPACA_PROMPT.format(system_prompt, input_xml)

    prompt_tokens = len(model.tokenizer.encode(prompt, add_special_tokens=False))

    while prompt_tokens > 2048 and len(input_parts) > 1:
        input_parts.pop(0)
        input_xml = "\n".join(input_parts)
        prompt = ALPACA_PROMPT.format(system_prompt, input_xml)
        prompt_tokens = len(model.tokenizer.encode(prompt, add_special_tokens=False))

    return prompt


def model0_call_api(prompt: str, idx: int) -> str:
    global output
    response = model.generate(prompt)
    return response 


def parse_model0_response(response: str) -> Tuple[str, int]:
    match = re.search(r"Answer:\s*(NEW|\d+)", response, re.IGNORECASE)

    if not match:
        return "UNKNOWN", 0

    answer = match.group(1)

    if answer.upper() == "NEW":
        return "NEW", 0
    else:
        return "EXISTING", int(answer)


def send_grouped_event_model1(hlc_events: List[str], group_id: int):
    global output
    print(f"Writing HLC group {group_id} with {len(hlc_events)} events")
    if output:
        output.write(f"<event>\n")
        for event in hlc_events:
            output.write(event + "\n")
        output.write(f"</event>\n\n")


def process_event(event_xml: str, context_events: List[Tuple[str, int]],
                  current_hlc_events: List[str], current_group_id: int,
                  system_prompt: str, idx: int) -> Tuple[List[str], int, str]:

    prompt = build_model0_prompt(context_events, event_xml, system_prompt)
    response = model0_call_api(prompt, idx)
    pred_type, group_num = parse_model0_response(response)

    if pred_type == "NEW":
        if current_hlc_events:
            send_grouped_event_model1(current_hlc_events, current_group_id)

        current_group_id += 1
        current_hlc_events = [event_xml]
        context_events.append((event_xml, current_group_id))

    else:
        current_hlc_events.append(event_xml)
        context_events.append((event_xml, group_num))

    return current_hlc_events, current_group_id, pred_type


def stream_events(events: List[str], system_prompt: str):
    context_events = []
    current_hlc_events = []
    current_group_id = -1

    for idx, event_xml in enumerate(events):
        current_hlc_events, current_group_id, _ = process_event(
            event_xml, context_events, current_hlc_events,
            current_group_id, system_prompt, idx
        )
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(events)} events...")

    if current_hlc_events:
        send_grouped_event_model1(current_hlc_events, current_group_id)


def main():
    global output

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    system_prompt = open("model_0/system_prompt.txt").read()

    events = load_xml_chunks(args.input)

    output = open(args.output, "w")

    stream_events(events, system_prompt)

    output.close()


if __name__ == "__main__":
    main()
