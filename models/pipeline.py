import argparse
import re
from typing import List, Tuple
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


class Model0:
    def __init__(self):
        print("Loading model...")

        adapter_id = "patea4/deepseek-r1-educational-lora-tuned3"
        base_id = "unsloth/DeepSeek-R1-Distill-Llama-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token


        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id,
            device_map="cuda:0",
        )


        # from transformers import AutoModelForCausalLM
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     base_id,
        #     load_in_4bit=True,
        #     device_map="cuda:0",
        # )

        print("Model loaded!")

    def generate(self, prompt: str) -> str:
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer:"},
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
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()



model = Model0()

output = None

def load_xml_chunks(xml_path: str) -> List[str]:
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"<(?:user_input|system_output)\b[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
        flags=re.DOTALL,
    )
    return pattern.findall(content)

def build_model0_prompt(context_events, current_event, system_prompt, max_context: int = 20) -> str:
    input_parts = []
    recent_events = context_events[-max_context:] if len(context_events) > max_context else context_events

    for prev_event_xml, group_id in recent_events:
        xml_with_group = prev_event_xml.replace(">", f' group="{group_id}">', 1)
        input_parts.append(xml_with_group)

    current_with_sortme = current_event.replace(">", ' sortme="True">', 1)
    input_parts.append(current_with_sortme)
    input_xml = "\n".join(input_parts)

    user_msg = f"{system_prompt}\n\n{input_xml}"

    tok = model.tokenizer
    while len(tok.encode(user_msg, add_special_tokens=False)) > 2048 and len(input_parts) > 1:
        input_parts.pop(0)
        input_xml = "\n".join(input_parts)
        user_msg = f"### Instruction:\n{system_prompt}\n\n### Input:\n{input_xml}"

    return user_msg


def model0_call_api(prompt: str, idx: int) -> str:
    global output
    print(f"Calling model for event {idx}")

    if output is not None:
        output.write(f"\n=== Event {idx} ===\n")
        output.write("INPUT:\n")
        output.write(prompt)
        output.write("\n\nOUTPUT:\n")

    response = model.generate(prompt)
    print(response)

    output.write(response)
    output.write("\n" + "="*50 + "\n")
    output.flush()

    return response 


def parse_model0_response(response: str, current_group_id: int) -> Tuple[str, int]:
    match = re.search(r"Answer:\s*(NEW|\d+)", response, re.IGNORECASE)

    if not match:
        # If not found assume its the same group
        return "EXISTING", current_group_id

    answer = match.group(1)

    if answer.upper() == "NEW":
        return "NEW", 0
    else:
        return "EXISTING", current_group_id


def send_grouped_event_model1(hlc_events: List[str], group_id: int):
    global output
    print(f"Writing HLC group {group_id} with {len(hlc_events)} events")
    if output is not None:
        output.write(f"<event>\n")
        for event in hlc_events:
            output.write(event + "\n")
        output.write(f"</event>\n\n")


def process_event(event_xml: str, context_events: List[Tuple[str, int]],
                  current_hlc_events: List[str], current_group_id: int,
                  system_prompt: str, idx: int) -> Tuple[List[str], int, str]:

    prompt = build_model0_prompt(context_events, event_xml, system_prompt)
    response = model0_call_api(prompt, idx)
    pred_type, group_num = parse_model0_response(response, current_group_id)

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
    print(f"Processing {len(events)} events...")
    context_events = []
    current_hlc_events = []
    current_group_id = -1

    for idx, event_xml in enumerate(events):
        if idx == 0:
            current_group_id = 0
            current_hlc_events = [event_xml]
            context_events.append((event_xml, current_group_id))
        else:
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

    system_prompt = open("model_0/system_prompt2.txt").read()

    events = load_xml_chunks(args.input)

    output = open(args.output, "w")

    stream_events(events, system_prompt)

    output.close()


if __name__ == "__main__":
    main()
