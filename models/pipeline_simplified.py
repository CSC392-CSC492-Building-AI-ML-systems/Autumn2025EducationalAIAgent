import argparse
import re
from typing import List, Tuple
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



model = Model0()

output = None

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
    global output
    print(f"Writing HLC group {group_id} with {len(hlc_events)} events")
    if output is not None:
        output.write(f"<event>\n")
        for event in hlc_events:
            output.write(event + "\n")
        output.write(f"</event>\n\n")


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

        # if output:
        #     output.write(f"\n=== Example {idx + 1} ===\n")
        #     output.write("INPUT:\n")
        #     output.write(input_xml)
        #     output.write("\n\nEXPECTED OUTPUT:\n")
        #     output.write(expected_output)
        #     output.write("\n\nMODEL OUTPUT:\n")
        #     output.write(response)
        #     output.write("\n" + "="*50 + "\n")
        #     output.flush()

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
    global output

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to output file")
    args = parser.parse_args()

    system_prompt = open("model_0/system_prompt.txt").read()

    output = open(args.output, "w")

    process_dataset_examples(system_prompt)

    output.close()


if __name__ == "__main__":
    main()
