import argparse
import re
from typing import List, Tuple
import torch
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer

################### Vars ########################

dataset = load_dataset(
    "patea4/educational-ai-agent-small",
    data_files="1721655754.jsonl"
)

MODEL1_PROMPT = "model_0/system_prompt.txt"
MODEL1_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# model = Model0()
output_1 = None
output_2 = None
annot = None

################### Model 0 Helpers ########################

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
    global output_1
    print(f"Writing HLC group {group_id} with {len(hlc_events)} events")
    if output_1 is not None:
        output_1.write(f"<event>\n")
        for event in hlc_events:
            output_1.write(event + "\n")
        output_1.write(f"</event>\n\n")

    grouped_xml = f'<event group="{group_id}">' + "".join(hlc_events) + "</event>\n"
    return grouped_xml

################### Model 1 Helpers ########################

def annotate_event_group(grouped_events, annot):
    annot.add_event(grouped_events)
    idx, depth, summary = annot.annotate_latest()
    print("="*80)
    print(f"[model_1] idx={idx} depth={depth} summary={summary}")
    print("="*80)

def save_model1_annotations(annot, fh):
    content = annot.format_summary_depth_lines()
    fh.write(content)
    fh.flush()
    print("[model_1] Wrote annotation summaries")

################### Main Logic ########################

def process_dataset_examples(system_prompt: str):
    global output_1, output_2, annot

    # fixes a vllm issue when initalizing at import time
    if annot is None:
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from model_1.model1StreamedTest import Model1Annotator
        annot = Model1Annotator(model_id=MODEL1_MODEL)

    print(f"Processing {len(dataset['train'])} examples from dataset")
    
    current_hlc_events = []
    current_group_id = 0

    for idx, example in enumerate(dataset['train']):
        input_xml = example['input']
        expected_output = example['output']

        print(f"Processing example {idx + 1}/{len(dataset['train'])}")

        # prompt = f"{system_prompt}\n\n{input_xml}"

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

            prediction_type, _ = parse_model0_response(expected_output)

            if prediction_type == "NEW":
                if current_hlc_events:
                    print(f"Annotating Grouped Event...")
                    grouped_events = send_grouped_event_model1(current_hlc_events, current_group_id)
                    annotate_event_group(grouped_events, annot)

                current_group_id += 1
                current_hlc_events = [clean_event]
            else:
                current_hlc_events.append(clean_event)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(dataset['train'])} examples")

    if current_hlc_events:
        grouped_events = send_grouped_event_model1(current_hlc_events, current_group_id)
        annotate_event_group(grouped_events, annot)
    
    save_model1_annotations(annot, output_2)

def main():
    global output_1, output_2  # mark them as globals so assignments affect the module-level vars

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model1", required=True, help="Path to grouped events XML (e.g., grouped_events.xml)")
    parser.add_argument("--output_model2", required=True, help="Path to annotated outputs")
    args = parser.parse_args()

    with open(MODEL1_PROMPT, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"Writing grouped events to: {args.output_model1}")
    print(f"Writing annotations to: {args.output_model2}")

    output_1 = open(args.output_model1, "w", encoding="utf-8")
    output_2 = open(args.output_model2, "w", encoding="utf-8")

    try:
        process_dataset_examples(system_prompt)
    finally:
        output_1.close()
        output_2.close()

    print("Grouping complete.\n")


if __name__ == "__main__":
    main()
