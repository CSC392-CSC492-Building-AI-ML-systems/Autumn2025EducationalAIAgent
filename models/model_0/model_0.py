from unsloth import FastLanguageModel
import os
import torch
import re
from datasets import Dataset
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer
import argparse

print("Loading...")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainConfig:
    model_name: str = "unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit"
    inputs_folder: str
    outputs_folder: str
    output_dir: str = "./lora_model"

    batch_size: int = 2
    lr: float = 2e-4
    epochs: int = 5
    max_length: int = 1024 * 2

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    logging_steps: int = 10


train_prompt_style = open("system_prompt.txt").read()


def get_base_filename(filepath):
    return os.path.basename(filepath).split(".")[0]


def chunk_input(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        input_content = file.read()

    pattern = re.compile(
        r"<(?:user_input|system_output)\b[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
        flags=re.DOTALL,
    )
    chunks = pattern.findall(input_content)
    return chunks


def get_output(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        output = file.readlines()
    return output


def group_chunks(input_filepath, output_filepath):
    chunks = chunk_input(input_filepath)
    output = get_output(output_filepath)

    chunk_tuples = []
    current_group = 0
    curr_line = 2

    for chunk in chunks:
        lines = len(chunk.splitlines())
        curr_line += lines

        chunk_tuples.append((chunk, current_group))
        if curr_line < len(output) and output[curr_line] == "0\n":
            current_group += 1

    return chunk_tuples


def format_data(input_path, response_path, paired_inputs, paired_responses):
    events = group_chunks(input_path, response_path)
    accumulated_events = []

    # One training point per event
    for i, (event_xml, group_num) in enumerate(events):
        # Add new event
        input_events = accumulated_events[:]
        current_event = event_xml.replace(">", f' sortme="True">', 1)
        input_events.append(current_event)

        input_xml = "\n".join(input_events)

        # Expected response
        if i == 0:
            output = "Answer: NEW"
        else:
            prev_group = events[i - 1][1]
            if group_num == prev_group:
                output = f"Answer: {group_num}"
            else:
                output = "Answer: NEW"

        # Add to training data
        paired_inputs.append(input_xml)
        paired_responses.append(output)

        # Increment current events
        event_with_group = event_xml.replace(">", f' group="{group_num}">', 1)
        accumulated_events.append(event_with_group)

    return


def load_paired_data(inputs_dir, outputs_dir):
    """
    Loads and pairs data by matching basenames from separate input and output folders.
    """
    print(f"Looking for inputs in: {inputs_dir}")
    print(f"Looking for outputs in: {outputs_dir}")

    # Create a map of {basename: full_path} for all files in the outputs directory
    response_map = {
        get_base_filename(f): os.path.join(outputs_dir, f)
        for f in os.listdir(outputs_dir)
    }

    paired_inputs = []
    paired_responses = []

    # Iterate through the inputs directory to find matches
    for filename in os.listdir(inputs_dir):
        base_name = get_base_filename(filename)

        # If a matching basename is found in the response map, pair them
        if base_name in response_map:
            input_path = os.path.join(inputs_dir, filename)
            response_path = response_map[base_name]

            print(f"✅ Matched: {filename} -> {os.path.basename(response_path)}")

            format_data(input_path, response_path, paired_inputs, paired_responses)

        else:
            print(f"⚠️ Warning: No matching output found for input file: {filename}")

    return paired_inputs, paired_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
    )
    parser.add_argument(
        "--outputs",
        type=str,
    )
    args = parser.parse_args()

    if not args.inputs:
        print("ERROR: --inputs missing")

    if not args.outputs:
        print("ERROR: --inputs missing")

    config = TrainConfig()
    config.inputs_folder = args.inputs
    config.outputs_folder = args.outputs

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    model_lora = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    # Prepare dataset
    inputs, responses = load_paired_data(config.inputs_folder, config.outputs_folder)
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for inp, out in zip(examples["Input"], examples["Response"]):
            texts.append(train_prompt_style.format(inp, out) + EOS_TOKEN)
        return {"text": texts}

    raw_ds = Dataset.from_dict({"Input": inputs, "Response": responses})
    formatted_ds = raw_ds.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model_lora,
        train_dataset=formatted_ds,
        dataset_text_field="text",
        max_seq_length=config.max_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=config.epochs,
            warmup_steps=5,
            learning_rate=config.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=config.output_dir,
            report_to="none",
        ),
    )

    # Train
    trainer.train()

    # output_gguf_path = os.path.join(config.output_dir, "model.gguf")
    # model.save_pretrained_gguf(output_gguf_path, tokenizer)
    # print(f"Merged full model saved in GGUF format at {config.output_dir}/model.gguf")
    print(
        "Skipping save_pretrained_gguf() in script. Appears to be some issue with unsloth and llamacpp which I was not able to solve.\n For now the model can be converted to gguf using:\n python3 llama.cpp/convert_hf_to_gguf.py --outtype f16 ./lora_model/merged_f16 --outfile ./lora_model/model.f16.gguf"
    )


if __name__ == "__main__":
    main()
