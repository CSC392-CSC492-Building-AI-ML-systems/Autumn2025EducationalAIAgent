from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load dataset
dataset = load_dataset("patea4/educational-ai-agent-small", split="train")

# Alpaca prompt format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(
            instruction,
            input_text if input_text else "",
            output
        ) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Use UnslothTrainer (faster than HF Trainer)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=None,
    dataset_num_proc=2,
    packing=False,  # Notebook uses False for reasoning models
    args=UnslothTrainingArguments(  # Use UnslothTrainingArguments!
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Disable wandb
        save_steps=1000,
        save_total_limit=2,
    ),
)

# Train with Unsloth optimizations
trainer_stats = trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

model.push_to_hub("patea4/deepseek-r1-educational-lora")
tokenizer.push_to_hub("patea4/deepseek-r1-educational-lora")
