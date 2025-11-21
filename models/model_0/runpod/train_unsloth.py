from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

dataset = load_dataset("Profit967/educational-ai-agent-final", split="train")


EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        user_prompt = f"{instruction}\n\n{input_text}"

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(text)
        text = text + tokenizer.eos_token
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=None,
    dataset_num_proc=2,
    packing=True,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=0,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_steps=50,
        save_total_limit=2,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

model.push_to_hub("patea4/deepseek-r1-educational-lora-tuned")
tokenizer.push_to_hub("patea4/deepseek-r1-educational-lora-tuned")
