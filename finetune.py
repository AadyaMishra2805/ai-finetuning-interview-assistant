from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# MODEL NAME
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# MODEL
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LOAD DATASET
dataset = load_dataset(
    "json",
    data_files="dataset/interview_data.json"
)

# FORMAT DATA
def formatting_func(example):
    return {
        "text": f"Instruction: {example['instruction']}\nResponse: {example['output']}"
    }

dataset = dataset.map(formatting_func)

# LoRA CONFIG
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# APPLY LoRA
model = get_peft_model(model, lora_config)

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4
)

# TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

# TRAIN
trainer.train()

# SAVE MODEL
model.save_pretrained("models/fine_tuned_model")
tokenizer.save_pretrained("models/fine_tuned_model")

print("Fine-tuning completed successfully!")