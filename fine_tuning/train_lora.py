from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def format_example(example: dict) -> dict:
    prompt = example["prompt"]
    response = example["response"]
    text = f"<s>[INST] {prompt} [/INST] {response}</s>"
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino LoRA (exemplo).")
    parser.add_argument("--base_model", required=True, help="Modelo base (ex: mistralai/Mistral-7B-Instruct-v0.2).")
    parser.add_argument("--train_file", required=True, help="JSONL com prompt/response.")
    parser.add_argument("--output_dir", required=True, help="Diretório de saída.")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        fp16=False,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
