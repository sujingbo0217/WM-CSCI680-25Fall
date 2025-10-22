import os
import random
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "data/finetune/data.csv"
MODEL_DIR = "data/models/t5-base"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Load pretrained tokenizer and model ===
print("Loading pretrained model and tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# === 2. Load fine-tuning dataset ===
print("Loading fine-tuning dataset:", DATA_PATH)
dataset = load_dataset("csv", data_files=DATA_PATH)["train"]

# Example dataset columns:
#   function_code, if_condition
# Each function_code contains at least one "if" statement.

# === 3. Mask one if-condition per function ===


def mask_if_condition(example):
    code = example["function_code"]
    condition = example["if_condition"]

    if "if" not in code or condition not in code:
        return None

    # Mask the condition once
    masked_code = code.replace(condition, "<mask>", 1)
    example["input_text"] = masked_code
    example["target_text"] = condition
    return example


print("Masking one if condition per function...")
dataset = dataset.map(mask_if_condition)
dataset = dataset.filter(lambda x: x is not None and "input_text" in x)

# === 4. Split dataset: train / validation / test ===
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_split = dataset["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = dataset["train"]
val_dataset = test_split["train"]
test_dataset = test_split["test"]

print(
    f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# === 5. Tokenization ===


def tokenize_batch(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        batch["target_text"],
        max_length=64,
        truncation=True,
        padding="max_length",
    ).input_ids
    model_inputs["labels"] = labels
    return model_inputs


train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

train_dataset.set_format(type="torch", columns=[
                         "input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=[
                       "input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=[
                        "input_ids", "attention_mask", "labels"])

# === 6. Fine-tuning ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("Starting fine-tuning...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete!")

# === 7. Evaluation & Predictions ===


def generate_predictions(dataset, name):
    results = []
    model.eval()

    for example in dataset:
        input_text = example["input_text"]
        expected = example["target_text"]

        inputs = tokenizer(input_text, return_tensors="pt",
                           truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=64, num_beams=4, return_dict_in_generate=True, output_scores=True)
        pred_ids = outputs.sequences[0]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        score = float(torch.exp(outputs.sequences_scores[0]).item(
        ) * 100) if hasattr(outputs, "sequences_scores") else 0.0

        correct = pred_text.strip() == expected.strip()
        results.append({
            "input_provided": input_text,
            "correct": correct,
            "expected_if_condition": expected,
            "predicted_if_condition": pred_text,
            "prediction_score": round(score, 2),
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")
    return df


# Run on both your test sets
generate_predictions(test_dataset, "generated-testset")
# Optionally load a provided external testset and evaluate:
# provided_testset = load_dataset("csv", data_files="data/finetune/provided_test.csv")["train"]
# generate_predictions(provided_testset, "provided-testset")

print("All predictions done!")
