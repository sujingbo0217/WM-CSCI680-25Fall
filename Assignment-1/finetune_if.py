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
from utils import extract_if_conditions

DATA_PATH = "data/finetune/data.csv"
# DATA_PATH = "data/finetune/benchmark_if_only.csv"
TOKEN_DIR = "results/tokenizers"
MODEL_DIR = "results/models/t5-base"
OUTPUT_DIR = "results"
# CHECKPOINT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Load pretrained tokenizer and model ===
print("Loading pretrained model and tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained(TOKEN_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# tokenizer = T5TokenizerFast.from_pretrained(CHECKPOINT_DIR)
# model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR)

# === 2. Load fine-tuning dataset ===
print("Loading fine-tuning dataset:", DATA_PATH)
dataset = load_dataset("csv", data_files=DATA_PATH)["train"]

columns = dataset.column_names
print("Detected columns:", columns)

# === 3. Mask one if-condition per function depending on format ===
def mask_if_condition(example):
    """
    main test dataset (repo, file, function_name, ifs, source)
    provided dataset (id, code, docstring, etc.)
    """

    code = example.get("source") or example.get("code")
    if not isinstance(code, str) or "if" not in code:
        return None

    if_conditions = extract_if_conditions(code)
    if not if_conditions:
        return None

    # randomly pick one condition to mask
    condition = random.choice(if_conditions)

    # Replace only the first occurrence of the condition
    masked_code = code.replace(condition, "<mask>", 1)

    example["input_text"] = masked_code
    example["target_text"] = condition
    return example


print("Masking one if condition per function...")
dataset = dataset.map(mask_if_condition)
dataset = dataset.filter(lambda x: x is not None and "input_text" in x)

print(f"After masking: {len(dataset)} examples remain.")

# === 4. Split dataset: train / validation / test ===
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_split = dataset["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = dataset["train"]        # 80% fine-tune set
val_dataset = test_split["train"]       # 10% validation set
test_dataset = test_split["test"]       # 10% test set

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

old_test_dataset = test_dataset
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
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
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
    model.to("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Average prediction score: {df['prediction_score'].mean():.2f}")
    print(f"Accuracy: {(df['correct'].sum()/len(df))*100:.2f}%")
    return df


generate_predictions(old_test_dataset, "generated-testset")
print("All predictions done!")
