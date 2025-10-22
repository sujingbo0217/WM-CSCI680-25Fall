import os
import random
import numpy as np
from transformers import (
    T5TokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_dataset

# === Paths ===
DATA_PATH = "data/pretrain/data.csv"
TOKENIZER_DIR = "data/tokenizers"
MODEL_DIR = "data/models/t5-large"

# === 1. Load trained tokenizer ===
print(f"Loading tokenizer from: {TOKENIZER_DIR}")
tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_DIR)

# Add mask and sentinel tokens for T5 span corruption
special_tokens = {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
tokenizer.add_special_tokens(special_tokens)

# === 2. Initialize model config ===
print("Initializing CodeT5-style model config")
config = T5Config(
    vocab_size=len(tokenizer),
    d_model=1024,    # 768
    d_ff=4096,       # 3072
    num_layers=24,   # 12
    num_heads=16,    # 12
    decoder_start_token_id=tokenizer.pad_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))

# === 3. Load and preprocess dataset ===
print(f"Loading dataset from: {DATA_PATH}")
dataset = load_dataset("csv", data_files=DATA_PATH)["train"]
dataset = dataset.select(range(150_000))  # first 150k samples
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# === 4. Define T5-style span corruption ===
def mask_random_tokens(example, mask_ratio=0.15):
    """Randomly mask 15% of tokens in the input text for self-supervised pretraining."""
    text = example.get("source", "")
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    tokens = text.split()
    n_tokens = len(tokens)
    if n_tokens < 3:
        return None

    # Determine how many tokens to mask (at least 1)
    num_to_mask = max(1, int(n_tokens * mask_ratio))
    mask_indices = random.sample(range(n_tokens), num_to_mask)

    # Create masked input
    masked_tokens = tokens.copy()
    for i in mask_indices:
        masked_tokens[i] = "<mask>"

    # Prepare target text as the masked tokens joined by spaces
    target_tokens = [tokens[i] for i in mask_indices]

    example["input_text"] = " ".join(masked_tokens)
    example["target_text"] = " ".join(target_tokens)
    return example

print("Masking 15% of tokens for self-supervised pretraining...")
dataset = dataset.map(mask_random_tokens, remove_columns=dataset["train"].column_names)
dataset = dataset.filter(lambda x: x is not None and "input_text" in x)

# === 5. Tokenization ===
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        padding=False,
        truncation=True,
        max_length=512,
    )
    labels = tokenizer(
        examples["target_text"],
        padding=False,
        truncation=True,
        max_length=256,
    ).input_ids
    inputs["labels"] = labels
    return inputs

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === 6. Training setup ===
os.makedirs(MODEL_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    save_total_limit=2,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    report_to="none",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

early_stop = EarlyStoppingCallback(
    early_stopping_patience=3  # stop if no improvement for 3 evals
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stop],
)

# === 7. Train ===
print("Starting CodeT5 pre-training...")
trainer.train()

# === 8. Save results ===
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"CodeT5 pre-training complete! Model saved to: {MODEL_DIR}")
