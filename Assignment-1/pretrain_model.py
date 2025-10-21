# pretrain_codet5.py
import os
import sys
import random
import math
import csv
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ====== Config / hyperparams ======
TOKENIZER_DIR = "data/tokenizer"
PRETRAIN_CSV = "data/pretrain/data.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-4
SPANS_MASKED_RATIO = 0.15   # fraction of tokens masked overall
MEAN_SPAN_LENGTH = 3.0      # geometric mean span length
MAX_SENTINELS = 100         # number of extra_id tokens available (extra_id_0 ... extra_id_99)

csv.field_size_limit(sys.maxsize)

# ====== Load tokenizer saved earlier ======
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
tokenizer.model_max_length = MAX_INPUT_LENGTH

# Add T5-style sentinel tokens if not present: <extra_id_0> ... <extra_id_{MAX_SENTINELS-1}>
extra_tokens = [f"<extra_id_{i}>" for i in range(MAX_SENTINELS)]
new_specials = []
for t in extra_tokens:
    if t not in tokenizer.get_vocab():
        new_specials.append(t)
if new_specials:
    tokenizer.add_tokens(new_specials)
    print(f"Added {len(new_specials)} T5 sentinel tokens to tokenizer")

# Update special tokens (pad/bos/eos are already defined from tokenizer training step)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<pad>"
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"
if tokenizer.bos_token is None:
    tokenizer.bos_token = "<s>"

# ====== Span masking utilities ======
def sample_spans(token_len: int, mask_ratio: float = 0.15, mean_span_len: float = 3.0):
    """
    Return a list of (start, end) token index pairs to mask.
    We follow T5: sample spans with geometric distribution over lengths until mask_ratio reached.
    """
    n_to_mask = max(1, int(round(token_len * mask_ratio)))
    spans = []
    masked = 0
    attempts = 0
    while masked < n_to_mask and attempts < n_to_mask * 3:
        attempts += 1
        # sample span length using geometric distribution -> P(L) ~ (1-p)^(L-1) p
        # mean = 1/p => p = 1/mean
        p = 1.0 / mean_span_len
        # sample L
        L = max(1, int(random.geometricvariate(p) if hasattr(random, "geometricvariate") else \
                       max(1, int(random.expovariate(1.0/mean_span_len)))))  # fallback to expovariate
        start = random.randint(0, max(0, token_len - 1))
        end = min(token_len, start + L)
        # avoid overlap
        overlap = any(not (end <= s or start >= e) for s, e in spans)
        if overlap:
            continue
        spans.append((start, end))
        masked += (end - start)
    # sort spans by start
    spans = sorted(spans, key=lambda x: x[0])
    return spans

def geometric_span_length(mean):
    # sample via geometric approximation using expovariate and rounding
    lam = 1.0 / mean
    L = max(1, int(random.expovariate(lam)))
    return L

def sample_spans_simple(token_len: int, mask_ratio: float = 0.15, mean_span_len: float = 3.0):
    n_to_mask = max(1, int(round(token_len * mask_ratio)))
    spans = []
    masked = 0
    tries = 0
    while masked < n_to_mask and tries < n_to_mask * 5:
        L = geometric_span_length(mean_span_len)
        start = random.randint(0, max(0, token_len - L))
        end = start + L
        # check overlap
        if any(not (end <= s or start >= e) for s, e in spans):
            tries += 1
            continue
        spans.append((start, end))
        masked += (end - start)
    spans.sort()
    return spans

# ====== Dataset ======
class SpanCorruptionDataset(Dataset):
    def __init__(self, csv_path, tokenizer: PreTrainedTokenizerFast,
                 max_input_len=512, max_target_len=256,
                 mask_ratio=0.15, mean_span_len=3.0, max_sentinels=100):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                text = (r.get("source") or r.get("code") or "").strip()
                if text:
                    self.rows.append(text)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.mask_ratio = mask_ratio
        self.mean_span_len = mean_span_len
        self.max_sentinels = max_sentinels

    def __len__(self):
        return len(self.rows)

    def build_example(self, raw_text: str):
        # 1) tokenize to token ids (no special tokens yet)
        enc = self.tokenizer(raw_text, add_special_tokens=False)
        input_ids = enc["input_ids"]
        token_len = len(input_ids)
        if token_len == 0:
            return None

        # 2) sample spans
        spans = sample_spans_simple(token_len, mask_ratio=self.mask_ratio, mean_span_len=self.mean_span_len)
        if not spans:
            # fallback: mask one short token in middle
            mid = token_len // 2
            spans = [(mid, mid+1)]

        # ensure we have enough sentinels
        if len(spans) > self.max_sentinels:
            spans = spans[:self.max_sentinels]

        # 3) build input tokens: replace each span by a sentinel token id (we will insert text sentinel tokens as strings, then re-tokenize)
        # We'll build textual input by decoding kept tokens and inserting sentinel text tokens.
        # Get per-token decoded pieces; safer approach: use tokenizer.convert_ids_to_tokens then join with ' '.
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # Build a text representation token-by-token (space joined)
        # We will join tokens using the tokenizer's token joiner behavior: using tokenizer.convert_tokens_to_string for slices
        input_pieces = []
        target_pieces = []
        last = 0
        sentinel_idx = 0
        for (s, e) in spans:
            # slice tokens from last->s (kept region)
            if s > last:
                kept_tokens = tokens[last:s]
                kept_text = self.tokenizer.convert_tokens_to_string(kept_tokens)
                input_pieces.append(kept_text)
            # sentinel token text <extra_id_{i}>
            sentinel_text = f"<extra_id_{sentinel_idx}>"
            input_pieces.append(sentinel_text)
            # target should contain sentinel + original span text
            span_tokens = tokens[s:e]
            span_text = self.tokenizer.convert_tokens_to_string(span_tokens)
            target_pieces.append(sentinel_text + " " + span_text)
            sentinel_idx += 1
            last = e
        # append suffix
        if last < len(tokens):
            kept_tokens = tokens[last:]
            kept_text = self.tokenizer.convert_tokens_to_string(kept_tokens)
            input_pieces.append(kept_text)

        # join input pieces (note: may produce double spaces; ok)
        corrupted_input = " ".join([p for p in input_pieces if p is not None and p != ""])
        target = " ".join([p for p in target_pieces]) + f" <extra_id_{sentinel_idx}>"  # final sentinel as end marker per T5 (optional)

        # 4) add special tokens and encode
        input_enc = self.tokenizer(corrupted_input, truncation=True, padding="max_length",
                                   max_length=self.max_input_len, return_tensors="pt")
        target_enc = self.tokenizer(target, truncation=True, padding="max_length",
                                    max_length=self.max_target_len, return_tensors="pt")

        # labels: replace pad token id with -100 so loss ignored on padding
        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def __getitem__(self, idx):
        raw = self.rows[idx]
        example = self.build_example(raw)
        # some examples might be None if weird; handle by sampling another
        if example is None:
            # naive fallback
            return self.__getitem__((idx + 1) % len(self))
        return example

# ====== Instantiate dataset ======
dataset = SpanCorruptionDataset(PRETRAIN_CSV, tokenizer, max_input_len=MAX_INPUT_LENGTH,
                                max_target_len=MAX_TARGET_LENGTH,
                                mask_ratio=SPANS_MASKED_RATIO, mean_span_len=MEAN_SPAN_LENGTH,
                                max_sentinels=MAX_SENTINELS)

print(f"Dataset size: {len(dataset)} examples")

# ====== Model setup (T5 from scratch) ======
config = T5Config(
    vocab_size=len(tokenizer),
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    dropout_rate=0.1,
    initializer_factor=1.0,
    feed_forward_proj="relu",
)

model = T5ForConditionalGeneration(config)
# resize token embeddings if tokenizer added new tokens
model.resize_token_embeddings(len(tokenizer))

# ====== Trainer ======
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=200,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    label_smoothing_factor=0.0,
    predict_with_generate=False,   # we don't need generation during pretraining
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(os.path.join(OUT_DIR, "final"))
print(f"Saved model to {os.path.join(OUT_DIR, 'final')}")
