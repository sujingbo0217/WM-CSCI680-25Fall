import os
import sys
import csv

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents

DATA_PATH = "data/pretrain/data.csv"
TOKENIZER_DIR = "data/tokenizer"
VOCAB_SIZE = 32000

os.makedirs(TOKENIZER_DIR, exist_ok=True)
csv.field_size_limit(sys.maxsize)

# === Collect all code ===
texts = []
with open(DATA_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        code = row["source"].strip()
        if code:
            texts.append(code)

# === Initialize tokenizer ===
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence(
    [NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# === Trainer ===
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
)

# === Train ===
print(f"Training tokenizer on {len(texts):,} samples ...")
tokenizer.train_from_iterator(texts, trainer=trainer)

# === Post-processing for seq2seq format ===
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# === Save ===
tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
print(f"Saved tokenizer to {TOKENIZER_DIR}/tokenizer.json")
