import os
import sys
import csv

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

INPUT_CSV = "data/pretrain/data.csv"
OUT_DIR = "data/tokenizers"
VOCAB_SIZE = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

os.makedirs(OUT_DIR, exist_ok=True)
csv.field_size_limit(sys.maxsize)

# 1) Create a simple iterator over code strings
def code_iterator(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("source") or row.get("code") or row.get("input") or ""
            if text:
                # keep raw code (preserve newlines)
                yield text

# 2) Initialize a BPE tokenizer with byte-level pretokenizer (robust for code)
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel()  # splits bytes into tokens similar to byte-level BPE

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS
)

print("Training tokenizer...")
tokenizer.train_from_iterator(code_iterator(INPUT_CSV), trainer=trainer)

# Add post-processor so we have explicit <s> ... </s> framing
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# Save tokenizer json
tok_json = os.path.join(OUT_DIR, "tokenizer.json")
tokenizer.save(tok_json)
print(f"Saved tokenizer to {tok_json}")

# Also write a tiny config so PreTrainedTokenizerFast can be loaded easily
from transformers import PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json,
                                       bos_token="<s>",
                                       eos_token="</s>",
                                       unk_token="<unk>",
                                       pad_token="<pad>",
                                       mask_token="<mask>")
hf_tokenizer.save_pretrained(OUT_DIR)
print(f"HuggingFace tokenizer saved to {OUT_DIR} (tokenizer.json + tokenizer_config.json)")
