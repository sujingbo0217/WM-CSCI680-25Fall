import csv
from pathlib import Path

def data_loader(path: Path):
    print(f"Loading sequences from: {path}...")
    seqs = []

    if not path.exists():
        print(f"Error: Dataset file not found at {path}")
        return seqs

    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, quotechar='"', escapechar='\\')
        
        if 'code_tokens' not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'code_tokens' column")
        
        for row in reader:
            raw = row.get('code_tokens')
            if not raw:
                continue

            toks = [token.strip() for token in raw.split(',') if token.strip()]
            if toks:
                seqs.append(toks)

    print(f"Loaded {len(seqs)} sequences.")
    return seqs
