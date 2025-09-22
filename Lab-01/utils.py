import csv

def data_loader(path):
    print(f"Loading sequences from: {path}...")
    seqs = []
    if not path.exists():
        print(f"Error: Dataset file not found at {path}")
        return seqs

    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'code_tokens' not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'code_tokens' column")
        for row in reader:
            raw = row['code_tokens']
            if not raw:
                continue
            
            # Updated parsing logic for comma-separated tokens in a single string.
            # This handles formats like "package, org, ., gradle, ;"
            # The split delimiter is now simply a comma.
            toks = [token.strip() for token in raw.split(',') if token.strip()]
            
            if toks:
                seqs.append(toks)

    print(f"Loaded {len(seqs)} sequences.")
    return seqs
