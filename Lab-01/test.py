import os
import argparse
import csv
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Manager

from utils import data_loader
from interfaces import NgramConfig
from n_gram import Ngram


def train_and_evaluate_args(args):
    n, train_data, eval_data, k = args
    print(f"\nEvaluating N={n}...")
    model = Ngram(NgramConfig(n=n, k=k))
    model.train(sequences=train_data)
    pp = model.PPL(eval_data)
    print(f"N={n}, Validation Perplexity = {pp:.4f}")
    return (n, pp)


def evaluate(train_data_all, n_list, k, val_ratio=0.1, max_workers=16):
    print(f"Splitting training data for validation ({1 - val_ratio:.0%}/{val_ratio:.0%})...")
    train_data, eval_data = train_test_split(train_data_all, test_size=val_ratio, random_state=42)

    args_list = [(n, train_data, eval_data, k) for n in n_list]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_and_evaluate_args, args_list))

    # Select best n and retrain
    best_n, best_ppl = min(results, key=lambda x: x[1])
    print(f"Best N={best_n} with perplexity {best_ppl:.4f}")
    
    best_model = Ngram(NgramConfig(n=best_n, k=k))
    best_model.train(sequences=train_data_all)

    return best_model, results


def build_contexts(test_data, n, target):
    """
    Builds a list of contexts from test sequences for prediction sampling.
    """
    contexts = []
    for seq in test_data:
        if len(seq) < 2:
            continue
        # Take up to 5 random snippets from each test sequence
        for _ in range(min(5, len(seq) - 1)):
            if len(contexts) >= target:
                return contexts
            div = random.randint(1, len(seq) - 1)
            contexts.append(seq[max(0, div - (n - 1)) : div])
    return contexts[:target]


def write_results(model, test_data, num_samples, topk, path):
    """Generates and writes model predictions to a JSONL file."""
    n = model.cfg.n
    contexts = build_contexts(test_data, n, num_samples)
    
    with path.open('w', encoding='utf-8') as f:
        for ctx in contexts:
            top_preds = model.topk(ctx, topk=topk)
            completion = model.generate(ctx)
            rec = {
                'context': ' '.join(ctx),
                'topk_predictions': {token: f"{prob:.6f}" for token, prob in top_preds},
                'sampled_completion': ' '.join(completion),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Wrote {len(contexts)} predictions to {path}")


def main():
    parser = argparse.ArgumentParser(description='N-gram model for Java Code Generation')
    parser.add_argument('--train_csv', type=str, default='../data/java_train_data.csv', help="Path to the CSV training data.")
    parser.add_argument('--test_csv', type=str, default='../data/java_eval_data.csv', help="Path to the CSV test data.")
    parser.add_argument('--n_list', type=int, nargs='+', default=[3, 5, 7, 11], help="List of N values to test.")
    parser.add_argument('--k', type=float, default=0.1, help="Value for Add-k smoothing.")
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Ratio of training data to use for validation.")
    parser.add_argument('--topk', type=int, default=10, help="Number of top predictions to save.")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples to generate from the test set.")
    parser.add_argument('--out_dir', type=str, default='output', help="Directory to save outputs.")
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(), help="Maximum number of worker threads.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_data = data_loader(Path(args.train_csv))
    test_data = data_loader(Path(args.test_csv))
    
    if not train_data or not test_data:
        print("Exiting: Training or test data could not be loaded.")
        return

    print(f"Total sequences loaded: TRAIN={len(train_data)}, TEST={len(test_data)}")

    best, results = evaluate(train_data, args.n_list, args.k, args.val_ratio, args.max_workers)
    
    best_n_val_pp = dict(results).get(best.cfg.n, float('inf'))
    print(f"Selected N={best.cfg.n} with validation perplexity: {best_n_val_pp:.4f}")

    # Evaluate the final selected model on the held-out test set
    test_pp = best.PPL(test_data)
    print(f"Test Perplexity of the best model (N={best.cfg.n}): {test_pp:.4f}")

    # Write outputs
    res_path = out / 'results.csv'
    with res_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'validation_perplexity'])
        for n, pp in results:
            w.writerow([n, f'{pp:.6f}'])
        w.writerow([]) # Spacer
        w.writerow(['best_model_test_perplexity', f'{test_pp:.6f}'])
    print(f"Metric results saved to {res_path}")

    # Save best model config
    best_model_path = out / 'best.json'
    best_model_path.write_text(
        json.dumps({'N': best.cfg.n, 'k': best.cfg.k}, indent=2), encoding='utf-8')
    print(f"Best model configuration saved to {best_model_path}")

    # Generate and save predictions
    predictions_path = out / 'pred.jsonl'
    write_results(best, test_data, args.num_samples, args.topk, predictions_path)


if __name__ == '__main__':
    main()

