import os
import ast
import csv
from pathlib import Path
from tqdm import tqdm
import argparse


def get_source_segment(filename, node):
    """Safely extract source code segment for a given AST node."""
    with open(filename, "r", encoding="utf-8") as f:
        src = f.read()
    return ast.get_source_segment(src, node)


def extract_functions_from_file(pyfile, repo_meta):
    """Extract all functions (with if-statements info) from a single Python file."""
    res = []
    try:
        with open(pyfile, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        tree = ast.parse(src)
    except Exception:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                src_seg = ast.get_source_segment(src, node)
            except Exception:
                continue

            # find if-statements inside the function
            if_nodes = [n for n in ast.walk(node) if isinstance(n, ast.If)]
            if args.split_type == "finetune" and len(if_nodes) <= 0:
                continue

            if_conditions = []
            for ifn in if_nodes:
                try:
                    cond = ast.get_source_segment(src, ifn.test)
                    if cond:
                        if_conditions.append(cond.strip())
                except Exception:
                    continue

            res.append({
                "repo": repo_meta.get("name"),
                "file": pyfile,
                "function_name": node.name,
                "num_lines": src_seg.count("\n") + 1,
                "num_if": len(if_nodes),
                "ifs": "; ".join(if_conditions),
                "source": src_seg.strip(),
            })
    return res


def extract_repo(repo_root, repo_meta):
    """Extract functions from all Python files in a repository."""
    pyfiles = [str(p) for p in Path(repo_root).rglob("*.py")]
    out = []
    for py in pyfiles:
        out.extend(extract_functions_from_file(py, repo_meta))
    return out


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.file_name)

    fieldnames = [
        "repo", "file", "function_name", "num_lines", "num_if", "ifs", "source"
    ]

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo_dir in tqdm(sorted(os.listdir(args.repos_dir))):
            repo_path = os.path.join(args.repos_dir, repo_dir)
            if not os.path.isdir(repo_path):
                continue

            repo_meta = {"name": repo_dir}
            try:
                functions = extract_repo(repo_path, repo_meta)
            except Exception as e:
                print(f"[ERROR] Skipping {repo_dir}: {e}")
                continue

            for fn in functions:
                if fn["num_lines"] < args.min_lines:
                    continue
                writer.writerow(fn)

    print(f"Extraction complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Python functions from local repos")
    parser.add_argument("--repos_dir", type=str, default="repos", help="Directory containing all cloned repos")
    parser.add_argument("--output_dir", type=str, default="data/functions", help="Where to save the extracted CSV file")
    parser.add_argument("--min_lines", type=int, default=3, help="Minimum number of lines in a function to include")
    parser.add_argument("--split_type", type=str, default="pretrain", help="data split type")
    parser.add_argument("--file_name", type=str, default="data.csv", help="data file name")
    args = parser.parse_args()
    main(args)
