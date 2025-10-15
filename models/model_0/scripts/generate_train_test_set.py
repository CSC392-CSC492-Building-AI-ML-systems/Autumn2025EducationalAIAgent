#!/usr/bin/env python3
import random
from pathlib import Path
import argparse
from huggingface_hub import HfApi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="HF username for the repository")
    ap.add_argument("--dir", required=True, help="Input folder for per-file .jsonl")
    ap.add_argument("--random-seed", default=42, help="Random seed to generate split")
    ap.add_argument("--n", default=10, help="Number of testcases to include")
    ap.add_argument("--r", default=0.3, help="The ratio of items for test set")
    ap.add_argument("--dname", default="educational-ai-agent-small")
    args = ap.parse_args()

    RANDOM_SEED = args.random_seed

    dataset_name = args.dname
    user=args.user
    REPO_ID = f"{user}/{dataset_name}"
    n = int(args.n)
    ratio = float(args.r)
    input_dir = args.dir

    files = list(Path(input_dir).glob("*"))
    files = [f for f in files if f.is_file()]

    random.seed(RANDOM_SEED)
    selected = random.sample(files, min(n, len(files)))

    cutoff_index = int((1 - ratio) * len(selected))
    train_selected = selected[:cutoff_index] # 70% of files is training set
    test_selected = selected[cutoff_index:] # 30% of files is test set

    api = HfApi()
    for f in train_selected:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo='train/'+f.name,
            repo_id=REPO_ID,
            repo_type="dataset"
        )

    for f in test_selected:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo='test/'+f.name,
            repo_id=REPO_ID,
            repo_type="dataset"
        )

if __name__ == "__main__":
    main()