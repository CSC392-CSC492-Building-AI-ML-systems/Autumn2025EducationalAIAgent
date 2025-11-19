#!/usr/bin/env python3
import random
from pathlib import Path
import argparse
from datasets import load_dataset
from datasets import DatasetDict

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

    # cutoff_index = int((1 - ratio) * len(selected))
    # train_selected = selected[:cutoff_index] # 70% of files is training set
    # test_selected = selected[cutoff_index:] # 30% of files is test set

    # test_selected = [Path('../../model0_data_test/1728666941.jsonl')]

    ds = load_dataset('json', data_files=[str(f) for f in selected])

    # data = {
    #     "train": [str(f) for f in train_selected],
    #     "test": [str(f) for f in test_selected]
    # }

    # ds = load_dataset("json", data_files=data)
    # ds.push_to_hub(REPO_ID)
    indices = []
    print(ds)
    print(ds['train'])
    outputs = ds['train']['output']

    for i in range(len(outputs)):
        if i % 10000 == 0:
            print("i={}".format(i))
        answer = outputs[i]
        if answer == 'Answer: NEW':
            indices.append(i)
            # Add a random not included index to catch up
            rand_index = random.randint(0, len(outputs))
            while rand_index in indices or outputs[rand_index] == "Answer: NEW":
                rand_index = random.randint(0, len(outputs))
        
            indices.append(rand_index)
    
    # Shuffle them for train/test
    random.shuffle(indices)
    cutoff_index = int((1 - ratio) * len(indices))
    train_indices = indices[:cutoff_index]
    test_indices = indices[cutoff_index:]

    new_ds = DatasetDict({
        "train": ds['train'].select(train_indices),
        "test": ds['train'].select(test_indices)
    })

    new_ds.push_to_hub(REPO_ID)

if __name__ == "__main__":
    main()