from glob import glob
import jsonlines
import json
import os
import argparse
from os import path as osp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name string input')
    args = parser.parse_args()

    files = glob(f'{args.name}/**/*.jsonl', recursive=True)

    target_path = f"{'/'.join(args.name.split('/')[-3:])}.jsonl"
    
    if osp.exists(target_path): 
        raise ValueError
    
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    print(f"Merged File: {target_path}")
    merged_f = open(target_path, 'w')
    for file in files:
        with jsonlines.open(file) as f:
            for item in f.iter():
                merged_f.write(json.dumps(item) + '\n')
                merged_f.flush()
    merged_f.close()