# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, help="Directory containing .npy files to write dummy split for")
parser.add_argument("--pattern", type=str, help="Pattern in files to include in test set")
parser.add_argument("--output_file", type=str, help="(Optional) path to output file")


def main(filelist_file, output_file, pattern=None):
    with open(filelist_file, "r") as f:
        lines = f.readlines()
    files = [x.strip() for x in lines]
    
    if pattern is None:
        test_set = list(range(len(files)))
    else:
        test_set = [i for i, name in enumerate(files) if pattern in name]
    
    data = {"train":[], "validation":[], "test":test_set}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    
    args = parser.parse_args()
    p = Path(args.data_dir)
    output_file = args.output_file if args.output_file is not None else p / "split.json"
    
    filelist_file = p / "filelist.txt"
    main(filelist_file, output_file, pattern=args.pattern)