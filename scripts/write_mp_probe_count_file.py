# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filelist", type=str, help="path to filelist.txt for the dataset")
parser.add_argument("--workers", type=int, default=1, help="Number of workers to use for calculations")

def count_elements_in_numpy_file(file_path):
    # Load the numpy file into a numpy array
    arr = np.load( file_path)
    # Count the number of elements in the array
    shape = arr.shape
    count = np.prod(shape[:3])  # exclude spin, if present

    file_stem = file_path.stem

    # Return the file path and count as a tuple
    return (file_stem, count, shape[0], shape[1], shape[2])


def count_elements_in_numpy_files(file_list_path, workers=10):

    # Read in the list of numpy files from the text file
    with open(file_list_path, 'r') as f:
        file_list = f.read().splitlines()

    file_parent = Path(file_list_path).parent

    file_list = [file_parent / f"{fil}.npy" for fil in file_list]
    
    # Create a pool of worker processes
    with Pool(workers) as p:
        # Map the file paths to the count_elements_in_numpy_file function across the worker processes
        results = list(tqdm(p.imap(count_elements_in_numpy_file, file_list), total=len(file_list)))


    # Create a pandas dataframe from the list of results
    df = pd.DataFrame(results, columns=['id', 'Count', "shape_x", "shape_y", "shape_z"])
    df.to_csv(Path(file_list_path).parent / 'probe_counts.csv', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Define the path to the text file containing the list of numpy files
    count_elements_in_numpy_files(args.filelist, workers=args.workers)
