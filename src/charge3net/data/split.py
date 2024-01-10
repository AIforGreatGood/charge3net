# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import json
import math
from torch.utils.data import Subset
import numpy as np


def split_data(dataset, val_frac=0.005, split_file=None):
    # Load or generate splits
    if split_file is not None:
        with open(split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * val_frac))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = Subset(dataset, indices)
    return datasplits
