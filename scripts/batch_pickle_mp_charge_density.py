#!/bin/python
# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).

import sys
import argparse
from pathlib import Path
from src.utils.data import decompress_file, read_vasp, load_numpy_density
from tqdm import tqdm
import numpy as np
import pickle

def main(mp_chg_dir, mp_pkl_dir, my_task_id, num_tasks, read_spin):
    mp_pkl_dir.mkdir(exist_ok=True)

    err_file = mp_pkl_dir / "errfile.txt"

    # get all mp materials
    with open(mp_chg_dir / "filelist.txt", "r") as f:
        mp_ids = [s.strip() for s in f.readlines()]
        
    mp_all = [mp_chg_dir / mpid for mpid in mp_ids]

    # get this task indices
    my_fnames = mp_all[my_task_id:len(mp_all):num_tasks]

    # iterate through my fnames
    for fname in tqdm(my_fnames, desc="Job files"):
        mpid = fname.parts[-1]
        chgcar_file = fname / "CHGCAR"
        
        if not chgcar_file.exists():
            print(f"Recorded an error with file {fname.stem}")
            with open(err_file, "a+") as f:
                f.write(f"{fname.stem}\n")
            continue
        
        # load data
        dec = decompress_file(str(chgcar_file))
        density, atoms, origin = read_vasp(dec, read_spin=read_spin)

        # dump data to pickle with np.save
        density_file = (mp_pkl_dir / mpid).with_suffix(".npy")
        atoms_file = (mp_pkl_dir / f'{mpid}_atoms').with_suffix(".pkl")
        if not density_file.exists():
            np.save(str(density_file), density)
            with open(atoms_file, "wb") as f:
                pickle.dump(atoms, f)

        # check for errors
        density2, atoms2 = load_numpy_density(root=mp_pkl_dir, mpid=mpid)
        if not (np.all(density2 == density) and atoms2 == atoms):
            # error!
            print(f"Recorded an error with file {fname.stem}")
            with open(err_file, "a+") as f:
                f.write(f"{fname.stem}\n")



if __name__ == "__main__":
    # Grab the arguments that are passed in
    parser = argparse.ArgumentParser(
        prog="ChargeConverter",
        description="From a directory with mp structures as subdirectories, outputs an atoms object representing the structure and a (nx, ny, nz, 2) npy array representing the gridded charge density and spin density"
    )
    parser.add_argument("raw_data_dir", type=str, help="path to directory of mpid subdirs, each containing a CHGCAR")
    parser.add_argument("pkl_data_dir", type=str, help="path to new directory to contain all pickled data")
    parser.add_argument('my_task_id', type=int, default=0)
    parser.add_argument('num_tasks', type=int, default=1)
    parser.add_argument('-s', '--spin', type=bool, default=False)

    args = parser.parse_args()

    main(args.raw_data_dir, args.pkl_data_dir, args.my_task_id, args.num_tasks, args.spin)