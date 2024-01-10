# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import argparse
from pathlib import Path
from src.utils.data import decompress_file, read_vasp
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--chgcar_file", type=str, help="CHGCAR file to convert")
parser.add_argument("--output_file", type=str, help="Output file name (no extension needed)")
parser.add_argument("--read_spin", action="store_true", help="Output files including spin density")


def convert(chgcar_file, output_density_file, output_atoms_file, filelist_file, overwrite=False, spin=False):
    # load data
    dec = decompress_file(str(chgcar_file))
    density, atoms, origin = read_vasp(dec, read_spin=spin)
    
    # dump data to pickle with np.save
    if not output_density_file.exists() or overwrite:
        np.save(str(output_density_file), density)
        with open(output_atoms_file, "wb") as f:
                pickle.dump(atoms, f)
    
    # write filename to filelist.txt
    filename = output_density_file.stem
    if filelist_file is not None:
        with open(filelist_file, "a") as myfile:
            myfile.write(f"{filename}\n")
        
                

if __name__ == "__main__":
    args = parser.parse_args()
    
    # read args
    input_file = Path(args.chgcar_file)
    output_file = Path(args.output_file)
    
    # output file ops
    output_file.parent.mkdir(exist_ok=True, parents=True)
    numpy_file = Path(str(output_file) + ".npy")  # with_suffix fails for some filenames
    atoms_file = Path(str(output_file) + "_atoms.pkl")
    filelist_file = output_file.parent / "filelist.txt"
    
    # run chgcar to input file conversion
    convert(input_file, numpy_file, atoms_file, filelist_file, overwrite=True, spin=args.read_spin)