# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import argparse
import numpy as np

from pymatgen.io.vasp import Chgcar
from ase.calculators.vasp import VaspChargeDensity

from src.utils.data import load_atoms_file, load_density_file

parser = argparse.ArgumentParser()
parser.add_argument("--density_file", type=str, help="path to .npy density file")
parser.add_argument("--atoms_file", type=str, help="path to .pkl atoms file")
parser.add_argument("--output_file", type=str, help="path to CHGCAR file to save out")
parser.add_argument("--aug_chgcar_file", type=str, default=None, help="path to original CHGCAR file to retrieve augmentation")


def deepdft_to_chgcar(density_file, atoms_file, aug_chgcar_file=None) -> VaspChargeDensity:
    density = load_density_file(density_file)
    atoms = load_atoms_file(atoms_file)
    

    # retrieve augmentation, if requested
    if aug_chgcar_file is not None:
        aug = Chgcar.from_file(aug_chgcar_file).data_aug
    else:
        aug = None
        
    # extract spin, if available
    if len(density.shape) == 4:  # implies a spin channel exists
        charge_grid = density[..., 0]
        spin_grid = density[..., 1]
    else:
        charge_grid = density
        spin_grid = np.zeros_like(density)
        
    # create Chgcar object
    vcd = VaspChargeDensity(filename=None)
    vcd.atoms.append(atoms)
    vcd.chg.append(charge_grid)
    vcd.chgdiff.append(spin_grid)
    if aug is not None:
        vcd.aug = "".join(aug["total"])
        vcd.augdiff = "".join(aug["diff"])
        
    return vcd

if __name__ == "__main__":
    args = parser.parse_args()
    
    chgcar = deepdft_to_chgcar(args.density_file, args.atoms_file, aug_chgcar_file=args.aug_chgcar_file)
    chgcar.write(args.output_file, format="chgcar")