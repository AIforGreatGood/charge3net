# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).

import lz4
import gzip, zlib
import tempfile
import os
import io
import pickle

from pathlib import Path
import numpy as np

import ase
from ase.calculators.vasp import VaspChargeDensity
import lz4.frame
from scipy.spatial import KDTree


def approximate_gradient(density, cell):
    # Note, units for x, y, z are in terms of unitcells, not Angstrom
    x = np.arange(density.shape[0]) / density.shape[0]
    y = np.arange(density.shape[1]) / density.shape[1]
    z = np.arange(density.shape[2]) / density.shape[2]

    # gradient wrt movment in cell
    # TODO: if pbd, wrap edges so central differences can be used on edge
    drho_dxyz = np.stack(np.gradient(density, x, y, z), axis=-1)

    # gradient wrt movement in cartesian
    drho_dxyz = np.dot(drho_dxyz, np.linalg.inv(cell.T))
    return drho_dxyz

def compute_zeta(density, atoms, dist_cutoff=np.inf):
    '''Compute zeta, or one minus the weighted average of cosine similarities between charge density gradient
    and unit vector to nearest atom, weighted by gradient norm. 
    
    This is computed as a measure of angular variance for the charge density field. Fields that are dependent
    solely on distance from the nearest atom should have zeta close to 0, whereas fields
    that have large amounts of angular dependence would have gradients in the charge density field that are
    orthogonal to the vector pointing to nearest atom, and zeta closer to 1. In spherical coordinates with an origin at the nearest 
    atom, these gradients would be pointing in the theta and phi directions.
    
    Inputs:
        density: the charge density field
        atoms: ase atoms object
        dist cutoff: a maximum distance (in angstroms) of probes from nearest atoms to consider for zeta calculation
        
    Outputs:
        zeta: measure of angular variation in the charge density field
    '''
    
    grad = approximate_gradient(density, atoms.get_cell()).reshape(-1,3)
    grad_norm = np.linalg.norm(grad, axis=-1).reshape(-1)
    grid_pos = calculate_grid_pos(density, [0.,0.,0.], atoms.get_cell()).reshape(-1,3)
    density = density.reshape(-1)
    
    atom_to_probe, dist, _ = voxel_nearest_atom(grid_pos, atoms)
    
    # apply distance cutoff
    dist_meets = np.bitwise_and(np.bitwise_not(dist == 0), dist < dist_cutoff)
    atom_to_probe = atom_to_probe[dist_meets]
    dist = dist[dist_meets]
    grad = grad[dist_meets]
    grad_norm = grad_norm[dist_meets] + 1e-10  # prevent ZeroDivisionError
    density = density[dist_meets]
    
    # compute dot product and cos similarity
    unit_probe_to_atom = -atom_to_probe / dist.reshape(-1, 1)
    dot_product = np.sum(unit_probe_to_atom * grad, axis=1)
    cosine_similarity = dot_product / grad_norm
    abs_cos_sim = np.abs(cosine_similarity)
    
    zeta = 1 - np.average(abs_cos_sim, weights=np.abs(grad_norm))
    return zeta

def calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""

    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent

def decompress_file(filepath):
    if filepath.endswith(".zz"):
        with open(filepath, "rb") as fp:
            f_bytes = fp.read()
        filecontent = zlib.decompress(f_bytes)
    elif filepath.endswith(".lz4"):
        with lz4.frame.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    elif filepath.endswith(".gz"):
        with gzip.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    elif filepath.endswith(".pkl"):
        filecontent = np.load(filepath, allow_pickle=True)
    else:
        with open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    return filecontent

def read_vasp(filecontent, read_spin=False):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpcharge3net")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    try:
        density = vasp_charge.chg[-1]  # separate density
        if read_spin:
            if len(vasp_charge.chgdiff) != 0:
                spin_density = vasp_charge.chgdiff[-1]
            else:
                # assume non-spin-polarized if there's no spin density data
                spin_density = np.zeros_like(density)
            density = np.stack([density, spin_density], axis=-1)
    except IndexError as e:
        print(e, f"\nFileconents of {filecontent} do not contain chg field")
    atoms = vasp_charge.atoms[-1]  # separate atom positions

    return density, atoms, np.zeros(3)  # TODO: Can we always assume origin at 0,0,0?


def read_cube(filecontent):
    textbuf = io.StringIO(filecontent.decode())
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr ** 3
    return cube["data"], cube["atoms"], origin


def load_numpy_density(root, mpid):
    '''
    Load pickled density and atoms files from "root" directory

    density file is assumed to have format f'{root}/{mpid}.npy'
    atoms file is assumed to have format f'{root}/{mpid}_atoms.pkl'
    '''
    root = Path(root)
    mpid = Path(mpid)
    density_file = str(root / f"{mpid}.npy")
    density = load_density_file(density_file)
    atoms = _load_pickled_atoms(root, mpid)
    return density, atoms

def load_density_file(filename: str):
    return np.load(filename)

def _load_pickled_atoms(root, mpid):
    root= Path(root)
    mpid = Path(mpid)
    atoms_file = str(root / f'{mpid}_atoms.pkl')
    return load_atoms_file(atoms_file)

def load_atoms_file(filename: str):
    with open(filename, "rb") as f:
        atoms = pickle.load(f)
    return atoms


def voxel_nearest_atom_dist(grid_pos, atoms, supercell=True):
    """
    For each voxel, compute the distance to the nearest atom. 
    
    Use supercell atoms if specified

    Output:
        atom_min_dist: cartesian distance to nearest atom
        atom_min_dist_num: atomic number of nearest atom
    """
    _, min_dist, min_dist_num = voxel_nearest_atom(grid_pos, atoms, supercell=supercell)
    return min_dist, min_dist_num


def voxel_nearest_atom(grid_pos, atoms, supercell=True):
    
    if supercell:
        atoms_positions, atomic_numbers = supercell_atoms_positions(atoms)
    else:
        atoms_positions, atomic_numbers = unitcell_atoms_positions(atoms)

    # nearest neighbor lookup
    kdtree = KDTree(atoms_positions)
    atom_min_dist, atom_min_idx = kdtree.query(grid_pos, k=1)
    min_atom_to_probe = grid_pos - atoms_positions[atom_min_idx]
    atom_min_dist_num = atomic_numbers[atom_min_idx].squeeze(-1)

    return min_atom_to_probe, atom_min_dist, atom_min_dist_num


def unitcell_atoms_positions(atoms):
    return atoms.positions, atoms.get_atomic_numbers()[...,None]


def supercell_atoms_positions(atoms):
    atoms_positions = atoms.positions
    atomic_numbers = atoms.get_atomic_numbers()

    # repeat directions
    repeats = [-1, 0, 1]  # only need one repeat around unit cell to test for closest atom
    repeat_offsets = np.array([(x, y, z) for x in repeats for y in repeats for z in repeats])
    # total repeats in all dimensions
    total_repeats = repeat_offsets.shape[0]
    # project repeat cell offsets into cartesian space
    repeat_offsets = np.dot(repeat_offsets, atoms.get_cell())
    # tile grid positions, subtract offsets 
    # (subtracting grid positions is like adding atom positions)
    offset_grid_pos = np.repeat(atoms_positions[..., None, :], total_repeats, axis=-2)
    atomic_numbers = np.repeat(atomic_numbers[...,None], total_repeats, axis=-2)
    offset_grid_pos -= repeat_offsets

    atoms_positions = offset_grid_pos.reshape(np.prod(offset_grid_pos.shape[:2]),3)
    return atoms_positions, atomic_numbers