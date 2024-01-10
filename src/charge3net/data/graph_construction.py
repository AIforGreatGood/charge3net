# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import warnings

from scipy.spatial import KDTree

import torch
import numpy as np
import ase
import ase.neighborlist
import logging

class GraphConstructor(object):
    def __init__(self, cutoff, num_probes=None, disable_pbc=False, sorted_edges=False):
        super().__init__()
        self.cutoff = cutoff
        self.disable_pbc = disable_pbc
        self.sorted_edges = sorted_edges
        self.default_type = torch.get_default_dtype()
        self.num_probes = num_probes

    def __call__(self,
        density,
        atoms,
        grid_pos,
        ):

        if self.disable_pbc:
            atoms = atoms.copy()
            atoms.set_pbc(False)

        probe_pos, probe_target = self.sample_probes(grid_pos, density)
        graph_dict = self.atoms_and_probes_to_graph(atoms, probe_pos)
        
        # pylint: disable=E1102
        graph_dict.update(
            probe_target=torch.tensor(probe_target, dtype=self.default_type),
            num_nodes=torch.tensor(graph_dict["nodes"].shape[0]),
            num_atom_edges=torch.tensor(graph_dict["atom_edges"].shape[0]),
            num_probes=torch.tensor(probe_target.shape[0]),
            num_probe_edges=torch.tensor(graph_dict["probe_edges"].shape[0]),
            probe_xyz=torch.tensor(probe_pos, dtype=self.default_type),
            atom_xyz=torch.tensor(atoms.get_positions(), dtype=self.default_type),
            # NOTE (teddy): tensor from list of ndarrays is very slow, see 
            # https://github.com/pytorch/pytorch/issues/13918
            cell=torch.tensor(np.array(atoms.get_cell()), dtype=self.default_type),
        )

        return graph_dict
        
    def sample_probes(self, grid_pos, density):
        if self.num_probes is not None:
            probe_choice_max = np.prod(grid_pos.shape[0:3])
            probe_choice = np.random.randint(probe_choice_max, size=self.num_probes)
            probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
            probe_pos = grid_pos[probe_choice]
            probe_target = density[probe_choice]
        else:
            probe_pos = grid_pos.reshape(-1,3)
            if len(density.shape) == 4: # spin density TODO: have actual arg for spin
                probe_target = density.reshape(-1, 2)
            else:
                probe_target = density.flatten()
        return probe_pos, probe_target

    
    def atoms_and_probes_to_graph(self, atoms, probe_pos):
        atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = self.atoms_to_graph(atoms)
        
        probe_edges, probe_edges_displacement = self.probes_to_graph(atoms, probe_pos, 
            neighborlist=neighborlist, inv_cell_T=inv_cell_T)        

        if self.sorted_edges:
            # Sort probe edges for reproducibility
            concat_pe = _sort_by_rows(np.concatenate((probe_edges, probe_edges_displacement), axis=1))
            probe_edges = concat_pe[:,:2].astype(int)
            probe_edges_displacement = concat_pe[:,2:]

        graph_dict = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
            "atom_edges_displacement": torch.tensor(
                np.concatenate(atom_edges_displacement, axis=0), dtype=self.default_type
            ),
            "probe_edges": torch.tensor(probe_edges),
            "probe_edges_displacement": torch.tensor(
                probe_edges_displacement, dtype=self.default_type
            ),
        }
        return graph_dict

    def atoms_to_graph(self, atoms):
        atom_edges = []
        atom_edges_displacement = []

        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # Compute neighborlist
        if (
            True # force ASE
            or np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
            )
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            # neighborlist = AseNeigborListWrapper(cutoff, atoms)
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        atom_positions = atoms.get_positions()

        for i in range(len(atoms)):
            neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, self.cutoff)

            self_index = np.ones_like(neigh_idx) * i
            edges = np.stack((neigh_idx, self_index), axis=1)

            neigh_pos = atom_positions[neigh_idx]
            this_pos = atom_positions[i]
            neigh_origin = neigh_vec + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            atom_edges.append(edges)
            atom_edges_displacement.append(neigh_origin_scaled)

        return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T

    def probes_to_graph(self,atoms, probe_pos, neighborlist=None, inv_cell_T=None):
        pass

class KdTreeGraphConstructor(GraphConstructor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probes_to_graph(self, atoms, probe_pos, neighborlist=None, inv_cell_T=None):
        # FIXME: can turn this into atoms_and_probes_to_graph. The atoms NNs can be extracted
        # from the KD tree. This will circumvent ASAP/Ase completely
        atom_positions = atoms.positions
        atom_idx = np.arange(len(atoms))

        if inv_cell_T is None:
            inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # get number of repeats in each dimension
        pbc = atoms.get_pbc()
        cell_heights = _cell_heights(atoms.get_cell())
        n_rep = np.ceil(self.cutoff / (cell_heights + 1e-12))
        _rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]
        repeat_offsets = np.array([(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)])

        # total repeats in all dimensions
        total_repeats = repeat_offsets.shape[0]
        # project repeat cell offsets into cartesian space
        repeat_offsets = np.dot(repeat_offsets, atoms.get_cell())
        # tile grid positions, subtract offsets 
        # (subtracting grid positions is like adding atom positions)
        supercell_atom_pos = np.repeat(atom_positions[..., None, :], total_repeats, axis=-2)
        supercell_atom_pos += repeat_offsets
        
        # store the original index of each atom
        supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)

        # flatten
        supercell_atom_positions = supercell_atom_pos.reshape(np.prod(supercell_atom_pos.shape[:2]), 3)
        supercell_atom_idx = supercell_atom_idx.reshape(np.prod(supercell_atom_pos.shape[:2]))

        # create KDTrees for atoms and probes
        atom_kdtree = KDTree(supercell_atom_positions)
        probe_kdtree = KDTree(probe_pos)

        # query points between kd tree
        query = probe_kdtree.query_ball_tree(atom_kdtree, r=self.cutoff)

        # set up vector of destination nodes (probes)
        edges_per_probe = [len(q) for q in query]
        dest_node_idx = np.concatenate([[i]*n for i,n in enumerate(edges_per_probe)]).astype(int)

        # get original atom idx from supercell idx
        supercell_neigh_idx = np.concatenate(query).astype(int)
        src_node_idx = supercell_atom_idx[supercell_neigh_idx]
        # create edges from src/dest nodes
        probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)

        # get non-supercell atom positions
        src_pos = atom_positions[src_node_idx]
        dest_pos = probe_pos[dest_node_idx]

        # FIXME: on the next two lines, what is the purpose of dest_pos? 
        # edge vector between supercell atoms and probe
        neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
        # compute displacement (number of unitcells in each dim)
        neigh_origin = neigh_vecs + dest_pos - src_pos
        probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)

        return probe_edges, probe_edges_displacement


class AsapAseGraphConstructor(GraphConstructor):
    def __init__(self, *args, probe_batch_size=50, **kwargs):
        self.probe_batch_size = probe_batch_size
        super().__init__(*args, **kwargs)

    def probes_to_graph(self, atoms, probe_pos, neighborlist=None, inv_cell_T=None):
        try:
            import asap3
            asap3_available = True
        except ImportError:
            warnings.warn("Could not import asap3, falling back to ase. See README for installation instructions")
            asap3_available = False

        probe_edges = []
        probe_edges_displacement = []
        if inv_cell_T is None:
            inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        if hasattr(neighborlist, "get_neighbors_querypoint"):
            results = neighborlist.get_neighbors_querypoint(probe_pos, self.cutoff)
            atomic_numbers = atoms.get_atomic_numbers()
        else:
            # Insert probe atoms
            #  use_ase = True
            use_ase = (
                    np.any(atoms.get_cell().lengths() <= 0.0001)
                    or (
                        np.any(atoms.get_pbc())
                        and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
                    )
                    or not asap3_available
                )

            num_probes = probe_pos.shape[0]
            split_idx = np.ceil(num_probes/self.probe_batch_size) if self.probe_batch_size > 0 else 1
            probe_positions = np.split(probe_pos, split_idx)
            results = []
            for probe_position in probe_positions:
                n_batch_probes = probe_position.shape[0]
                probe_atoms = ase.Atoms(numbers=[0] * n_batch_probes, positions=probe_position)
                atoms_with_probes = atoms.copy()
                atoms_with_probes.extend(probe_atoms)
                atomic_numbers = atoms_with_probes.get_atomic_numbers()
                if use_ase:
                    neighborlist = AseNeigborListWrapper(self.cutoff, atoms_with_probes)
                else:
                    # neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
                    neighborlist = asap3.FullNeighborList(self.cutoff, atoms_with_probes)

                results.extend( [neighborlist.get_neighbors(i+len(atoms), self.cutoff) for i in range(n_batch_probes)])

        atom_positions = atoms.get_positions()
        for i, (neigh_idx, neigh_vec, _) in enumerate(results):
            neigh_atomic_species = atomic_numbers[neigh_idx]

            neigh_is_atom = neigh_atomic_species != 0
            neigh_atoms = neigh_idx[neigh_is_atom]
            self_index = np.ones_like(neigh_atoms) * i
            edges = np.stack((neigh_atoms, self_index), axis=1)

            neigh_pos = atom_positions[neigh_atoms]
            this_pos = probe_pos[i]
            neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            probe_edges.append(edges)
            probe_edges_displacement.append(neigh_origin_scaled)

        
        probe_edges = np.concatenate(probe_edges, axis=0) 
        probe_edges_displacement = np.concatenate(probe_edges_displacement, axis=0)

        return probe_edges, probe_edges_displacement

        
class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)
        return indices, rel_positions, dist2


def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights

def _sort_by_rows(arr):
    assert len(arr.shape) == 2, "Only 2D arrays"
    return np.array(sorted([tuple(x) for x in arr.tolist()]))