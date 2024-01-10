# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from typing import List
import torch
import numpy as np
import ase
import ase.neighborlist
from src.charge3net.data.layer import pad_and_stack
import math
import multiprocessing
import time
import queue
import logging

class CollateFuncRandomSample:
    def __init__(self, cutoff, num_probes, pin_memory=True, disable_pbc=False, probe_batch_size=50):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.disable_pbc = disable_pbc
        self.probe_batch_size = probe_batch_size

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.disable_pbc:
                atoms = i["atoms"].copy()
                atoms.set_pbc(False)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_and_probe_sample_to_graph_dict(
                i["density"],
                atoms,
                i["grid_position"],
                self.cutoff,
                self.num_probes,
                probe_batch_size=self.probe_batch_size
            ))

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)

class CollateFuncAtoms:
    def __init__(self, cutoff, pin_memory=True, disable_pbc=False):
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.disable_pbc = disable_pbc

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.disable_pbc:
                atoms = i["atoms"].copy()
                atoms.set_pbc(False)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_to_graph_dict(
                atoms,
                self.cutoff,
            ))

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


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

def grid_iterator_worker(atoms, meshgrid, probe_count, cutoff, slice_id_queue, result_queue):
    try:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)
    except Exception as e:
        logging.info("Failed to create asap3 neighborlist, this might be very slow. Error: %s", e)
        neighborlist = None
    while True:
        try:
            slice_id = slice_id_queue.get(True, 1)
        except queue.Empty:
            while not result_queue.empty():
                time.sleep(1)
            result_queue.close()
            return 0
        res = DensityGridIterator.static_get_slice(slice_id, atoms, meshgrid, probe_count, cutoff, neighborlist=neighborlist)
        result_queue.put((slice_id, res))

class DensityGridIterator:
    def __init__(self, densitydict, ignore_pbc: bool, probe_count: int, cutoff: float):
        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        self.num_slices = int(math.ceil(num_positions / probe_count))
        self.probe_count = probe_count
        self.cutoff = cutoff
        self.ignore_pbc = ignore_pbc

        if ignore_pbc:
            self.atoms = densitydict["atoms"].copy()
            self.atoms.set_pbc(False)
        else:
            self.atoms = densitydict["atoms"]

        self.meshgrid = densitydict["grid_position"]

    def get_slice(self, slice_index):
        return self.static_get_slice(slice_index, self.atoms, self.meshgrid, self.probe_count, self.cutoff)

    @staticmethod
    def static_get_slice(slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None):
        num_positions = np.prod(meshgrid.shape[0:3])
        flat_index = np.arange(slice_index*probe_count, min((slice_index+1)*probe_count, num_positions))
        pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
        probe_pos = meshgrid[pos_index]
        probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist)

        if not probe_edges:
            probe_edges = [np.zeros((0,2), dtype=np.int)]
            probe_edges_displacement = [np.zeros((0,3), dtype=np.float32)]

        res = {
            "probe_edges": np.concatenate(probe_edges, axis=0),
            "probe_edges_displacement": np.concatenate(probe_edges_displacement, axis=0).astype(np.float32),
        }
        res["num_probe_edges"] = res["probe_edges"].shape[0]
        res["num_probes"] = len(flat_index)
        res["probe_xyz"] = probe_pos.astype(np.float32)

        return res


    def __iter__(self):
        self.current_slice = 0
        slice_id_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(100)
        self.finished_slices = dict()
        for i in range(self.num_slices):
            slice_id_queue.put(i)
        self.workers = [multiprocessing.Process(target=grid_iterator_worker, args=(self.atoms, self.meshgrid, self.probe_count, self.cutoff, slice_id_queue, self.result_queue)) for _ in range(6)]
        for w in self.workers:
            w.start()
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1

            # Retrieve finished slices until we get the one we are looking for
            while this_slice not in self.finished_slices:
                i, res = self.result_queue.get()
                res = {k: torch.tensor(v) for k,v in res.items()} # convert to torch tensor
                self.finished_slices[i] = res
            return self.finished_slices.pop(this_slice)
        else:
            for w in self.workers:
                w.join()
            raise StopIteration


def atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes, probe_batch_size=50):
    # Sample probes on the calculated grid
    probe_choice_max = np.prod(grid_pos.shape[0:3])
    probe_choice = np.random.randint(probe_choice_max, size=num_probes)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    probe_pos = grid_pos[probe_choice]
    probe_target = density[probe_choice]

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)
    probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, 
        neighborlist=neighborlist, inv_cell_T=inv_cell_T, batch_size=probe_batch_size)

    default_type = torch.get_default_dtype()

    if not probe_edges:
        probe_edges = [np.zeros((0,2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0,3), dtype=np.int)]
    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
        "probe_edges_displacement": torch.tensor(
            np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_target": torch.tensor(probe_target, dtype=default_type),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(res["probe_target"].shape[0])
    res["probe_xyz"] = torch.tensor(probe_pos, dtype=default_type)
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    # NOTE (teddy): tensor from list of ndarrays is very slow, see 
    # https://github.com/pytorch/pytorch/issues/13918
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)
    res["voxel_volume"] = torch.tensor(atoms.get_volume()/np.prod(density.shape))
    res["num_voxels"] = torch.tensor(np.prod(grid_pos.shape[:3]))

    return res

def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = torch.get_default_dtype()

    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res

def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
        np.any(atoms.get_cell().lengths() <= 0.0001)
        or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        # neighborlist = AseNeigborListWrapper(cutoff, atoms)
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T

def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None, batch_size=0):
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        use_ase = (
                np.any(atoms.get_cell().lengths() <= 0.0001)
                or (
                    np.any(atoms.get_pbc())
                    and np.any(_cell_heights(atoms.get_cell()) < cutoff)
                )
            )

        num_probes = probe_pos.shape[0]
        split_idx = np.ceil(num_probes/batch_size) if batch_size > 0 else 1
        probe_positions = np.split(probe_pos, split_idx)
        results = []
        for probe_position in probe_positions:
            n_batch_probes = probe_position.shape[0]
            probe_atoms = ase.Atoms(numbers=[0] * n_batch_probes, positions=probe_position)
            atoms_with_probes = atoms.copy()
            atoms_with_probes.extend(probe_atoms)
            atomic_numbers = atoms_with_probes.get_atomic_numbers()
            if use_ase:
                neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
            else:
                # neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
                neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

            results.extend( [neighborlist.get_neighbors(i+len(atoms), cutoff) for i in range(n_batch_probes)])

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

    return probe_edges, probe_edges_displacement

def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {}
    for k,v  in dict_of_lists.items():
        if not k in ["filename", "load_time"]:
            collated[k] = pin(pad_and_stack(v))
        else:
            collated[k] = v
    # collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated

    
def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights