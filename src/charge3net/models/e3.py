# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import ase
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.util.jit import compile_mode
from torch import nn


import src.charge3net.data.layer as layer


def get_irreps(total_mul, lmax):
    """
    Get irreps up to lmax, all with roughly the same multiplicity with a total multiplicity of total_mul
    Example:
        get_irreps(500, lmax=2) = 167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e
    """
    return [
        (round(total_mul / (lmax + 1) / (l * 2 + 1)), (l, p))
        for l in range(lmax + 1)
        for p in [-1, 1]
    ]


class E3DensityModel(nn.Module):
    def __init__(
        self,
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.spin = spin

        self.atom_model = E3AtomRepresentationModel(
            num_interactions,
            num_neighbors,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
        )

        self.probe_model = E3ProbeMessageModel(
            num_interactions,
            num_neighbors,
            self.atom_model.atom_irreps_sequence,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
            spin=spin
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        # if spin == False, (n_batch, n_probe). if spin == True, (n_batch, n_probe, 2)
        # allow it to output spin density of up/down electrons separately
        # TODO: is it better to train on spin up/down density, or charge density + spin density (like in CHGCAR)?
        probe_result = self.probe_model(input_dict, atom_representation)   
        if self.spin:
            spin_up, spin_down = probe_result[:, :, 0], probe_result[:, :, 1]
            probe_result[:, :, 0] = spin_up + spin_down
            probe_result[:, :, 1] = spin_up - spin_down
        return probe_result


class E3AtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species}x 0e"  # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for _ in range(num_interactions):
            # scalar irreps that exist in the tensor product between node and edge irreps
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # store output node irreps for each layer
            self.atom_irreps_sequence.append(irreps_node)  

    def forward(self, input_dict):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])

        # one-hot encode atoms
        nodes = F.one_hot(nodes, num_classes=self.num_species)

        # Node attributes are not used here
        node_attr = nodes.new_ones(nodes.shape[0], 1)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        nodes_list = []
        # Apply interaction layers
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = conv(
                nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            nodes = gate(nodes)
            nodes_list.append(nodes)

        return nodes_list


class E3ProbeMessageModel(torch.nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        atom_irreps_sequence,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # scalar inputs with even parity (for probes its just 0s)
        irreps_node_input = "0e"
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for i in range(num_interactions):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

            conv = ConvolutionOneWay(
                irreps_sender_input=atom_irreps_sequence[i],
                irreps_sender_attr=irreps_node_attr,
                irreps_receiver_input=irreps_node,
                irreps_receiver_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=gate.irreps_in,
                fc_neurons=fc_neurons,
                num_neighbors=num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

        # last layer, scalar output
        if spin:
            out = "1x0e+1x0o"
        else:
            out = "0e"
        self.readout = Linear(irreps_node, out)

    def forward(self, input_dict, atom_representation):
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        probe_edge_vec = calc_edge_vec_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )
        probe_edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), probe_edge_vec, True, normalization="component"
        )
        probe_edge_length = probe_edge_vec.norm(dim=1)
        probe_edge_length_embedding = self.basis(probe_edge_length)

        probe_edge_src = probe_edges[:, 0]
        probe_edge_dst = probe_edges[:, 1]

        # initialize probes
        probes = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 1),
            device=atom_representation[0].device,
        )

        # Probe attributes are not used here
        probe_attr = probes.new_ones(probes.shape[0], 1)

        # Node attributes are not used here
        atom_node_attr = probes.new_ones(atom_xyz.shape[0], 1)

        # Apply interaction layers
        for conv, gate, atom_nodes in zip(
            self.convolutions, self.gates, atom_representation
        ):
            probes = conv(
                atom_nodes,
                atom_node_attr,
                probes,
                probe_attr,
                probe_edge_src,
                probe_edge_dst,
                probe_edge_attr,
                probe_edge_length_embedding,
            )
            probes = gate(probes)

        probes = self.readout(probes).squeeze()

        # rebatch
        probes = layer.pad_and_stack(
            torch.split(
                probes,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
        )
        return probes


class RadialBasis(nn.Module):
    r"""
    Wrapper for e3nn.math.soft_one_hot_linspace, with option for normalization
    Args:
        start (float): mininum value of basis
        end (float): maximum value of basis
        number (int): number of basis functions
        basis ({'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}): basis family
        cutoff (bool): all x outside interval \approx 0
        normalize (bool): normalize function to have a mean of 0, std of 1
        samples (int): number of samples to use to find mean/std
    """
    def __init__(
        self,
        start,
        end,
        number,
        basis="gaussian",
        cutoff=False,
        normalize=True,
        samples=4000
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.number = number
        self.basis = basis
        self.cutoff = cutoff
        self.normalize = normalize

        if normalize:
            with torch.no_grad():
                rs = torch.linspace(start, end, samples+1)[1:]
                bs = soft_one_hot_linspace(rs, start, end, number, basis, cutoff)
                assert bs.ndim == 2 and len(bs) == samples
                std, mean = torch.std_mean(bs, dim=0)
            self.register_buffer("mean", mean)
            self.register_buffer("inv_std", torch.reciprocal(std))
        
    def forward(self, x):
        x = soft_one_hot_linspace(x, self.start, self.end, self.number, self.basis, self.cutoff)
        if self.normalize:
            x = (x - self.mean) * self.inv_std
        return x


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


def calc_edge_vec(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
):
    """
    Calculate vectors of edges
    (modified from src.data.layer.calc_distance)

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3
    return vec


def calc_edge_vec_to_probe(
    positions: torch.Tensor,
    positions_probe: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate vectors of edges from atoms to probes
    (modified from src.data.layer.calc_distance)

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        positions_probe: Tensor of shape (num_probes, 3) with xyz coordinates of probes inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3
    return vec


# Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
# University of California, through Lawrence Berkeley National Laboratory
# (subject to receipt of any required approvals from the U.S. Dept. of Energy), 
# Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
# and Kostiantyn Lapchevskyi. All rights reserved.
# Modified from https://github.com/e3nn/e3nn/blob/05b386177ed039156526f9c67d0d87b6c21ff5d3/e3nn/nn/models/v2103/points_convolution.py
#  - Remove torch_scatter dependency
#  - Add support for differently indexed sending/receiver nodes.
#  - Sender and receiver nodes can have different irreps.
@compile_mode("script")
class Convolution(torch.nn.Module):
    """
    Equivariant Convolution
    Args:
        irreps_node_input (e3nn.o3.Irreps): representation of the input node features
        irreps_node_attr (e3nn.o3.Irreps): representation of the node attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    ) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(
            edge_features, edge_dst, dim_size=node_input.shape[0]
        ).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


@compile_mode("script")
class ConvolutionOneWay(torch.nn.Module):
    """
    Equivariant Convolution, but receiving nodes are differently indexed from sending nodes.
    Additionally, sender and receiver nodes can have different irreps.

    Args:
        irreps_sender_input (e3nn.o3.Irreps): representation of the input sender nodes
        irreps_sender_attr (e3nn.o3.Irreps): representation of the sender attributes
        irreps_receiver_input(e3nn.o3.Irreps): representation of the input receiver nodes
        irreps_receiver_attr (e3nn.o3.Irreps): representation of the receiver attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_sender_input,
        irreps_sender_attr,
        irreps_receiver_input,
        irreps_receiver_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_sender_input = o3.Irreps(irreps_sender_input)
        self.irreps_sender_attr = o3.Irreps(irreps_sender_attr)
        self.irreps_receiver_input = o3.Irreps(irreps_receiver_input)
        self.irreps_receiver_attr = o3.Irreps(irreps_receiver_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_receiver_input,
            self.irreps_receiver_attr,
            self.irreps_node_output,
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_sender_input, self.irreps_sender_attr, self.irreps_sender_input
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_sender_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_sender_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, "0e"
        )

    def forward(
        self,
        sender_input,
        sender_attr,
        receiver_input,
        receiver_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        receiver_self_connection = self.sc(receiver_input, receiver_attr)

        sender_features = self.lin1(sender_input, sender_attr)

        edge_features = self.tp(sender_features[edge_src], edge_attr, weight)

        # scatter edge features from sender (atoms) to receiver (probes)
        receiver_features = scatter(
            edge_features, edge_dst, dim_size=receiver_input.shape[0]
        ).div(self.num_neighbors**0.5)

        receiver_conv_out = self.lin2(receiver_features, receiver_attr)
        receiver_angle = 0.1 * self.lin3(receiver_features, receiver_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = receiver_angle.cos(), receiver_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * receiver_self_connection + sin * receiver_conv_out
