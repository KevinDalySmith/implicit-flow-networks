# Copyright 2022 by Kevin D. Smith and Francesco Seccamonte.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from scipy.io import loadmat
from os.path import join
from ifn.utils import num2index

# MATPOWER data format is specified in Appendix B of the user manual:
# https://matpower.org/docs/MATPOWER-manual.pdf

# MATPOWER bus data columns
BUS_I = 0    # int label for the bus
PD = 2       # real power demand (MW)
GS = 4       # shunt conductance (pu)
VM = 7       # voltage magntiude (pu)

# MATPOWER generator data columns
GEN_BUS = 0  # int label of the bus
PG = 1       # real power generation (MW)

# MATPOWER branch data columns
F_BUS = 0    # int label of the source bus
T_BUS = 1    # int label of the destination bus
BR_X = 3     # series reactance (pu)
TAP = 8      # tap ratio
PF = 13      # active power flow (MW)


class MatpowerDataset(InMemoryDataset):

    def __init__(self, root, case_name, transform=None, pre_transform=None, pre_filter=None):
        self.case_name = case_name
        super().__init__(root, transform, pre_transform, pre_filter)
        path = join(self.processed_dir, f'{case_name}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [f'{self.case_name}TS.m']

    @property
    def processed_file_names(self):
        return [f'{self.case_name}.pt']

    def download(self):
        pass

    def process(self):

        data_list = []

        for filename in self.raw_file_names:

            # Load MATPOWER data
            path = join(self.raw_dir, filename)
            matpower_data = loadmat(path)
            branch_data = matpower_data['branchResults']
            bus_data = matpower_data['busResults']
            gen_data = matpower_data['genResults']

            # Construct edge index
            src_node = num2index(branch_data[0, :, F_BUS], bus_data[0, :, BUS_I])
            dest_node = num2index(branch_data[0, :, T_BUS], bus_data[0, :, BUS_I])
            edge_index = torch.LongTensor(np.stack([src_node, dest_node]))

            # Construct PyG data objects
            for t in range(bus_data.shape[0]):
                edge_attr, edge_weight = relevant_edge_attr_lossless(
                    edge_index, bus_data[t, ...], branch_data[t, ...])
                data = Data(
                    edge_index=edge_index,
                    num_nodes=bus_data.shape[1],
                    x=compute_net_injections(bus_data[t, ...], gen_data[t, ...]),
                    edge_attr=edge_attr,
                    edge_weight=torch.as_tensor(edge_weight, dtype=torch.float),
                    f_true=torch.as_tensor(branch_data[t, :, PF], dtype=torch.float))
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def relevant_edge_attr_lossless(edge_index, bus_data, branch_data):
    """
    Select the edge attributes that are relevant to active power flow
    in lossless AC grids.
    There are four relevant attributes:
        1. Source node voltage magnitude
        2. Destination node voltage magnitude
        3. Tap ratio (1.0 for transmission lines, other for transformers)
        4. Series reactance
    """
    src_vm = bus_data[edge_index[0], VM]
    dest_vm = bus_data[edge_index[1], VM]
    tap = branch_data[:, TAP]
    # For some reason, MATPOWER sets tap = 0 for transmission lines, but
    # physically, trans lines have a tap ratio of 1
    tap[tap == 0] = 1.0
    reac = branch_data[:, BR_X]
    edge_attr = torch.stack([
        torch.as_tensor(src_vm, dtype=torch.float),
        torch.as_tensor(dest_vm, dtype=torch.float),
        torch.as_tensor(tap, dtype=torch.float),
        torch.as_tensor(reac, dtype=torch.float)], dim=-1)

    # Compute edge weights
    a = src_vm * dest_vm / reac / tap
    edge_weight = 1.0 / a
    return edge_attr, edge_weight


def compute_net_injections(bus_data, gen_data):
    """
    Compute the net active power injections at each bus.
    Net power inections are the sum of three quantities:
        1. Injections from generators
        2. Withdrawals due to constant demands
        3. Withdrawals due to shunt conductance
    """
    n_bus = bus_data.shape[0]
    u = np.zeros(n_bus, )
    gen_bus_idx = num2index(gen_data[:, GEN_BUS], bus_data[:, BUS_I])
    u[gen_bus_idx] += gen_data[:, PG]       # Generator injections
    u -= bus_data[:, PD]                    # Constant demands
    vmag_sqr = np.square(bus_data[:, VM])
    u -= vmag_sqr * bus_data[:, GS]         # Shunt demands
    return torch.as_tensor(u, dtype=torch.float)
