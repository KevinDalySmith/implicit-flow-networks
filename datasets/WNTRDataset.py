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

import networkx as nx
import wntr
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch
from os.path import join
from torch_geometric.data import InMemoryDataset


class WNTRDataset(InMemoryDataset):

    def __init__(self, root, network_name, transform=None, pre_transform=None, pre_filter=None):
        self.network_name = network_name
        super().__init__(root, transform, pre_transform, pre_filter)
        path = join(self.processed_dir, f'{network_name}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [f'{self.network_name}.inp']

    @property
    def processed_file_names(self):
        return [f'{self.network_name}.pt']

    def download(self):
        pass

    def process(self):
        data_list = dataset_from_inp(self.raw_paths[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def dataset_from_inp(inp_path):
    """
    Create a dataset (for one network) from an EPANET file.

    Parameters
    ----------
    inp_path: str
        EPANET file ending in .inp
    n_samples : int
        Number of steady states to sample.
    random_scale : float
        Scale of random variations in demand timeseries.

    Returns
    -------
    data_list : List
        List of PyG data objects containing hourly snapshots of the water network state.

    Notes
    -----
    In order to make the network homogeneous, 'non-junction' nodes are removed.
    Pipes between junctions and non-junctions are also removed.
    The flow rate on these pipes is added to (or subtracted from) the junction's demand.
    """

    # Run WNTR simulation on default 24-hour timeseries of demands
    N_HOURS = 24
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = 3600 * (N_HOURS - 1)
    wn.options.hydraulic.accuracy = 1e-8

    # Eliminate minor losses
    for _, pipe in wn.pipes():
        pipe.minor_loss = 0.0

    # sim = wntr.sim.EpanetSimulator(wn)
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    # Remove non-junctions from the network
    G = load_pipes_and_junctions(wn)
    pipe_btwn_junction_names = {data['name'] for _, _, data in G.edges(data=True)}
    demand = remove_non_junctions(wn, results, set(G.nodes), pipe_btwn_junction_names)

    # Move data to NetworkX graph
    for node_name in G.nodes:
        G.nodes[node_name]['demands'] = demand[node_name].values
        G.nodes[node_name]['pressures'] = results.node['pressure'][node_name].values
        G.nodes[node_name]['heads'] = results.node['head'][node_name].values
    for u, v, data in G.edges(data=True):
        pipe_name = data['name']
        G.edges[(u, v)]['flowrates'] = results.link['flowrate'][pipe_name].values
        G.edges[(u, v)]['velocities'] = results.link['velocity'][pipe_name].values

    # Move data from NetworkX graph to PyG datasets
    all_data = from_networkx(G)
    data_list = []

    edge_attr = torch.stack([
        all_data.length, all_data.diameter, all_data.roughness
    ], dim=1)
    edge_weight = 0.27855 * (all_data.roughness ** 1.852 * all_data.diameter ** 4.871 * all_data.length ** -1) ** 0.54

    for t in range(N_HOURS):

        data = Data(
            edge_index=all_data.edge_index,
            num_nodes=all_data.num_nodes,
            x=-all_data.demands[:, t],
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            heads=all_data.heads[:, t],
            pressures=all_data.pressures[:, t],
            f_true=all_data.flowrates[:, t],
            velocities=all_data.velocities[:, t])

        data_list.append(data)

    return data_list


def remove_non_junctions(wn, results, junction_names, pipe_btwn_junction_names):
    """
    Modify junction demands to remove non-junctions from the dataset.

    Parameters
    ----------
    wn : wntr.network.WaterNetworkModel
        WNTR model.
    results : WNTR simulation results object
        Results of simulation.
    junction_names : collection of strings
        Names of all junction nodes.
    pipe_btwn_junction_names : collection of strings
        Names of all links representing pipes between two junctions.

    Returns
    -------
    demand : pd.DataFrame
        Table of modified junction demands.
    """
    demand, flowrate = results.node['demand'], results.link['flowrate']
    external_flow = flowrate.drop(pipe_btwn_junction_names, axis=1)

    for pipe_name in external_flow.columns:

        source = wn.links[pipe_name].start_node_name
        sink = wn.links[pipe_name].end_node_name

        if source in junction_names and sink not in junction_names:
            demand[source] += external_flow[pipe_name]
        elif source not in junction_names and sink in junction_names:
            demand[sink] -= external_flow[pipe_name]

    return demand


def load_pipes_and_junctions(wn):
    """
    Read all pipes and junctions from a WNTR model into a graph.
    Note that pumps, valves, tanks, reservoirs, etc. are ignored.

    Parameters
    ----------
    wn : wntr.network.WaterNetworkModel
        WNTR model.

    Returns
    -------
    G : nx.Graph
        Simple graph with junctions and pipes.
        Node attributes are:
            - x
            - y
            - elevation
        Edge attributes are:
            - name
            - length
            - diameter
            - roughness
    """

    G = nx.DiGraph()

    # Pumps sources should not be included, as they may have negative pressure
    pump_sources = {pump_data.start_node.name for _, pump_data in wn.pumps()}

    # Record node attributes
    for node in wn.junctions():
            name, data = node
            if name not in pump_sources:
                G.add_node(name,
                           xpos=data.coordinates[0],
                           ypos=data.coordinates[1],
                           zpos=data.elevation)

    # Record edge attributes
    for edge in wn.pipes():

        name, data = edge
        u, v = data.start_node_name, data.end_node_name

        # Only add pipes between junctions
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v,
                       name=name,
                       length=data.length,
                       diameter=data.diameter,
                       roughness=data.roughness)

    return G
