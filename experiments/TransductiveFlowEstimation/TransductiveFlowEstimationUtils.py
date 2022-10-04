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
import networkx as nx


def pyg_to_nx(data):
    """
    Convert flow network data in a PyG object to a format compatible with Silva's code.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG data object.

    Returns
    -------
    G : nx.DiGraph
        NetworkX graph.
        Nodes are named {0, 1, ..., n-1}.
        Edge weights include:
            - f: ground-truth edge flow
            - edge_attr: array of edge attributes
            - edge_weight: (optional) array of edge weights
    """
    G = nx.DiGraph()
    for m in range(data.edge_index.shape[-1]):

        if hasattr(data, 'edge_weight'):
            G.add_edge(data.edge_index[0, m].item(),
                       data.edge_index[1, m].item(),
                       f_true=data.f_true[m].item(),
                       edge_attr=data.edge_attr[m, :].numpy(),
                       edge_weight=data.edge_weight[m].item())
        else:
            G.add_edge(data.edge_index[0, m].item(),
                       data.edge_index[1, m].item(),
                       f_true=data.f_true[m].item(),
                       edge_attr=data.edge_attr[m, :].numpy())

    return G


def add_source_node(G, injections, source_node, virtual_edge_attr):
    """
    Incorporate known nodal injections by adding a source node and virtual edges.
    Assumes that nodes in G are named {0, 1, ..., n-1}.
    The graph G is modified in place.
    G and a list of new virtual edges is returned.
    """
    virtual_edges = []
    original_nodes = list(G.nodes)
    for i in original_nodes:
        G.add_edge(source_node,
                   i,
                   f_true=injections[i],
                   edge_attr=virtual_edge_attr)
        virtual_edges.append((source_node, i))
    return G, virtual_edges


def make_non_neg_norm(G):
    """
    Normalize all flows in G.
    Reverse the orientation of edges with negative flows, to ensure that all
    flows are positive.
    Edge attributes are not modified.
    The graph G is modified in place.
    G and the normalization scaling factor are returned.
    """
    scale_factor = max(map(abs, nx.get_edge_attributes(G, 'f_true').values()))
    original_edges = list(G.edges)
    for i, j in original_edges:
        norm_flow = G.edges[(i, j)]['f_true'] / scale_factor
        if G.edges[(i, j)]['f_true'] < 0:
            G.add_edge(j, i, f_true=-norm_flow, edge_attr=G.edges[(i, j)]['edge_attr'])
            G.remove_edge(i, j)
        else:
            G.edges[(i, j)]['f_true'] = norm_flow
    return G, scale_factor


def edge_set_to_mask(edge_index, edge_set):

    # Create map from each edge to its column index in edge_index
    edge_to_idx = dict()
    for edge_idx in range(edge_index.shape[-1]):
        source = edge_index[0, edge_idx].item()
        dest = edge_index[1, edge_idx].item()
        edge_to_idx[(source, dest)] = edge_idx

    # Create mask
    mask = torch.zeros(edge_index.shape[-1], dtype=torch.bool)
    for u, v in edge_set:
        if (u, v) in edge_to_idx:
            edge_idx = edge_to_idx[(u, v)]
        else:
            edge_idx = edge_to_idx[(v, u)]
        mask[edge_idx] = True

    return mask
