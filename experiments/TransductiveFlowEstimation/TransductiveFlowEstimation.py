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

import random
from os.path import join, dirname
from dataclasses import dataclass
from typing import List
import torch_geometric.data

from constants import SOURCE_NODE
from datasets import REGISTERED_DATASETS
from experiments.TransductiveFlowEstimation.TransductiveFlowEstimationUtils import *
from experiments.TransductiveFlowEstimation.TransductiveFlowEstimationModels import *

# Registry of train-test functions for each model that is implemented
TRAIN_TEST_FUNCTIONS = {
    'ifn': train_test_ifn,
    'ifn+loglinweight': train_test_ifn_loglinweight
}


@dataclass(init=False, repr=False)
class MissingFlowsExperiment:
    """
    Class to store transductive flow estimation data, settings, and results.
    """

    # Network data
    dataset_name: str       # Name of the dataset (e.g., 'water' or 'power')
    network_name: str       # Name of the particular network (e.g., 'case300' or 'fairfield')
    data: torch_geometric.data.Data
    graph: nx.DiGraph       # NetworkX graph with true edge flows and attribs
    virtual_edges: List     # List of virtual edges (i.e., edges representing nodal injections)

    # Model details
    model_name: str
    model_params: dict

    # Experiment settings
    frac_labeled: float     # Fraction of edges that are labeled
    train_flows: dict
    test_flows: dict
    scale_factor: float
    seed: int

    # Results
    edge_weights: dict
    pred_flows: dict        # Predicted edge flows
    test_rmse: float        # Testing RMSE


def run(dataset_name, network_name, model_name, model_params, seed, frac_labeled, dev_name):

    if model_name in TRAIN_TEST_FUNCTIONS:
        train_test_fn = TRAIN_TEST_FUNCTIONS[model_name]
    else:
        raise ValueError(f'Model {model_name} is not implemented for transductive flow estimation')

    dev = torch.device(dev_name)

    experiment = prepare_experiment(dataset_name, network_name, model_name, model_params, seed, frac_labeled)
    experiment = train_test_fn(experiment, dev)
    return experiment


def prepare_experiment(dataset_name, network_name, model_name, model_params, seed, frac_labeled):
    """
    Prepare experiment.
    """

    # Load the network flow data as a PyG object and cast to desired dtype
    data_dir, dataset_module = REGISTERED_DATASETS[dataset_name]
    data_path = join(dirname(f'./data/{data_dir}/'))
    data = dataset_module(data_path, network_name)[0]
    data.x = data.x.type(DTYPE)
    data.f_true = data.f_true.type(DTYPE)
    data.edge_attr = data.edge_attr.type(DTYPE)
    if hasattr(data, 'edge_weight'):
        data.edge_weight = data.edge_weight.type(DTYPE)

    # Convert PyG data to networkx graph, for compatability with Silva's code
    n_edges = data.edge_index.shape[-1]
    nx_data = pyg_to_nx(data)

    # Add source node / edges to incorporate known nodal injections in the baselines
    virtual_edge_attr = np.zeros(data.edge_attr.shape[-1],)
    nx_data, virtual_edges = add_source_node(nx_data, data.x, SOURCE_NODE, virtual_edge_attr)

    # Normalize flows, and ensure all flows are positive by switching edge orientations
    # (Silva's code performs this pre-processing step)
    # The normalization operation is f -> f / scale_factor, and
    # scale_factor is the magnitude of the largest flow.
    nx_data, scale_factor = make_non_neg_norm(nx_data)
    data.x = data.x / scale_factor
    data.f_true = data.f_true / scale_factor

    # Swap order of virtual edges, if the order was swapped in make_non_neg_norm
    for e_idx in range(len(virtual_edges)):
        if virtual_edges[e_idx] not in nx_data.edges:
            i, j = virtual_edges[e_idx]
            virtual_edges[e_idx] = (j, i)

    # Identify the physical edges (i.e., the edges that are not virtual)
    physical_edges = [edge for edge in nx_data.edges if
                      edge[0] != SOURCE_NODE and edge[1] != SOURCE_NODE]

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_labeled_edges = int(frac_labeled * n_edges)

    # Create new experiment data object
    experiment = MissingFlowsExperiment()
    experiment.dataset_name = dataset_name
    experiment.network_name = network_name
    experiment.data = data
    experiment.graph = nx_data
    experiment.virtual_edges = virtual_edges
    experiment.model_name = model_name
    experiment.model_params = model_params
    experiment.frac_labeled = frac_labeled
    experiment.scale_factor = scale_factor
    experiment.seed = seed

    # Randomly sample labeled edges from the physical edges
    # Note that virtual edges (representing external injections) are not sampled,
    # and do not count toward the number of labeled edges.
    labeled_edges = random.sample(physical_edges, n_labeled_edges)
    unlabeled_edges = [edge for edge in physical_edges if edge not in labeled_edges]

    # Set up the training data
    # Training flows consist (i) flows from all labeled edges, and (ii) flows on
    # ALL virtual edges, as external injections are known.
    experiment.train_flows = {edge: nx_data.edges[edge]['f_true'] for edge in labeled_edges}
    experiment.train_flows.update({edge: nx_data.edges[edge]['f_true'] for edge in virtual_edges})

    # Set up the testing data
    # Testing flows consist of flows from all unlabeled physical edges.
    # Testing flows do NOT include flows on virtual edges.
    experiment.test_flows = {edge: nx_data.edges[edge]['f_true'] for edge in unlabeled_edges}

    return experiment
