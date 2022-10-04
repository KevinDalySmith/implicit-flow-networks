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
from constants import DTYPE
from ifn.modules import IFNLayer
from experiments.TransductiveFlowEstimation.TransductiveFlowEstimationUtils import edge_set_to_mask


def train_test_ifn_loglinweight(experiment, dev):

    # Get boolean mask of edges to use for training
    # Mask is such that data.f_true[train_edge_idx] is the tensor of training flows on physical edges
    # Note that virtual edge flows are not included in IFN's training data, since IFN incorporates
    # external injections directly and does not operate on the augmented network.
    physical_train_edges = set(experiment.train_flows.keys()) - set(experiment.virtual_edges)
    train_edge_idx = edge_set_to_mask(experiment.data.edge_index, physical_train_edges)

    # Train and make predictions
    n_edge_features = experiment.data.edge_attr.shape[-1]
    ifn_model = IFNTransductiveFlowEstimator(
        edge_weight_model=LogLinearModel(n_edge_features),
        **experiment.model_params)
    ifn_model.trainloop(experiment.data, train_edge_idx)
    experiment.pred_flows = ifn_model(experiment.data)

    # Compute the RMSE
    test_edge_idx = edge_set_to_mask(experiment.data.edge_index, set(experiment.test_flows.keys()))
    test_flow_error = experiment.pred_flows[test_edge_idx] - experiment.data.f_true[test_edge_idx]
    test_flow_error = experiment.scale_factor * test_flow_error
    experiment.test_rmse = test_flow_error.square().mean().sqrt().item()

    return experiment


def train_test_ifn(experiment, dev):

    # Get boolean mask of edges to use for training
    # Mask is such that data.f_true[train_edge_idx] is the tensor of training flows on physical edges
    # Note that virtual edge flows are not included in IFN's training data, since IFN incorporates
    # external injections directly and does not operate on the augmented network.
    physical_train_edges = set(experiment.train_flows.keys()) - set(experiment.virtual_edges)
    train_edge_idx = edge_set_to_mask(experiment.data.edge_index, physical_train_edges)

    # Train and make predictions
    ifn_model = IFNTransductiveFlowEstimator(
        edge_weight_model=None,
        **experiment.model_params)
    ifn_model.trainloop(experiment.data, train_edge_idx)
    experiment.pred_flows = ifn_model(experiment.data)

    # Compute the RMSE
    test_edge_idx = edge_set_to_mask(experiment.data.edge_index, set(experiment.test_flows.keys()))
    test_flow_error = experiment.pred_flows[test_edge_idx] - experiment.data.f_true[test_edge_idx]
    test_flow_error = experiment.scale_factor * test_flow_error
    experiment.test_rmse = test_flow_error.square().mean().sqrt().item()

    return experiment


class IFNTransductiveFlowEstimator(torch.nn.Module):

    def __init__(self, hidden_size, dmin, dmax, mitr, eps, lr, epochs, early_stop, edge_weight_model, **kwargs):
        super().__init__()
        self.edge_weight_model = edge_weight_model
        self.ifn_layer = IFNLayer(hidden_size, dmin, dmax, mitr, eps, dtype=DTYPE)
        self.lr, self.epochs, self.early_stop = lr, epochs, early_stop

    def forward(self, data):
        if self.edge_weight_model is None:
            edge_weight = data.edge_weight
        else:
            edge_weight = self.edge_weight_model(data)
        f = self.ifn_layer(data.x, data, edge_weight)
        return f

    def trainloop(self, data, train_edge_idx):

        opt = torch.optim.Adam(self.parameters(), self.lr)
        train_losses = []

        for epoch in range(self.epochs):

            # Compute loss
            opt.zero_grad()
            f_predicted = self.forward(data)
            train_flow_error = f_predicted[train_edge_idx] - data.f_true[train_edge_idx]
            train_loss = train_flow_error.square().mean().sqrt()

            # Backprop
            train_loss.backward()
            opt.step()

            # Check for early stopping
            print("epoch: ", epoch, " train loss = ", train_loss.item())
            train_losses.append(train_loss.item())
            if epoch > self.early_stop and train_losses[-1] > np.mean(train_losses[-(self.early_stop + 1):-1]):
                print("Early stopping...")
                break


class LogLinearModel(torch.nn.Module):

    def __init__(self, n_edge_attr):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_edge_attr, 1, dtype=DTYPE))

    def forward(self, data):
        x = data.edge_attr
        x = x.log()
        x = self.model(x)
        x = x.exp()
        return x.squeeze()
