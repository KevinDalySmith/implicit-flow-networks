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
from torch_geometric.utils import coalesce


def cycle_projection(X, edge_index, edge_weights, num_nodes, driver='gelsd'):
    """
    Oblique projection onto the cycle space ker(B) parallel to Img(AB.T).

    Parameters
    ----------
    X : torch.Tensor
        (m, d) matrix to project.
    edge_index : torch.LongTensor
        (2, m) edge index tensor.
    edge_weights : torch.Tensor
        (m,) edge weight tensor.
    num_nodes : int
        Number of nodes, n.
    driver : str, optional, default=gelsd
        Which LAPACK / MAGMA function to use for linear least squares.
        See https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html for details.
        Default is "gelsd", which works well for poorly-conditioned matrices, at the
        expense of only running on the CPU.

    Returns
    -------
    Y : torch.Tensor
        (m, d) projection of X.
    """

    m = edge_index.shape[-1]
    weights = torch.sparse_coo_tensor(
        torch.stack([torch.arange(m,), torch.arange(m,)], dim=0),
        edge_weights,
        size=(m, m))

    # Construct Laplacian matrix
    B = get_incidence_matrix(edge_index, num_nodes).type(weights.dtype)
    Bt = B.transpose(-2, -1)
    L = torch.sparse.mm(torch.sparse.mm(B, weights), Bt)

    # Project
    V = multiply_incidence_matrix(edge_index, X, num_nodes)
    S, _, _, _ = torch.linalg.lstsq(L.to_dense(), V, driver=driver)
    T = edge_weights.unsqueeze(-1) * (S[edge_index[0], :] - S[edge_index[1], :])
    return X - T


def get_f_cut(edge_index, num_nodes, u, driver='gelsd'):
    """
    Find the vector f_cut in the cutset space that solves B @ f_cut = u.

    Parameters
    ----------
    edge_index : torch.LongTensor
        (2, m) edge index tensor.
    num_nodes : int
        Number of nodes, n.
    u : torch.Tensor
        (n,) tensor.
    driver : str, optional, default=gels
        Which LAPACK / MAGMA function to use for linear least squares.
        See https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html for details.

    Returns
    -------
    f_cut : torch.Tensor
        (m,) tensor.
    """
    B = get_incidence_matrix(edge_index, num_nodes)
    B = B.to_dense().type(u.dtype)
    f_cut, _, _, _ = torch.linalg.lstsq(B, u, driver=driver)
    return f_cut


def multiply_incidence_matrix(edge_index, X, num_nodes, transpose=False, mode=0):
    """
    Multiply the incidence matrix with a batch of other matrices.
    Does not explicitly construct the incidence matrix.
    We adopt the sign convention where B(i,e) = 1 for
    the source node i, and B(j,e) = -1 for the target j.

    Parameters
    ----------
    edge_index : torch.LongTensor
        (2, m) edge index tensor.
    X : torch.Tensor
        (..., m, d) or (..., n, d) batch of matrices to multiply.
    num_nodes : int
        Number of nodes, n.
    transpose : bool, optional, default='normal'
        Whether to compute B'X instead of BX.
    mode : int, optional, default=0
        Multiplication mode. Options include:
         - 0: use signed incidence matrix: Y = BX
         - 1: use absolute value of incidence matrix: Y = |B|X
         - 2: use in-incidence matrix: Y = BinX
         - 3: use out-incidence matrix: Y = BoutX

    Returns
    -------
    Y : torch.Tensor
        (..., n, d) or (..., m, d) batch of matrix products.
    """

    if transpose:

        source_embedding = X[..., edge_index[0], :]
        target_embedding = X[..., edge_index[1], :]

        if mode == 0:
            Y = source_embedding - target_embedding
        elif mode == 1:
            Y = source_embedding + target_embedding
        elif mode == 2:
            Y = target_embedding
        elif mode == 3:
            Y = source_embedding
        else:
            raise ValueError('Mode "{}" is not valid'.format(mode))

    else:

        y_size = X.shape[:-2] + (num_nodes, X.shape[-1])
        Y = torch.zeros(size=y_size, dtype=X.dtype, device=X.device)

        if mode in [0, 1, 3]:
            edge_source_index = edge_index[0].unsqueeze(-1).expand(*X.shape)
            Y.scatter_add_(-2, edge_source_index, X)

        if mode == 0:
            edge_target_index = edge_index[1].unsqueeze(-1).expand(*X.shape)
            Y.scatter_add_(-2, edge_target_index, -X)
        elif mode in [1, 2]:
            edge_target_index = edge_index[1].unsqueeze(-1).expand(*X.shape)
            Y.scatter_add_(-2, edge_target_index, X)

    return Y


def get_incidence_matrix(edge_index, num_nodes, force_undirected=False):
    """
    Create signed incidence matrix.

    Parameters
    ----------
    edge_index : torch.LongTensor
        Edge index tensor.
    num_nodes : int
        Number of nodes in the graph.
    force_undirected : bool, optional, default=False
        If True, B will represent an undirected graph with arbitrary edge orientation.
        If False, B will have precisely one edge corresponding to each column of edge_index,
        in the same order.

    Returns
    -------
    B : torch.Tensor
        Incidence matrix.
    """

    # Get unique undirected edges and impose arbitrary edge orientations
    if force_undirected:
        edge_index_oriented, _ = edge_index.sort(dim=0)
        edge_index = coalesce(edge_index_oriented, num_nodes=num_nodes)

    m = edge_index.shape[1]
    idx = torch.stack([
        torch.cat([edge_index[0], edge_index[1]]),
        torch.arange(m, device=edge_index.device).repeat(2)])
    vals = torch.cat([torch.ones(m, dtype=torch.long), -torch.ones(m, dtype=torch.long)])
    B = torch.sparse_coo_tensor(idx, vals, (num_nodes, m), device=edge_index.device)

    return B
