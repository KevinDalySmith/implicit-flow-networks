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
from ifn.utils import cycle_projection, get_f_cut
from .DerivConstrainedPerceptron import DerivConstrainedPerceptron
import logging
logger = logging.getLogger(__name__)


class IFNLayer(torch.nn.Module):

    def __init__(self, hidden_size, dmin, dmax, mitr, eps, use_autograd=True, dtype=torch.float):

        super().__init__()
        self.use_autograd = use_autograd
        self.h_inv = DerivConstrainedPerceptron(1.0 / dmax, 1.0 / dmin, hidden_size, dtype=dtype)
        self.dmin, self.mitr, self.eps = dmin, mitr, eps

    def forward(self, u, data, edge_weights):
        if self.use_autograd:
            f = IFNSolver.solve_flow(u.unsqueeze(-1), self.h_inv, data.edge_index, data.num_nodes,
                                     edge_weights, self.dmin, self.mitr, self.eps,
                                     *tuple(self.h_inv.parameters())).squeeze()
        else:
            f = IFNSolver.apply(u, edge_weights, self.h_inv,
                                data.edge_index, data.num_nodes, self.dmin,
                                self.mitr, self.eps, *tuple(self.h_inv.parameters()))
        return f


class IFNSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, edge_weights, h_inv, edge_index, num_nodes, d_min, mitr, eps, *args):

        # Compute solution to the flow equations
        logger.info('FixedPointIFN solving forward pass')
        f = IFNSolver.solve_flow(u.unsqueeze(-1), h_inv, edge_index, num_nodes, edge_weights, d_min, mitr, eps, *args)
        f = f.squeeze(-1)

        # Compute Jacobian matrices of h_inv, evaluated at the solution
        y = f / edge_weights
        jac_mats = torch.autograd.functional.jacobian(h_inv, (y,) + args)
        jac_mats = [jac_mat.unsqueeze(-1) if jac_mat.ndim < 2 else jac_mat for jac_mat in jac_mats]
        y_grad = torch.diag(jac_mats[0])

        # Compute Jacobian matrices of v with respect to theta and a
        theta_jacs = [edge_weights.unsqueeze(-1) * jac for jac in jac_mats[1:]]
        a_grad = h_inv(y, *args) - f * y_grad / edge_weights

        ctx.save_for_backward(
            edge_weights, y_grad, edge_index, torch.as_tensor(num_nodes),
            torch.as_tensor(d_min), torch.as_tensor(mitr), torch.as_tensor(eps), a_grad, *theta_jacs)

        return f

    @staticmethod
    def backward(ctx, backward_f):

        # Unpack data saved from the forward pass
        edge_weights, y_grad, edge_index, num_nodes, d_min, mitr, eps, a_grad = ctx.saved_tensors[:8]
        theta_jacs = ctx.saved_tensors[8:]
        num_nodes, d_min, mitr, eps = num_nodes.item(), d_min.item(), mitr.item(), eps.item()

        # Compute gradient wrt edge weights
        logger.info('FixedPointIFN solving edge weight gradients')
        num_edges = edge_index.shape[-1]
        df_da = IFNSolver.solve_flow(
            U=torch.zeros(num_nodes, num_edges),
            h_inv=lambda Y: y_grad.unsqueeze(-1) * Y + torch.diag(a_grad / edge_weights),
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
            d_min=d_min,
            mitr=mitr,
            eps=eps)
        da = backward_f @ df_da

        # Compute gradients wrt thetas
        dthetas = []
        for theta_jac in theta_jacs:
            logger.info('FixedPointIFN solving theta gradients')
            df_dtheta = IFNSolver.solve_flow(
                U=torch.zeros(num_nodes, theta_jac.shape[-1]),
                h_inv=lambda Y: y_grad.unsqueeze(-1) * Y + theta_jac / edge_weights.unsqueeze(-1),
                edge_index=edge_index,
                num_nodes=num_nodes,
                edge_weights=edge_weights,
                d_min=d_min,
                mitr=mitr,
                eps=10 * eps)
            dtheta = backward_f @ df_dtheta
            dthetas.append(dtheta)

        # Compute gradients wrt u (not yet implemented)
        du = None

        return (du, da) + (None,) * 6 + tuple(dthetas)

    @staticmethod
    def solve_flow(U, h_inv, edge_index, num_nodes, edge_weights, d_min, mitr, eps, *args):
        """
        Compute the output of the implicit flow network.

        Parameters
        ----------
        U : torch.Tensor
            (n, d) tensor of balanced nodal injections.
        h_inv : Callable
            Function that maps an (m, d) tensor of potential differences to an (m, d) tensor of unweighted flows.
        edge_index : torch.LongTensor
            (m, 2) tensor of edge indices.
        num_nodes : int
            Number of nodes, n.
        edge_weights : torch.Tensor
            (m,) tensor of edge weights.
        d_min : float
            Minimum slope of h (i.e., reciprocal of the maximum slope of h_inv).
        mitr : int
            Max number of iterations.
        eps : float
            Convergence tolerance

        Returns
        -------
        f : torch.Tensor
            (m, d) flow vector.
        """
        logger.debug('Running FixedPointIFN.solve_flow')
        f_cut = get_f_cut(edge_index, num_nodes, U)
        f_cyc = torch.zeros_like(f_cut)
        for i in range(mitr):
            f_cyc_new = IFNSolver.fixed_point_map(f_cyc, f_cut, h_inv, d_min, edge_index, num_nodes, edge_weights, *args)
            err = torch.linalg.norm(
                (f_cyc_new - f_cyc) / edge_weights.sqrt().unsqueeze(-1),
                dim=0).max()
            # rel_errs = torch.linalg.norm(f_cyc_new - f_cyc, dim=0) / (1e-6 + torch.linalg.norm(f_cyc_new, dim=0))
            # err = rel_errs.max()
            f_cyc = f_cyc_new
            logger.debug(f'Iter {i+1} err={err:.4f}')
            if err <= eps:
                return f_cyc + f_cut
        print(f'IFNSolver did not converge; final error={err:.6f}')
        return f_cyc + f_cut
        
    @staticmethod
    def fixed_point_map(f_cyc, f_cut, h_inv, d_min, edge_index, num_nodes, edge_weights, *args):
        """
        Evaluate the fixed point map T(f_cyc).

        Parameters
        ----------
        f_cyc : torch.Tensor
            (m, d) cycle space component of f.
        f_cut : torch.Tensor
            (m, d) cutset space component of f.
        h_inv : Callable
            Function that maps an (m, d) tensor of potential differences to an (m, d) tensor of unweighted flows.
        d_min : float
            Minimum slope of h (i.e., reciprocal of the maximum slope of h_inv).
        edge_index : torch.LongTensor
            (m, 2) tensor of edge indices.
        num_nodes : int
            Number of nodes, n.
        edge_weights : torch.Tensor
            (m, d) tensor of edge weights.

        Returns
        -------
        t : torch.Tensor
            (m,) evaluation of T(f_cyc).
        """
        y = (f_cyc + f_cut) / edge_weights.unsqueeze(-1)
        z = f_cyc - d_min * edge_weights.unsqueeze(-1) * h_inv(y, *args)
        t = cycle_projection(z, edge_index, edge_weights, num_nodes)
        return t
