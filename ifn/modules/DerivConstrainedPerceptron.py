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


class DerivConstrainedPerceptron(torch.nn.Module):

    def __init__(self, dmin, dmax, hidden_size, dtype=torch.float):
        """
        Construct a scalar-to-scalar perceptron, whose slope satisfies an interval constraint.

        Parameters
        ----------
        dmin : float
            Minimum value of the slope.
        dmax : float
            Maximum value of the slope.
        hidden_size : int
            Size of the hidden layer.
        """
        super().__init__()

        self.dmin, self.dmax = dmin, dmax

        self.a = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype))
        self.b = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype))
        self.c_hat = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype))

        torch.nn.init.normal_(self.a)
        torch.nn.init.normal_(self.b)
        torch.nn.init.normal_(self.c_hat)

        self.d_center = (dmin + dmax) / 2
        self.d_range = (dmax - dmin) / 2

    def forward(self, X, a, b, c_hat):
        """
        Evaluate the function element-wise on a given input.
        Note that the model parameters a, b, c_hat must be provided.

        Parameters
        ----------
        X : torch.tensor
            Input.
        a : torch.tensor
            Literally self.a. This is to allow for easy gradient computation in FixedPointIFN.
        b : torch.tensor
            Literally self.b.
        c_hat : torch.tensor
            Literally self.c_hat.

        Returns
        -------
        Y : torch.tensor
            Output, the same size as X.
        """
        A = a.expand(X.shape + (-1,))
        B = b.expand(X.shape + (-1,))
        Y = torch.relu(A * X.unsqueeze(-1) + B)
        Y = Y @ self.compute_c(a, c_hat)
        Y = (self.d_range * Y) + (self.d_center * X)
        return Y

    @staticmethod
    def compute_c(a, c_hat):
        ca_norm = torch.norm(a) * torch.norm(c_hat)
        scale_factor = 1 - torch.relu(ca_norm - 1) / ca_norm
        return scale_factor * c_hat
