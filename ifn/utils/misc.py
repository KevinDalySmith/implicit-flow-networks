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

import numpy as np


def num2index(x, y):
    """
    Replace integer names with indices.
    I.e., get the indices of where elements in x occur in y.

    Parameters
    ----------
    x : np.ndarray
        Array of ints. Elements should be a subset of y.
    y : np.ndarray
        Array of ints. Each element should be unique.

    Returns
    -------
    x_idx : np.ndarray
        Array of ints, where x_idx[i] is the index such that x[i] = y[x_idx[i]].
    """
    index = np.argsort(y)
    x_pos = np.searchsorted(y[index], x)
    return index[x_pos]
