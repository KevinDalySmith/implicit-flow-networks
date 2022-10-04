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

# transductive_flow_estimation.py
# Code to reproduce results from "Physics-informed implicit representations of equilibrium network flows",
# to appear in NeurIPS 2022 (paper 7508).

import argparse
import pickle
import time
from os.path import join
from experiments.TransductiveFlowEstimation import TransductiveFlowEstimation
from datasets import REGISTERED_DATASETS


RESULTS_PATH = './results/TransductiveFlowEstimation/'


def main():

    timestr = time.strftime('%Y%m%d-%H%M%S')

    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument('dataset_name', choices=list(REGISTERED_DATASETS.keys()))
    parser.add_argument('network_name', type=str)
    parser.add_argument('model_name', choices=list(TransductiveFlowEstimation.TRAIN_TEST_FUNCTIONS.keys()))
    parser.add_argument('--frac-labeled', type=float, default=0.1, help='Fraction of edges to include in training set')

    # Options
    parser.add_argument('--dev-name', type=str, default='cuda', help='Name of PyTorch device (e.g., cpu or cuda)')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output-fname', type=str, default=None, help='Name of results file')

    # IFN settings
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--dmin', type=float, default=0.5, help='Min slope of the flow function (NOT its inverse)')
    parser.add_argument('--dmax', type=float, default=2.0, help='Max slope of the flow function (NOT its inverse)')
    parser.add_argument('--mitr', type=int, default=100,
                        help='Max number of iterations for forward / backward pass')
    parser.add_argument('--eps', type=float, default=1e-2, help='Convergence tolerance for forward / backward pass')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learn rate for IFN')
    parser.add_argument('--epochs', type=int, default=300, help='Max number of training epochs')
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Number of epochs for early stopping')

    args = parser.parse_args()

    experiment = TransductiveFlowEstimation.run(
        dataset_name=args.dataset_name,
        network_name=args.network_name,
        model_name=args.model_name,
        model_params=vars(args),
        seed=args.seed,
        frac_labeled=args.frac_labeled,
        dev_name=args.dev_name)

    # Save results
    if args.output_fname is None:
        fname = '-'.join([args.dataset_name, args.network_name, args.model_name, timestr]) + '.pickle'
    else:
        fname = args.output_fname
    results_fname = join(RESULTS_PATH, fname)
    with open(results_fname, 'wb') as outfile:
        pickle.dump(experiment, outfile)
    print(f'Final test RMSE: {experiment.test_rmse}')
    print(f'Results saved to {results_fname}')


if __name__ == '__main__':
    main()
