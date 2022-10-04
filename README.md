
# Implicit Flow Networks

This repository is the official implementation of Implicit Flow Networks, from the paper
"Physics-Informed Implicit Representations of Equilibrium Network Flows", to appear in NeurIPS 2022. 

## Command Line Usage

Use the following command to train and test IFN:

```commandline
python transductive_flow_estimation.py <dataset_name> <network_name> <model_name>
```

The script performs the following steps:
1. randomly splits the edges of the specified network into train and test edges,
2. trains a new IFN model on the train edges,
3. computes the RMSE loss from the test edges, and
4. saves all of the relevant data and results to a pickle file in ``/results/TransductiveFlowEstimation/``.

### Arguments

#### dataset_name
Name of the dataset from which to generate experiments. Currently implemented datasets are ``power`` (AC power flows) and ``water`` (municipal water network flows).

#### network_name
Name of a particular network within the dataset. Options for power are ``case57``, ``case118``, ``case145``, ``case300``, ``case_ACTIVSg200``, and ``case_ACTIVSg500``. Options for water are ``bellingham``, ``fairfield``, ``Net3``, ``northpenn``, and ``oberlin``.

#### model_name
IFN model to train and test. Options are ``ifn`` (IFN using edge weights directly from the dataset) and ``ifn+loglinweight`` (IFN with a log-linear model to learn edge weights from edge attributes).

#### Additional arguments
Hyperparameter settings and other options can be provided via optional flags:

| Flag                 | Description                                          | Default |
|----------------------|------------------------------------------------------|---------|
| ``--frac-labeled``   | Fraction of edges to include in training set         | 0.1     |
| ``--hidden-size``    | Hidden layer size for inverse flow function model    | 128     |
| ``--dmin``           | Min slope of the flow function (NOT its inverse)     | 0.5     |
| ``--dmax``           | Max slope of the flow function (NOT its inverse)     | 2.0     |
| ``--mitr``           | Max number of iterations for forward / backward pass | 100     |
| ``--eps``            | Convergence tolerance for forward / backward pass    | 1e-2    |
| ``--lr``             | Learn rate                                           | 1e-2    |
| ``--epochs``         | Max number of training epochs                        | 300     |
| ``--early-stop``     | Number of epochs for early stopping                  | 10      |
| ``--dev-name``       | PyTorch device                                       | "cuda"  |
| ``--seed``           | Random seed                                          | 1234    |
| ``--output-fname``   | Name of results file                                 | None    |

### Examples

```commandline
python transductive_flow_estimation.py water fairfield ifn --dmin 0.2 --dmax 20.0 --eps 1e-4 --epochs 2000
```

```commandline
python transductive_flow_estimation.py power case300 ifn+loglinweight --dmin 0.5 --dmax 2.0
```

## Reference

```
@inproceedings{smith2022ifn,
    author={Smith, Kevin D. and Seccamonte, Francesco and Swami, Ananthram and Bullo, Francesco},
    booktitle={Advances in Neural Information Processing Systems},
    title={Physics-Informed Implicit Representations of Equilibrium Network Flows},
    year={2022}
}
``` 
