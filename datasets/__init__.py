from . import MatpowerDataset, WNTRDataset
from .MatpowerDataset import MatpowerDataset
from .WNTRDataset import WNTRDataset

# List of registered datasets.
# Dictionary maps the dataset name to a tuple with the following elements:
#   - Subdirectory of data/ containing the raw and processed data
#   - Module in ifn/datasets/ implementing the dataset
REGISTERED_DATASETS = {
    'power': ('matpower', MatpowerDataset),
    'water': ('wntr', WNTRDataset)
}
