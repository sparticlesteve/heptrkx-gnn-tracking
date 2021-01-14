"""Dataset specification for hit graphs using pytorch_geometric formulation"""

# System imports
import os

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric

# Local imports
from utils.metadata import read_metadata

def load_graph(filename):
    with np.load(filename) as f:
        x, y = f['X'], f['y']
        Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
        Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
        n_edges = Ri_cols.shape[0]
        edge_index = np.zeros((2, n_edges), dtype=int)
        edge_index[0, Ro_cols] = Ro_rows
        edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None, real_weight=1.0):
        self.metadata = read_metadata(input_dir)
        if n_samples is not None:
            self.metadata = self.metadata.iloc[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = 1 #real_weight / (2 * real_weight - 1)

    def __getitem__(self, index):
        x, edge_index, y = load_graph(self.metadata.file.iloc[index])
        # Compute weights
        w = y * self.real_weight + (1-y) * self.fake_weight
        return torch_geometric.data.Data(x=torch.from_numpy(x),
                                         edge_index=torch.from_numpy(edge_index),
                                         y=torch.from_numpy(y), w=torch.from_numpy(w),
                                         i=index)

    def __len__(self):
        return len(self.metadata)

    def size(self):
        """We are using number of edges as sample size for now"""
        return self.metadata.n_edges.values

def get_datasets(n_train, n_valid, input_dir=None, real_weight=1.0):
    data = HitGraphDataset(input_dir=input_dir, n_samples=n_train+n_valid,
                           real_weight=real_weight)
    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data
