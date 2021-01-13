"""Metadata handing utilities"""

# System
import os
import logging

# Externals
import numpy as np
import pandas as pd

def _process_file(filename):
    """Prepare metadata for one data file"""
    with np.load(filename) as f:
        n_nodes = f['X'].shape[0]
        n_edges = f['y'].shape[0]
        purity = f['y'].mean()
    return dict(n_nodes=n_nodes, n_edges=n_edges, purity=purity)

def prepare_metadata(data_dir, n_files=None):
    """Prepare metadata dataframe for a data directory"""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.startswith('event') and not f.endswith('_ID.npz')]
    print('%i total files' % len(files))
    if n_files is not None:
        files = files[:n_files]
    return (pd.DataFrame.from_records([_process_file(f) for f in files])
            .assign(file=files))

def save_metadata(metadata, data_dir):
    """Write metadata to directory"""
    metadata.to_csv(os.path.join(data_dir, 'metadata.csv'))

def read_metadata(data_dir):
    """Read metadata from directory"""
    mdfile = os.path.expandvars(os.path.join(data_dir, 'metadata.csv'))
    logging.info('Reading metadata from %s', mdfile)
    return pd.read_csv(mdfile)
