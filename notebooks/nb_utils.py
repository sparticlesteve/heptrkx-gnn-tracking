"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple

# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import Subset, DataLoader

# Locals
from models import get_model
import datasets.hitgraphs
from torch_geometric.data import Batch
from datasets.hitgraphs_sparse import HitGraphDataset

def get_output_dir(config):
    return os.path.expandvars(config['output_dir'])

def get_input_dir(config):
    return os.path.expandvars(config['data']['input_dir'])

def load_config_file(config_file):
    """Load configuration from a specified yaml config file path"""
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_config_dir(result_dir):
    """Load pickled config saved in a result directory"""
    config_file = os.path.join(result_dir, 'config.pkl')
    with open(config_file, 'rb') as f:
        return pickle.load(f)

# Back-compat
load_config = load_config_file

def load_summaries(config):
    summary_file = os.path.join(get_output_dir(config), 'summaries_0.csv')
    return pd.read_csv(summary_file)

def load_model(config, reload_epoch):
    """deprecated"""
    model_config = config['model']
    model_config.pop('loss_func', None)
    model = get_model(**model_config)

    # Reload specified model checkpoint
    output_dir = get_output_dir(config)
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model

def get_dataset(config):
    return HitGraphDataset(get_input_dir(config))

def get_test_data_loader(config, n_test=16, batch_size=1):
    # Take the test set from the back
    full_dataset = get_dataset(config)
    test_indices = len(full_dataset) - 1 - torch.arange(n_test)
    test_dataset = Subset(full_dataset, test_indices.tolist())
    return DataLoader(test_dataset, batch_size=batch_size,
                      collate_fn=Batch.from_data_list)

def get_dense_dataset(config):
    return datasets.hitgraphs.HitGraphDataset(get_input_dir(config))

def get_dense_test_data_loader(config, n_test=16):
    # Take the test set from the back
    full_dataset = get_dense_dataset(config)
    test_indices = len(full_dataset) - 1 - torch.arange(n_test)
    test_dataset = Subset(full_dataset, test_indices.tolist())
    return DataLoader(test_dataset, batch_size=1,
                      collate_fn=datasets.hitgraphs.collate_fn)

@torch.no_grad()
def apply_model(model, data_loader):
    preds, targets = [], []
    for batch in data_loader:
        preds.append(torch.sigmoid(model(batch)).squeeze(0))
        targets.append(batch.y.squeeze(0))
    return preds, targets

@torch.no_grad()
def apply_dense_model(model, data_loader):
    preds, targets = [], []
    for inputs, target in data_loader:
        preds.append(model(inputs).squeeze(0))
        targets.append(target.squeeze(0))
    return preds, targets

# Define our Metrics class as a namedtuple
Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall',
                                 'prc_precision', 'prc_recall', 'prc_thresh',
                                 'roc_fpr', 'roc_tpr', 'roc_thresh', 'roc_auc'])

def compute_metrics(preds, targets, threshold=0.5):
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # Decision boundary metrics
    y_pred, y_true = (preds > threshold), (targets > threshold)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    # Precision recall curves
    prc_precision, prc_recall, prc_thresh = sklearn.metrics.precision_recall_curve(y_true, preds)
    # ROC curve
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, preds)
    roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
    # Organize metrics into a namedtuple
    return Metrics(accuracy=accuracy, precision=precision, recall=recall,
                   prc_precision=prc_precision, prc_recall=prc_recall, prc_thresh=prc_thresh,
                   roc_fpr=roc_fpr, roc_tpr=roc_tpr, roc_thresh=roc_thresh, roc_auc=roc_auc)

def plot_train_history(summaries, figsize=(12, 10), loss_yscale='linear'):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axs = axs.flatten()

    # Plot losses
    axs[0].plot(summaries.epoch, summaries.train_loss, label='Train')
    axs[0].plot(summaries.epoch, summaries.valid_loss, label='Validation')
    axs[0].set_yscale(loss_yscale)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc=0)

    # Plot accuracies
    axs[1].plot(summaries.epoch, summaries.valid_acc, label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].legend(loc=0)

    # Plot model weight norm
    axs[2].plot(summaries.epoch, summaries.l2)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Model L2 weight norm')

    # Plot learning rate
    axs[3].plot(summaries.epoch, summaries.lr)
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Learning rate')

    plt.tight_layout()

def plot_metrics(preds, targets, metrics):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16,5))

    # Plot the model outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==False], label='fake', **binning)
    ax0.hist(preds[labels==True], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot precision and recall
    ax1.plot(metrics.prc_thresh, metrics.prc_precision[:-1], label='purity')
    ax1.plot(metrics.prc_thresh, metrics.prc_recall[:-1], label='efficiency')
    ax1.set_xlabel('Model threshold')
    ax1.legend(loc=0)

    # Plot the ROC curve
    ax2.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel('False positive rate')
    ax2.set_ylabel('True positive rate')
    ax2.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)

    plt.tight_layout()

def plot_outputs_roc(preds, targets, metrics):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,5))

    # Plot the model outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==False], label='fake', **binning)
    ax0.hist(preds[labels==True], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot the ROC curve
    ax1.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)
    plt.tight_layout()

def draw_sample(X, Ri, Ro, y, cmap='bwr_r', alpha_labels=True, figsize=(15, 7)):
    # Select the i/o node features for each segment
    feats_o = X[np.where(Ri.T)[1]]
    feats_i = X[np.where(Ro.T)[1]]

    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap(cmap)

    # Draw the hits (r, phi, z)
    ax0.scatter(X[:,2], X[:,0], c='k')
    ax1.scatter(X[:,1], X[:,0], c='k')

    # Draw the segments
    for j in range(y.shape[0]):
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,2], feats_i[j,2]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
        ax1.plot([feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
    # Adjust axes
    ax0.set_xlabel('$z$')
    ax1.set_xlabel('$\phi$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()

def draw_sample_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    x = hits[:,0] * np.cos(hits[:,1])
    y = hits[:,0] * np.sin(hits[:,1])
    fig, ax0 = plt.subplots(figsize=figsize)

    # Draw the hits
    ax0.scatter(x, y, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='k', alpha=preds[j])

    return fig, ax0