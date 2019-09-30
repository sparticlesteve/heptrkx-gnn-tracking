"""Utility functions for building models"""

import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def make_mlp(input_size, sizes,
             hidden_activation=nn.ReLU,
             output_activation=nn.ReLU,
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.apply(init_weights)
    return net
