"""
Python module for holding our PyTorch models.
"""

import importlib

def get_model(name, **model_args):
    """Top-level factory function for getting your models"""
    module = importlib.import_module('.' + name, 'models')
    return module.build_model(**model_args)
