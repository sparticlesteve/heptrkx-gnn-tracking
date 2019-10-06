"""
Common PyTorch trainer code.
"""

# System
import os
import re
import logging
import time
import math
from functools import partial

# Externals
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

# Locals
from models import get_model

class GNNBaseTrainer(object):
    """
    Base class for GNN PyTorch trainers.
    This implements the common training logic,
    model construction and distributed training setup,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed_mode=None, rank=0, n_ranks=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = 'cpu'
        self.distributed_mode = distributed_mode
        self.summaries = None
        self.summary_file = None
        self.rank = rank
        self.n_ranks = n_ranks

    def _build_optimizer(self, parameters, name='Adam', learning_rate=0.001,
                         lr_scaling=None, lr_warmup_epochs=0, lr_decay_schedule=[],
                         **optimizer_args):
        """Construct the training optimizer and scale learning rate.
        Should be called by build_model rather than called directly."""

        # Compute the scaled learning rate and corresponding initial warmup factor
        warmup_factor = 1
        if lr_scaling == 'linear':
            learning_rate = learning_rate * self.n_ranks
            warmup_factor = 1. / self.n_ranks
        elif lr_scaling == 'sqrt':
            learning_rate = learning_rate * math.sqrt(self.n_ranks)
            warmup_factor = 1. / math.sqrt(self.n_ranks)

        # Construct the optimizer
        OptimizerType = getattr(torch.optim, name)
        optimizer = OptimizerType(parameters, lr=learning_rate, **optimizer_args)

        # Distribute the optimizer if requested
        if self.distributed_mode == 'cray':
            from distributed.cray import distribute_optimizer
            optimizer = distribute_optimizer(optimizer)

        # Prepare the learning rate scheduler
        def _lr_schedule(epoch, warmup_factor=1, warmup_epochs=0, decays=[]):
            if epoch < warmup_epochs:
                return (1 - warmup_factor) * epoch / warmup_epochs + warmup_factor
            for decay in decays:
                if epoch >= decay['start_epoch'] and epoch < decay['end_epoch']:
                    return decay['factor']
            return 1
        lr_schedule = partial(_lr_schedule, warmup_factor=warmup_factor,
                              warmup_epochs=lr_warmup_epochs, decays=lr_decay_schedule)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
        return optimizer, lr_scheduler

    def build_model(self, name='gnn_sparse',
                    loss_func='binary_cross_entropy_with_logits',
                    optimizer_config={}, **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)

        # PyTorch distributed data parallel
        if self.distributed_mode in ['ddp-file', 'ddp-mpi']:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        # Construct the loss function
        self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer and learning rate scheduler
        self.optimizer, self.lr_scheduler = self._build_optimizer(
            self.model.parameters(), **optimizer_config)

    def save_summary(self, summaries):
        """Save summary information.
        This implementation is currently inefficient for simplicity:
            - we build a new DataFrame each time
            - we write the whole summary file each time
        """
        if self.summaries is None:
            self.summaries = pd.DataFrame([summaries])
        else:
            self.summaries = self.summaries.append([summaries], ignore_index=True)
        if self.output_dir is not None:
            summary_file = os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)
            self.summaries.to_csv(summary_file, index=False)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        # If using DistributedDataParallel, just save the wrapped model state
        model_state_dict = (self.model.module.state_dict()
                            if self.distributed_mode in ['ddp-file', 'ddp-mpi']
                            else self.model.state_dict())
        checkpoint = dict(checkpoint_id=checkpoint_id,
                          model=model_state_dict,
                          optimizer=self.optimizer.state_dict(),
                          lr_scheduler=self.lr_scheduler.state_dict())
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_id=-1):
        """Load a model checkpoint"""
        assert self.output_dir is not None

        # Load the summaries
        summary_file = os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)
        if not os.path.exists(summary_file):
            logging.info('Summary file does not exist. Will not load checkpoint')
            return
        logging.info('Reloading summary at %s', summary_file)
        self.summaries = pd.read_csv(summary_file)

        # Load the checkpoint
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if checkpoint_id == -1:
            checkpoint_id = self.summaries.epoch.iloc[-1]
        checkpoint_file = os.path.join(
            checkpoint_dir, 'model_checkpoint_%03i.pth.tar' % checkpoint_id)
        logging.info('Reloading checkpoint at %s', checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        # If using DistributedDataParallel, just load the wrapped model state
        if self.distributed_mode in ['ddp-file', 'ddp-mpi']:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel() for p in self.model.parameters())))

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def predict(self, data_loader):
        """Virtual method to apply a model"""
        raise NotImplementedError

    def train(self, train_data_loader, valid_data_loader=None,
              n_epochs=-1, n_total_epochs=0):
        """Run the model training"""

        # Determine initial epoch in case resuming training
        start_epoch = 0
        if self.summaries is not None:
            start_epoch = self.summaries.epoch.max() + 1

        # Determine how many epochs we run in this call
        end_epoch = n_total_epochs
        if n_epochs >= 0:
            end_epoch = min(start_epoch+n_epochs, n_total_epochs)

        # Loop over epochs
        for epoch in range(start_epoch, end_epoch):
            self.logger.info('Epoch %i' % epoch)
            try:
                train_data_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            # Train on this epoch
            start_time = time.time()
            summary = self.train_epoch(train_data_loader)
            self.lr_scheduler.step()
            summary['epoch'] = epoch
            summary['train_time'] = time.time() - start_time

            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader))
                summary['valid_time'] = time.time() - start_time

            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None and self.rank == 0:
                self.write_checkpoint(checkpoint_id=epoch)
            
        return self.summaries
