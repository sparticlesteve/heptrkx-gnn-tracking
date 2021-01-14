"""This module contains the implementatin of our balanced batch sampler"""

# System
import logging

# Externals
import numpy as np
import torch

class DistributedBalancedBatchSampler(torch.utils.data.Sampler):
    """
    A balanced batch sampler for distributed GNN training.

    This sampler requires the dataset class to define a size() method which
    returns a numpy array of sample sizes. Sample size must be a single number.
    """

    def __init__(self, dataset, batch_size, n_buckets, rank=0, n_ranks=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = -1
        self.rank = rank
        self.n_ranks = n_ranks

        # Determine sample sizes, allowing for Subset wrapper
        if type(dataset) == torch.utils.data.dataset.Subset:
            self.sample_sizes = dataset.dataset.size()[dataset.indices]
        else:
            self.sample_sizes = dataset.size()

        # Divisibility checks
        assert (len(dataset) % n_buckets) == 0, (
            'Dataset size not divisible by buckets')
        bucket_size = len(dataset) // n_buckets
        assert (bucket_size % (batch_size*n_ranks)) == 0, (
            'Bucket size not divisible by global batch size')

        if rank == 0:
            logging.info('Setting up balanced batch sampler ' +
                         'with %i data buckets of size %i',
                         n_buckets, bucket_size)
        # Make buckets from sorted sample indices
        self.buckets = self.sample_sizes.argsort().reshape(-1, bucket_size)

    def __iter__(self):

        # Increment epoch
        self.epoch += 1

        # Deterministic shuffling based on epoch
        g = np.random.default_rng(self.epoch)

        # Shuffle samples within each bucket
        for bucket in self.buckets:
            g.shuffle(bucket)

        # Form batches - here I copy to avoid repeating sort + bucket
        global_batch_size = self.batch_size * self.n_ranks
        batches = self.buckets.copy().reshape(-1, global_batch_size)

        # Shuffle the batches
        g.shuffle(batches)

        # Loop and yield batch indices for this rank
        for batch in batches:
            yield batch[self.rank:global_batch_size:self.n_ranks]

    def __len__(self):
        return len(self.dataset) // self.batch_size
