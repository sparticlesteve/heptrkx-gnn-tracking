"""
PyTorch dataset specifications.
"""

# Externals
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

# Locals
from .sampler import DistributedBalancedBatchSampler

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     balanced_sampler=False, data_buckets=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'dummy':
        from .dummy import get_datasets
        train_dataset, valid_dataset = get_datasets(**data_args)
    elif name == 'hitgraphs':
        from . import hitgraphs
        train_dataset, valid_dataset = hitgraphs.get_datasets(**data_args)
        collate_fn = hitgraphs.collate_fn
    elif name == 'hitgraphs_sparse':
        from torch_geometric.data import Batch
        from . import hitgraphs_sparse
        train_dataset, valid_dataset = hitgraphs_sparse.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    else:
        raise Exception('Dataset %s unknown' % name)

    # Setup the distributed samplers
    train_sampler, train_batch_sampler, valid_sampler = None, None, None
    if distributed:
        # Balanced batch sampler
        if balanced_sampler:
            train_batch_sampler = DistributedBalancedBatchSampler(
                train_dataset, batch_size=batch_size, n_buckets=data_buckets,
                rank=rank, n_ranks=n_ranks)
        # Normal distributed sampler
        else:
            train_sampler = DistributedSampler(train_dataset, rank=rank,
                                               num_replicas=n_ranks)
        # Normal distributed sampler for validation dataset
        valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)

    # Construct the data loaders
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=collate_fn,
                                   num_workers=n_workers,
                                   sampler=train_sampler,
                                   batch_sampler=train_batch_sampler,
                                   shuffle=(not distributed))
    if valid_dataset is not None:
        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn,
                                       num_workers=n_workers,
                                       sampler=valid_sampler)
    else:
        valid_data_loader = None
    return train_data_loader, valid_data_loader
