"""Opt-in helpers for safer DDP validation and checkpoint writing."""

from numbers import Real

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError('num_replicas must be a positive integer')
        if not isinstance(rank, int) or rank < 0 or rank >= num_replicas:
            raise ValueError('rank must be in [0, num_replicas)')
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def ddp_safe_validation_enabled(hypes):
    """Return the explicitly configured DDP-safe validation flag."""
    train_params = hypes.get('train_params', {})
    config = train_params.get('ddp_safe_validation')
    if config is None:
        return False
    if not isinstance(config, dict):
        raise TypeError('train_params.ddp_safe_validation must be a mapping')

    unknown_keys = set(config) - {'enabled'}
    if unknown_keys:
        raise ValueError(
            'Unknown train_params.ddp_safe_validation keys: %s' %
            sorted(unknown_keys))

    enabled = config.get('enabled', False)
    if type(enabled) is not bool:
        raise TypeError(
            'train_params.ddp_safe_validation.enabled must be a boolean')
    return enabled


def checkpoint_writer_enabled(safe_enabled, distributed, rank):
    """Preserve legacy all-rank writes unless safe distributed mode is active."""
    return not (safe_enabled and distributed) or rank == 0


def sync_module_buffers_from_rank0(module):
    """Synchronize every registered module buffer from rank 0."""
    with torch.no_grad():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)


def all_ranks_have_valid_training_batch(local_valid, device,
                                        all_reduce=None):
    """Return true only when every distributed rank has a valid batch."""
    if type(local_valid) is not bool:
        raise TypeError('local_valid must be a boolean')

    validity = torch.tensor([int(local_valid)],
                            dtype=torch.int32,
                            device=device)
    if all_reduce is None:
        dist.all_reduce(validity, op=dist.ReduceOp.MIN)
    else:
        all_reduce(validity, op=dist.ReduceOp.MIN)
    return validity.item() == 1


def reduce_validation_sum_count(local_loss_sum, local_count, device,
                                all_reduce=None):
    """Reduce validation loss by global sum/count, not mean-of-rank-means."""
    if not isinstance(local_loss_sum, Real):
        raise TypeError('local_loss_sum must be a real number')
    if not isinstance(local_count, int) or isinstance(local_count, bool):
        raise TypeError('local_count must be an integer')
    if local_count < 0:
        raise ValueError('local_count must be non-negative')

    totals = torch.tensor([float(local_loss_sum), float(local_count)],
                          dtype=torch.float64,
                          device=device)
    if all_reduce is None:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    else:
        all_reduce(totals, op=dist.ReduceOp.SUM)

    global_loss_sum = totals[0].item()
    global_count = int(totals[1].item())
    if global_count <= 0:
        raise RuntimeError('Validation produced no usable batches on any rank')
    return global_loss_sum / global_count, global_loss_sum, global_count
