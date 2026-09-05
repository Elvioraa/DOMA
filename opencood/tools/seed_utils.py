"""YAML-controlled random-number seeding for experiment entry points.

No fixed seed is synthesized when ``seed`` is absent from the loaded YAML.
"""

import hashlib
import random

import numpy as np
import torch


def get_seed(hypes):
    """Return the explicit YAML seed, or ``None`` when it is not configured."""
    if "seed" not in hypes:
        return None

    seed = hypes["seed"]
    if type(seed) is not int:
        raise TypeError("YAML seed must be an integer")
    if seed < 0 or seed >= 2 ** 32:
        raise ValueError("YAML seed must be in the range [0, 2**32)")
    return seed


def seed_everything(seed):
    """Seed Python, NumPy, Torch, and all CUDA devices when requested."""
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_from_hypes(hypes):
    """Apply the optional top-level YAML ``seed`` and return its value."""
    seed = get_seed(hypes)
    seed_everything(seed)
    return seed


def derive_seed(seed, namespace):
    """Derive a stable named stream from an explicit YAML seed."""
    get_seed({"seed": seed})
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("seed namespace must be a non-empty string")
    payload = "{}:{}".format(seed, namespace).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_worker(_worker_id):
    """Seed each DataLoader worker from its Torch-assigned worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def dataloader_seed_kwargs(seed, rank=0):
    """Return deterministic DataLoader kwargs only for an explicit seed."""
    if seed is None:
        return {}
    if type(rank) is not int or rank < 0:
        raise ValueError("DataLoader rank must be a non-negative integer")

    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return {
        "generator": generator,
        "worker_init_fn": seed_worker,
    }


def distributed_sampler_seed_kwargs(seed):
    """Return sampler kwargs only when YAML provides an explicit seed."""
    if seed is None:
        return {}
    return {"seed": seed}
