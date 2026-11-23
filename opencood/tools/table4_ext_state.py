"""Atomic epoch-boundary state; used only by train_table4_ext, never Full DOMA.

Checkpoints are trusted local experiment artifacts (torch pickle). A failed
epoch is replayed, not resumed at an arbitrary batch. CUDA bitwise equivalence
is not promised. Dataset reinitialization and each rank's RNG/buffers are saved.
"""

import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def load(path):
    # Explicit for torch >=2.6, compatible with the historical torch version.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def fingerprint(directory):
    return hashlib.sha256((Path(directory) / "config.yaml").read_bytes()).hexdigest()


def save_epoch(directory, model, optimizer, scheduler, scaler, next_epoch,
               best_loss, best_epoch, loaders, dataset_rng, steps):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local = dict(rng=rng_state(), dataset_rng=dataset_rng,
                 generators=[x.generator.get_state() for x in loaders],
                 buffers={k: v.detach().cpu().clone() for k, v in model.named_buffers()})
    ranks = [None] * world
    if distributed:
        dist.all_gather_object(ranks, local)
    else:
        ranks[0] = local
    if rank == 0:
        best_path = Path(directory) / ("net_epoch_bestval_at%d.pth" % best_epoch)
        best_model = load(best_path) if best_path.exists() else None
        atomic_save(dict(schema=1, config_sha256=fingerprint(directory),
                         world_size=world, next_epoch=next_epoch,
                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict() if scaler is not None else None,
                         best_loss=best_loss, best_epoch=best_epoch, best_model=best_model,
                         ranks=ranks, cumulative_steps=steps),
                    Path(directory) / "resume_state.pth")
    if distributed:
        dist.barrier()


def restore_epoch(directory, model, optimizer, scheduler, scaler, loaders, dataset):
    path = Path(directory) / "resume_state.pth"
    if not path.exists():
        return None
    state = load(path)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if state["schema"] != 1 or state["world_size"] != world:
        raise RuntimeError("Resume schema/world size mismatch")
    if state["config_sha256"] != fingerprint(directory):
        raise RuntimeError("Config changed since saved training state")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (state["scaler"] is None) != (scaler is None):
        raise RuntimeError("AMP mode changed")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    local = state["ranks"][rank]
    with torch.no_grad():
        for key, buffer in model.named_buffers():
            buffer.copy_(local["buffers"][key])
    set_rng(local["dataset_rng"])
    dataset.reinitialize()
    for loader, generator_state in zip(loaders, local["generators"]):
        loader.generator.set_state(generator_state)
    set_rng(local["rng"])
    # Roll back bestval artifacts written during a subsequently interrupted epoch.
    # Preserve those files for audit instead of deleting evidence.
    if rank == 0:
        expected = "net_epoch_bestval_at%d.pth" % state["best_epoch"]
        for old in Path(directory).glob("net_epoch_bestval_at*.pth"):
            if old.name != expected:
                archive = Path(directory) / "interrupted_bestval"
                archive.mkdir(exist_ok=True)
                destination = archive / old.name
                index = 0
                while destination.exists():
                    index += 1
                    destination = archive / (old.name + ".%d" % index)
                os.replace(str(old), str(destination))
        if state["best_model"] is not None:
            atomic_save(state["best_model"], Path(directory) / expected)
    if distributed:
        dist.barrier()
    return state
