"""Real DAIR two-step smoke test. Requires installed encoders and actual data.

No mock modules, no full training, no persistent experiment checkpoint. Supports
torchrun; reported short validation loss is diagnostic, never formal bestval/AP.
"""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from opencood.tools.dair_protocol import (
    assert_shared_equal, build_model, inherit_state, load_state, require, shared_keys,
    validate_protocol, raw_config, digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage1-checkpoint')
    parser.add_argument('--stage1-config')
    parser.add_argument('--num-workers', type=int, default=0)
    opt = parser.parse_args()
    from opencood.tools import train_utils, seed_utils, ddp_safe_utils, multi_gpu_utils
    from opencood.data_utils.datasets import build_dataset
    cfg = validate_protocol(raw_config(opt.config))
    require(cfg['dair_protocol']['stage'] in ('stage1', 'stage2'), 'Training-stage smoke only')
    distributed = SimpleNamespace(dist_url='env://')
    multi_gpu_utils.init_distributed_mode(distributed)
    temporary = None
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        seed = seed_utils.seed_from_hypes(cfg)
        cfg, model = build_model(opt.config)
        stage2 = cfg['dair_protocol']['stage'] == 'stage2'
        source_record = None
        if stage2:
            require(opt.stage1_checkpoint and opt.stage1_config, 'Stage2 smoke requires explicit Stage1 sources')
            source_cfg = validate_protocol(raw_config(opt.stage1_config), 'stage1')
            require(source_cfg['dair_protocol']['method'] == cfg['dair_protocol']['method'], 'Wrong source method')
            if cfg['dair_protocol']['method'] == 'DOMA':
                from opencood.tools.doma_tools import doma_method_fingerprint
                require(doma_method_fingerprint(source_cfg['model']['args']['doma']) ==
                        doma_method_fingerprint(cfg['model']['args']['doma']), 'Source DOMA method differs')
            source = load_state(opt.stage1_checkpoint)
            source_record = dict(checkpoint=str(Path(opt.stage1_checkpoint).resolve()),
                                 checkpoint_sha256=digest(opt.stage1_checkpoint),
                                 config=str(Path(opt.stage1_config).resolve()),
                                 config_sha256=digest(opt.stage1_config))
            _, source_model = build_model(opt.stage1_config)
            source_model.load_state_dict(source, strict=True)
            del source_model
            inherit_state(model, source)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        plain = model
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[distributed.gpu],
                                                               find_unused_parameters=True)
        optimizer = train_utils.setup_optimizer(cfg, plain)
        criterion = train_utils.create_loss(cfg)
        train = build_dataset(cfg, visualize=False, train=True)
        val = build_dataset(cfg, visualize=False, train=False)
        require(len(train) == 4811 and len(val) == 1789, 'Unexpected dataset sizes')
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=seed)
        sampler.set_epoch(1 if stage2 else 0)
        loader = DataLoader(train, batch_size=2, sampler=sampler, drop_last=True,
                            num_workers=opt.num_workers, collate_fn=train.collate_batch_train,
                            **seed_utils.dataloader_seed_kwargs(seed, rank))
        model.train()
        plain.model_train_init()
        frozen = {k: v.detach().cpu().clone() for k, v in plain.state_dict().items()
                  if stage2 and k in shared_keys(plain.state_dict())}
        initial = {k: p.detach().cpu().clone() for k, p in plain.named_parameters() if p.requires_grad}
        steps, gradients, agents = 0, set(), []
        for index, batch in enumerate(loader):
            if index >= 32:
                break
            valid = batch is not None and bool(batch['ego']['object_bbx_mask'].sum())
            if dist.is_initialized():
                valid = ddp_safe_utils.all_ranks_have_valid_training_batch(valid, device)
            if not valid:
                continue
            batch = train_utils.to_device(batch, device)
            batch['ego']['epoch'] = 1 if stage2 else 0
            optimizer.zero_grad()
            output = model(batch['ego'])
            loss = criterion(output, batch['ego']['label_dict'])
            if getattr(train, 'supervise_single', False):
                loss = loss + criterion(output, batch['ego']['label_dict_single'], suffix='_single') * cfg['train_params'].get('single_weight', 1)
            require(torch.isfinite(loss).item(), 'Non-finite training loss')
            loss.backward()
            for name, p in plain.named_parameters():
                if p.grad is not None:
                    require(torch.isfinite(p.grad).all().item(), 'Non-finite gradient: ' + name)
                    gradients.add(name)
            optimizer.step()
            agents.extend(batch['ego'].get('record_len', torch.tensor([1])).detach().cpu().tolist())
            steps += 1
            if steps == 2:
                break
        require(steps == 2, 'Could not execute two synchronized optimizer steps')
        state = plain.state_dict()
        require(all(torch.equal(v, state[k].cpu()) for k, v in frozen.items()), 'Frozen parameter/buffer changed')
        changed = [k for k, p in plain.named_parameters() if k in initial and not torch.equal(initial[k], p.cpu())]
        require(any(not k.startswith('doma_') for k in changed), 'No Base trainable parameter updated')
        if cfg['dair_protocol']['method'] == 'DOMA':
            prefix = 'doma_object_adapter_m2.' if stage2 else 'doma_shared_object_refiner.'
            require(any(k.startswith(prefix) for k in gradients), 'Expected DOMA module has no gradient')
            require(any(k.startswith(prefix) for k in changed), 'Expected DOMA module did not update')
        model.eval()
        if dist.is_initialized():
            ddp_safe_utils.sync_module_buffers_from_rank0(plain)
        vs = ddp_safe_utils.DistributedEvalSampler(val, world, rank)
        vl = DataLoader(val, sampler=vs, batch_size=1, num_workers=opt.num_workers,
                        collate_fn=val.collate_batch_train)
        total, count = 0., 0
        with torch.no_grad():
            for i, batch in enumerate(vl):
                if i >= 2:
                    break
                if batch is None:
                    continue
                batch = train_utils.to_device(batch, device)
                output = plain(batch['ego'])
                loss = criterion(output, batch['ego']['label_dict'])
                require(torch.isfinite(loss).item(), 'Non-finite validation loss')
                total += loss.item()
                count += 1
        if dist.is_initialized():
            mean, _, count = ddp_safe_utils.reduce_validation_sum_count(total, count, device)
        else:
            require(count > 0, 'No validation batches')
            mean = total / count
        location = [None]
        if rank == 0:
            temporary = tempfile.TemporaryDirectory(prefix='dair-smoke-')
            location[0] = str(Path(temporary.name) / 'smoke_only.pth')
            torch.save(plain.state_dict(), location[0])
        if dist.is_initialized():
            dist.broadcast_object_list(location, src=0)
        restored = load_state(location[0])
        require(all(torch.equal(v.cpu(), restored[k]) for k, v in plain.state_dict().items()),
                'Rank checkpoint differs from current parameters/buffers')
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(json.dumps(dict(status='REAL_DATA_SMOKE_PASS', world_size=world, optimizer_steps=steps,
                  global_batch=2*world, validation_shards=[len(ddp_safe_utils.DistributedEvalSampler(val,world,r)) for r in range(world)],
                  short_validation_batches=count, short_validation_loss=mean, rank0_agent_counts=agents,
                  stage1_source=source_record,
                  updated_parameters=len(changed), checkpoint='temporary rank0 save/reload; deleted; not a trained Stage1'), indent=2))
    finally:
        if temporary is not None:
            temporary.cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
