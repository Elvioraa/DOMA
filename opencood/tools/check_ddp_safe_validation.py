"""CPU/mock checks for the opt-in DDP-safe validation path."""

import ast
import inspect
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from opencood.tools import ddp_safe_utils


class _IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class _OptionalLossDataset(Dataset):
    def __init__(self):
        self.losses = [1.0, None, 3.0]

    def __len__(self):
        return len(self.losses)

    def __getitem__(self, index):
        return self.losses[index]


class _BufferSyncToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(2)
        self.register_buffer('other_buffer', torch.zeros(2))

    def forward(self, inputs):
        return self.batch_norm(inputs) + self.other_buffer


def _find_free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def _gloo_buffer_sync_worker(rank, world_size, init_method):
    dist.init_process_group(
        backend='gloo',
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30))
    try:
        ddp_model = DistributedDataParallel(_BufferSyncToyModule())
        with torch.no_grad():
            if rank == 0:
                ddp_model.module.batch_norm.running_mean.copy_(
                    torch.tensor([1.0, 2.0]))
                ddp_model.module.batch_norm.running_var.copy_(
                    torch.tensor([3.0, 4.0]))
                ddp_model.module.batch_norm.num_batches_tracked.fill_(5)
                ddp_model.module.other_buffer.copy_(torch.tensor([5.0, 6.0]))
            else:
                ddp_model.module.batch_norm.running_mean.copy_(
                    torch.tensor([-1.0, -2.0]))
                ddp_model.module.batch_norm.running_var.copy_(
                    torch.tensor([7.0, 8.0]))
                ddp_model.module.batch_norm.num_batches_tracked.fill_(9)
                ddp_model.module.other_buffer.copy_(torch.tensor([-5.0, -6.0]))

            ddp_model.module.batch_norm.weight.fill_(10.0 + rank)
        ddp_model.module.batch_norm.weight.grad = torch.full_like(
            ddp_model.module.batch_norm.weight, 20.0 + rank)
        parameter_before = ddp_model.module.batch_norm.weight.detach().clone()
        gradient_before = ddp_model.module.batch_norm.weight.grad.clone()

        ddp_model.eval()
        assert not ddp_model.training
        assert not ddp_model.module.training

        ddp_safe_utils.sync_module_buffers_from_rank0(ddp_model.module)

        assert torch.equal(
            ddp_model.module.batch_norm.running_mean,
            torch.tensor([1.0, 2.0]))
        assert torch.equal(
            ddp_model.module.batch_norm.running_var,
            torch.tensor([3.0, 4.0]))
        assert ddp_model.module.batch_norm.num_batches_tracked.item() == 5
        assert torch.equal(
            ddp_model.module.other_buffer,
            torch.tensor([5.0, 6.0]))
        assert torch.equal(
            ddp_model.module.batch_norm.weight,
            parameter_before)
        assert torch.equal(
            ddp_model.module.batch_norm.weight.grad,
            gradient_before)

        wrapper_forward_called = [False]

        def mark_wrapper_forward(_module, _inputs):
            wrapper_forward_called[0] = True

        hook = ddp_model.register_forward_pre_hook(mark_wrapper_forward)
        try:
            output = ddp_model.module(torch.tensor([[1.0, 2.0]]))
        finally:
            hook.remove()
        assert output.shape == (1, 2)
        assert not wrapper_forward_called[0]

        training_model = DistributedDataParallel(nn.Linear(1, 1, bias=False))
        optimizer = torch.optim.SGD(training_model.parameters(), lr=0.1)
        local_validity = (
            [True, True, False, False, True]
            if rank == 0 else
            [True, False, True, False, True]
        )
        expected_decisions = [True, False, False, False, True]
        decisions = []
        optimizer_step_count = 0

        for local_valid in local_validity:
            train_batch_valid = \
                ddp_safe_utils.all_ranks_have_valid_training_batch(
                    local_valid, torch.device('cpu'))
            decisions.append(train_batch_valid)
            if not train_batch_valid:
                continue

            optimizer.zero_grad()
            inputs = torch.tensor([[float(rank + 1)]])
            targets = torch.tensor([[1.0]])
            loss = (training_model(inputs) - targets).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer_step_count += 1

        assert decisions == expected_decisions
        assert optimizer_step_count == 2

        parameters = training_model.module.weight.detach()
        gathered_parameters = [torch.zeros_like(parameters)
                               for _ in range(world_size)]
        dist.all_gather(gathered_parameters, parameters)
        assert all(torch.equal(gathered_parameters[0], parameter)
                   for parameter in gathered_parameters[1:])
    finally:
        dist.destroy_process_group()


def _assert_config_parsing():
    assert not ddp_safe_utils.ddp_safe_validation_enabled({})
    assert not ddp_safe_utils.ddp_safe_validation_enabled(
        {'train_params': {}})
    assert not ddp_safe_utils.ddp_safe_validation_enabled(
        {'train_params': {'ddp_safe_validation': {'enabled': False}}})
    assert ddp_safe_utils.ddp_safe_validation_enabled(
        {'train_params': {'ddp_safe_validation': {'enabled': True}}})
    parsed = yaml.safe_load(
        'train_params:\n'
        '  ddp_safe_validation:\n'
        '    enabled: true\n')
    assert ddp_safe_utils.ddp_safe_validation_enabled(parsed)

    for invalid in ('true', 1, None):
        if invalid is None:
            continue
        try:
            ddp_safe_utils.ddp_safe_validation_enabled({
                'train_params': {
                    'ddp_safe_validation': {'enabled': invalid}
                }
            })
        except TypeError:
            pass
        else:
            raise AssertionError('non-boolean enabled value was accepted')


def _mock_reducer(peer_loss_sum, peer_count):
    def all_reduce(totals, op):
        assert op == torch.distributed.ReduceOp.SUM
        totals += torch.tensor([peer_loss_sum, peer_count],
                               dtype=totals.dtype,
                               device=totals.device)
    return all_reduce


def _mock_min_reducer(peer_valid):
    def all_reduce(validity, op):
        assert op == torch.distributed.ReduceOp.MIN
        peer = torch.tensor([int(peer_valid)],
                            dtype=validity.dtype,
                            device=validity.device)
        validity.copy_(torch.minimum(validity, peer))
    return all_reduce


def _assert_training_batch_validity():
    cases = (
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    )
    for local_valid, peer_valid, expected in cases:
        actual = ddp_safe_utils.all_ranks_have_valid_training_batch(
            local_valid,
            torch.device('cpu'),
            _mock_min_reducer(peer_valid))
        assert actual is expected


def _assert_global_sum_count():
    rank0_loss, rank0_sum, rank0_count = \
        ddp_safe_utils.reduce_validation_sum_count(
            10, 4, torch.device('cpu'), _mock_reducer(20, 6))
    rank1_loss, rank1_sum, rank1_count = \
        ddp_safe_utils.reduce_validation_sum_count(
            20, 6, torch.device('cpu'), _mock_reducer(10, 4))

    assert rank0_loss == rank1_loss == 3.0
    assert rank0_sum == rank1_sum == 30.0
    assert rank0_count == rank1_count == 10
    naive_rank_mean = ((10 / 4) + (20 / 6)) / 2
    assert naive_rank_mean != rank0_loss

    try:
        ddp_safe_utils.reduce_validation_sum_count(
            0, 0, torch.device('cpu'), _mock_reducer(0, 0))
    except RuntimeError:
        pass
    else:
        raise AssertionError('empty global validation set was accepted')


def _assert_nonpadding_validation_shards():
    dataset = _IndexDataset(5)
    rank0 = list(ddp_safe_utils.DistributedEvalSampler(
        dataset, num_replicas=2, rank=0))
    rank1 = list(ddp_safe_utils.DistributedEvalSampler(
        dataset, num_replicas=2, rank=1))
    assert rank0 == [0, 2, 4]
    assert rank1 == [1, 3]
    assert sorted(rank0 + rank1) == list(range(len(dataset)))
    assert set(rank0).isdisjoint(rank1)


def _assert_dataloader_preserves_tail_batch():
    dataset = _IndexDataset(5)
    rank_loaders = []
    for rank in range(2):
        sampler = ddp_safe_utils.DistributedEvalSampler(
            dataset, num_replicas=2, rank=rank)
        rank_loaders.append(DataLoader(
            dataset,
            sampler=sampler,
            batch_size=2,
            drop_last=False))

    rank_batches = [
        [batch.tolist() for batch in loader]
        for loader in rank_loaders
    ]
    rank_indices = [
        [index for batch in batches for index in batch]
        for batches in rank_batches
    ]
    combined_indices = rank_indices[0] + rank_indices[1]

    assert rank_batches[0] == [[0, 2], [4]]
    assert rank_batches[1] == [[1, 3]]
    assert rank_batches[0][-1] == [4]
    assert sorted(combined_indices) == list(range(len(dataset)))
    assert len(combined_indices) == len(set(combined_indices))


def _assert_none_batch_is_not_counted():
    def collate_optional_loss(batch):
        assert len(batch) == 1
        return batch[0]

    loader = DataLoader(
        _OptionalLossDataset(),
        batch_size=1,
        drop_last=False,
        collate_fn=collate_optional_loss)

    local_loss_sum = 0.0
    local_count = 0
    for batch_loss in loader:
        if batch_loss is None:
            continue
        local_loss_sum += batch_loss
        local_count += 1

    assert local_loss_sum == 4.0
    assert local_count == 2

    global_loss, global_loss_sum, global_count = \
        ddp_safe_utils.reduce_validation_sum_count(
            local_loss_sum,
            local_count,
            torch.device('cpu'),
            _mock_reducer(6.0, 2))
    assert global_loss_sum == 10.0
    assert global_count == 4
    assert global_loss == 2.5


def _assert_gloo_buffer_sync_and_eval_mode():
    world_size = 2
    port = _find_free_tcp_port()
    init_method = 'tcp://127.0.0.1:%d' % port
    mp.spawn(
        _gloo_buffer_sync_worker,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True)


def _assert_checkpoint_writer():
    writer = ddp_safe_utils.checkpoint_writer_enabled
    safe_writers = [rank for rank in range(2)
                    if writer(True, True, rank)]
    legacy_writers = [rank for rank in range(2)
                      if writer(False, True, rank)]
    assert safe_writers == [0]
    assert legacy_writers == [0, 1]
    assert writer(True, False, 1)


def _assert_training_log_accounting():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    counts = {'epoch': 0, 'cumulative': 0}

    def record_optimizer_step(_optimizer, _args, _kwargs):
        counts['epoch'] += 1
        counts['cumulative'] += 1

    hook = optimizer.register_step_post_hook(record_optimizer_step)
    try:
        epoch_results = []
        for validity in ([True, False, True, False, True],
                         [True, False, True, True]):
            loader_iterations = 0
            skipped_iterations = 0
            counts['epoch'] = 0
            for train_batch_valid in validity:
                loader_iterations += 1
                if not train_batch_valid:
                    skipped_iterations += 1
                    continue
                optimizer.zero_grad()
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
            epoch_results.append((loader_iterations,
                                  skipped_iterations,
                                  counts['epoch'],
                                  counts['cumulative']))
        assert epoch_results == [(5, 2, 3, 3), (4, 1, 3, 6)]

        counts['epoch'] = 0
        scaler = torch.amp.GradScaler('cpu')
        optimizer.zero_grad()
        overflow_loss = (parameter * torch.tensor(float('inf'))).sum()
        scaler.scale(overflow_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        assert counts['epoch'] == 0
        assert counts['cumulative'] == 6

        optimizer.zero_grad()
        amp_loss = parameter.square().sum()
        scaler.scale(amp_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        assert counts['epoch'] == 1
        assert counts['cumulative'] == 7
    finally:
        hook.remove()


def _assert_optimizer_group_labels():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    train_ddp_path = os.path.join(tools_dir, 'train_ddp.py')
    with open(train_ddp_path, 'r', encoding='utf-8') as source_file:
        syntax_tree = ast.parse(source_file.read())
    label_function = next(
        node for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and
        node.name == '_optimizer_group_labels')
    namespace = {}
    exec(compile(ast.Module(body=[label_function], type_ignores=[]),
                 train_ddp_path, 'exec'), namespace)
    identify = namespace['_optimizer_group_labels']

    class OptimizerGroupToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.doma_probe = nn.Linear(2, 1)

    model = OptimizerGroupToy()
    reversed_optimizer = torch.optim.Adam([
        {'params': model.doma_probe.parameters(), 'weight_decay': 0.0},
        {'params': model.base.parameters(), 'weight_decay': 1e-4},
    ], lr=1e-3)
    assert identify(reversed_optimizer, model) == ['DOMA', 'Base']

    named_optimizer = torch.optim.Adam([
        {'params': model.base.parameters(), 'name': 'Backbone'},
        {'params': model.doma_probe.parameters(), 'name': 'Adapters'},
    ], lr=1e-3)
    assert identify(named_optimizer, model) == ['Backbone', 'Adapters']


def _assert_train_ddp_wiring():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    train_ddp_path = os.path.join(tools_dir, 'train_ddp.py')
    with open(train_ddp_path, 'r', encoding='utf-8') as source_file:
        source = source_file.read()
    helper_source = inspect.getsource(ddp_safe_utils)

    required_train_tokens = (
        'ddp_safe_validation_enabled(hypes)',
        'DistributedEvalSampler(',
        'checkpoint_writer_enabled(',
        'sync_module_buffers_from_rank0(',
        'all_ranks_have_valid_training_batch(',
        'reduce_validation_sum_count(',
        'validation_model = model_without_ddp if ddp_safe_active else model',
        'model.eval()',
        'local_loss_sum += final_loss.item()',
        'local_count += 1',
        'dist.broadcast(is_new_best, src=0)',
        'dist.barrier()',
        'checkpoint_writer=rank0',
        'validation_reduction=global_sum_count',
        'statistics.mean(valid_ave_loss)',
        'optimizer.register_step_post_hook(record_optimizer_step)',
        'loader_iterations += 1',
        'skipped_iterations += 1',
        "optimizer_step_counts['epoch']",
        "optimizer_step_counts['cumulative']",
        '[Optimizer Groups]',
        '[DDP Train Epoch Summary]',
        '[before training]',
    )
    for token in required_train_tokens:
        assert token in source, token
    assert 'dist.all_reduce(totals, op=dist.ReduceOp.SUM)' in helper_source
    assert 'dist.broadcast(buffer, src=0)' in helper_source
    assert 'dist.all_reduce(validity, op=dist.ReduceOp.MIN)' in helper_source

    training_loop = source.index(
        'for i, batch_data in enumerate(train_loader):')
    validity_collective = source.index(
        'ddp_safe_utils.all_ranks_have_valid_training_batch(', training_loop)
    synchronized_skip = source.index(
        'if not train_batch_valid:', validity_collective)
    training_forward = source.index(
        "ouput_dict = model(batch_data['ego'])", synchronized_skip)
    assert (training_loop < validity_collective < synchronized_skip <
            training_forward)
    assert 'continue' not in source[training_loop:validity_collective]

    eval_event = source.index(
        "if epoch % hypes['train_params']['eval_freq'] == 0:")
    buffer_sync = source.index(
        'ddp_safe_utils.sync_module_buffers_from_rank0(', eval_event)
    validation_loop = source.index(
        'for i, batch_data in enumerate(val_loader):', buffer_sync)
    validation_forward = source.index(
        "ouput_dict = validation_model(batch_data['ego'])", validation_loop)
    assert eval_event < buffer_sync < validation_loop < validation_forward


def main():
    _assert_config_parsing()
    _assert_training_batch_validity()
    _assert_global_sum_count()
    _assert_nonpadding_validation_shards()
    _assert_dataloader_preserves_tail_batch()
    _assert_none_batch_is_not_counted()
    _assert_gloo_buffer_sync_and_eval_mode()
    _assert_checkpoint_writer()
    _assert_training_log_accounting()
    _assert_optimizer_group_labels()
    _assert_train_ddp_wiring()
    print(json.dumps({
        'status': 'PASS',
        'safe_mode_default': False,
        'mock_world_size': 2,
        'training_validity_cases': [True, False, False, False],
        'gloo_training_decisions': [True, False, False, False, True],
        'gloo_optimizer_steps_per_rank': 2,
        'gloo_final_parameters_synchronized': True,
        'global_validation_loss': 3.0,
        'dataloader_indices': [0, 1, 2, 3, 4],
        'rank0_tail_batch': [4],
        'none_batch_local_count': 2,
        'gloo_buffer_sync_world_size': 2,
        'batchnorm_buffers_synced': True,
        'wrapper_module_eval_consistent': True,
        'validation_forward_uses_underlying_module': True,
        'safe_checkpoint_writers': [0],
        'legacy_checkpoint_writers': [0, 1],
        'epoch_accounting': [[5, 2, 3, 3], [4, 1, 3, 6]],
        'amp_optimizer_step_hook_count': 1,
        'amp_overflow_step_hook_count': 0,
        'reversed_optimizer_group_labels': ['DOMA', 'Base'],
    }, indent=2))


if __name__ == '__main__':
    main()
