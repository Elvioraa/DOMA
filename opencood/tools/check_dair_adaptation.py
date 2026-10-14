"""CPU unit acceptance using synthetic BEV tensors, NOT real DAIR inference.

Real DOMA object modules are exercised without mocking optional encoders. Tiny
linear/BN fixtures below test checkpoint contracts only, not PointPillar/SECOND/LSS.
"""
import ast
import copy
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.doma_object_loss import compute_doma_object_loss
from opencood.models.sub_modules.doma_object import (
    install_doma_modules, configure_doma_trainability, build_collab_doma_context,
    attach_collab_doma_pyramid_context, run_doma_training,
    build_single_doma_context, attach_single_doma_pyramid_context,
)
from opencood.tools.dair_protocol import (
    validate_protocol, raw_config, inherit_state, assert_shared_equal, parameter_counts,
    RANGE, digest, load_training_state,
)
from opencood.tools.doma_tools import apply_doma_merge_ownership
from opencood.tools.train_utils import setup_optimizer

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'opencood/hypes_yaml/dairv2x/MoreModality/DAIR_PROTOCOL'


class ObjectHarness(nn.Module):
    def __init__(self, args, fixtures=False):
        super().__init__()
        self.modality_name_list = [k for k in args if k.startswith('m') and k[1:].isdigit()]
        self.sensor_type_dict = {k: args[k]['sensor_type'] for k in self.modality_name_list}
        for k in self.modality_name_list:
            if self.sensor_type_dict[k] == 'camera':
                grid = args[k]['camera_mask_args']['grid_conf']
                setattr(self, 'crop_ratio_H_' + k, args['lidar_range'][4] / grid['ybound'][1])
                setattr(self, 'crop_ratio_W_' + k, args['lidar_range'][3] / grid['xbound'][1])
        install_doma_modules(self, args)
        configure_doma_trainability(self)
        self._doma_log_printed = True
        if fixtures:
            self.pyramid_backbone = nn.BatchNorm1d(2)
            self.cls_head = nn.Linear(2, 2)
            for m in self.modality_name_list:
                setattr(self, 'encoder_' + m, nn.Linear(2, 2))
            if args['doma']['mode'] == 'stage2_adapt':
                for module in (self.pyramid_backbone, self.cls_head):
                    module.requires_grad_(False).eval()


def rejected(call):
    try:
        call()
    except (RuntimeError, ValueError, FileNotFoundError, KeyError):
        return
    raise AssertionError('Expected fail-closed rejection')


def object_step(model, agents, modality):
    # Actual DAIR physical range and common-BEV shape, synthetic feature values.
    detail = torch.randn(agents, 64, 128, 256, requires_grad=True)
    context_feature = torch.randn(agents, 128, 64, 128, requires_grad=True)
    transforms = torch.eye(2, 3).reshape(1, 1, 1, 2, 3).repeat(1, agents, agents, 1, 1)
    lengths = torch.tensor([agents])
    modalities = [modality] * agents
    if model.doma_config['mode'] == 'stage2_adapt':
        context = build_single_doma_context(model, detail, modality)
        attach_single_doma_pyramid_context(model, context, [detail, context_feature], modality)
    else:
        context = build_collab_doma_context(model, detail, lengths, transforms, modalities)
        attach_collab_doma_pyramid_context(model, context, [detail, context_feature], lengths, transforms, modalities)
    data = dict(object_bbx_center=torch.tensor([[[0., 0., -1., 1.56, 1.6, 3.9, .2],
                                               [10., 5., -1., 1.6, 2., 4., -.1]]]),
                object_bbx_mask=torch.tensor([[1, 1]], dtype=torch.bool))
    loss, stats = compute_doma_object_loss(run_doma_training(model, context, data))
    assert torch.isfinite(loss) and stats['doma_valid_object_ratio'] > 0
    loss.backward()
    assert detail.grad is not None and torch.isfinite(detail.grad).all()
    assert context_feature.grad is not None and torch.isfinite(context_feature.grad).all()
    return float(loss.detach())


def main():
    torch.set_num_threads(2)
    torch.manual_seed(303)
    files = sorted(CONFIGS.rglob('config.yaml'))
    assert len(files) == 14
    for file in files:
        validate_protocol(load_yaml(str(file)))
    # Exact preservation of original DAIR stage budgets/frontends and V2R2 method.
    original = CONFIGS.parent / 'HEAL'
    method = raw_config(ROOT / 'opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2/stage1/m1.yaml')['model']['args']['doma']
    for family in ('HEAL', 'DOMA'):
        s1 = raw_config(CONFIGS / family / 'stage1/config.yaml')
        baseline = raw_config(original / 'stage1/m1_pyramid.yaml')
        assert s1['model']['args']['m1'] == baseline['model']['args']['m1']
        for e, filename in zip(('E1', 'E2', 'E3'), ('m2', 'm3', 'm4')):
            c = raw_config(CONFIGS / family / e / 'stage2_m2/config.yaml')
            b = raw_config(original / ('stage2/' + filename + '_single_pyramid.yaml'))
            assert c['model']['args']['m2'] == b['model']['args']['m2']
            assert c['heter'] == b['heter'] and c['label_type'] == 'camera'
            assert c['lr_scheduler'] == b['lr_scheduler']
            if family == 'DOMA':
                d = copy.deepcopy(c['model']['args']['doma'])
                d['mode'] = 'stage1_anchor'
                d.pop('active_modality')
                assert d == method
    cfg = raw_config(CONFIGS / 'DOMA/stage1/config.yaml')
    assert cfg['model']['args']['doma'] == method
    s1 = ObjectHarness(cfg['model']['args'], fixtures=True)
    optimizer = setup_optimizer(cfg, s1)
    assert [g['lr'] for g in optimizer.param_groups] == [.002, .002]
    assert [g['weight_decay'] for g in optimizer.param_groups] == [1e-4, 0.]
    losses = []
    for agents in (1, 2):
        optimizer.zero_grad()
        losses.append(object_step(s1, agents, 'm1'))
        shared = [p for n, p in s1.named_parameters() if n.startswith('doma_shared_')]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in shared)
        optimizer.step()
    state1 = copy.deepcopy(s1.state_dict())
    for e in ('E1', 'E2', 'E3'):
        scfg = raw_config(CONFIGS / 'DOMA' / e / 'stage2_m2/config.yaml')
        s2 = ObjectHarness(scfg['model']['args'], fixtures=True)
        inherit_state(s2, state1)
        assert all(not p.requires_grad for n, p in s2.named_parameters() if n.startswith('doma_shared_'))
        opt = setup_optimizer(scfg, s2)
        assert [g['lr'] for g in opt.param_groups] == [.001, .001]
        frozen = copy.deepcopy(s2.state_dict())
        opt.zero_grad()
        object_step(s2, 1, 'm2')
        assert s2.doma_object_adapter_m2.delta[-1].weight.grad is not None
        opt.step()
        assert_shared_equal(frozen, s2.state_dict())
        assert not torch.equal(frozen['doma_object_adapter_m2.delta.3.weight'], s2.state_dict()['doma_object_adapter_m2.delta.3.weight'])
        state2 = s2.state_dict()
        merged = dict(state2)
        merged.update(state1)
        merged = apply_doma_merge_ownership(merged, [state2, state1])
        final_cfg = raw_config(CONFIGS / 'DOMA' / e / 'final_m1m2/config.yaml')
        final = ObjectHarness(final_cfg['model']['args'], fixtures=True)
        final.load_state_dict(merged, strict=True)
        assert torch.equal(merged['encoder_m1.weight'], state1['encoder_m1.weight'])
        assert torch.equal(merged['encoder_m2.weight'], state2['encoder_m2.weight'])
        assert_shared_equal(state1, merged)
        bad = copy.deepcopy(state2)
        bad['pyramid_backbone.running_mean'][0] += 1
        rejected(lambda: apply_doma_merge_ownership(merged, [bad, state1]))
        bad = {k: v for k, v in state1.items() if not k.startswith('doma_')}
        rejected(lambda: inherit_state(s2, bad))
        bad = dict(state1, surprise=torch.zeros(1))
        rejected(lambda: inherit_state(s2, bad))
        bad = copy.deepcopy(state1)
        bad['cls_head.weight'] = torch.zeros(3, 2)
        rejected(lambda: inherit_state(s2, bad))
        bad = dict(state2, encoder_m3_weight=torch.zeros(1))
        rejected(lambda: apply_doma_merge_ownership(merged, [bad, state1]))
        bad = dict(state2)
        bad.pop('doma_object_adapter_m2.delta.3.bias')
        rejected(lambda: apply_doma_merge_ownership(merged, [bad, state1]))
    # Exercise the actual load_saved_model epoch and provenance path on fixture
    # checkpoints in a temporary directory, never on experiment artifacts.
    from opencood.tools.train_utils import load_saved_model
    import shutil
    with tempfile.TemporaryDirectory(prefix='dair-contract-test-') as temporary:
        root = Path(temporary)
        cp = root / 'source.pth'
        torch.save(state1, cp)
        source_config = CONFIGS / 'DOMA/stage1/config.yaml'
        stage2_config = CONFIGS / 'DOMA/E3/stage2_m2/config.yaml'
        run = root / 'stage2'
        run.mkdir()
        shutil.copyfile(cp, run / 'net_epoch1.pth')
        shutil.copyfile(stage2_config, run / 'config.yaml')
        record = dict(checkpoint=str(cp.resolve()), checkpoint_sha256=digest(cp),
                      config=str(source_config.resolve()), config_sha256=digest(source_config),
                      config_snapshot=raw_config(source_config), stage2_config_sha256=digest(stage2_config))
        (run / 'stage1_source.json').write_text(json.dumps(record), encoding='utf8')
        fresh = ObjectHarness(scfg['model']['args'], fixtures=True)
        epoch, _ = load_saved_model(str(run), fresh, dair_config=scfg)
        assert epoch == 1 and len(range(epoch, max(30, epoch))) == 29
        torch.save(state2, run / 'net_epoch_bestval_at3.pth')
        epoch, _ = load_saved_model(str(run), fresh, dair_config=scfg)
        assert epoch == 3
        corrupted = copy.deepcopy(state2)
        corrupted['pyramid_backbone.running_var'][0] += 1
        torch.save(corrupted, run / 'net_epoch_bestval_at3.pth')
        rejected(lambda: load_saved_model(str(run), fresh, dair_config=scfg))
        record['checkpoint_sha256'] = 'wrong'
        (run / 'stage1_source.json').write_text(json.dumps(record), encoding='utf8')
        rejected(lambda: load_saved_model(str(run), fresh, dair_config=scfg))
    # Non-square ROI center/endpoints map independently in x/y.
    grid = s1.doma_object_roi._build_grid(torch.tensor([[0., 0., 0., 1., 2., 4., 0.]]),128,256)
    assert torch.allclose(grid[0,2,2], torch.zeros(2), atol=1e-7)
    assert abs(float(grid[...,0].max()) - 1.6/102.4) < 1e-6
    assert abs(float(grid[...,1].max()) - .8/51.2) < 1e-6
    # Real module counts, not full detector counts.
    counts = parameter_counts(s2)
    assert counts['all']['total'] == counts['all']['trainable'] + counts['all']['frozen']
    # Entry point must not call the ordered main or mutate protocol fields.
    text = (ROOT / 'opencood/tools/inference_dair.py').read_text()
    assert 'inference_intermediate_fusion_doma(batch, model, dataset)' in text
    assert 'update_dict' not in text and '_in_order.json' not in text
    tree = ast.parse((ROOT / 'opencood/utils/eval_utils.py').read_text())
    keys = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {'ap30', 'ap_50', 'ap_70'} <= keys
    from opencood.tools.ddp_safe_utils import DistributedEvalSampler
    shards = [list(DistributedEvalSampler(range(1789), 2, r)) for r in (0, 1)]
    assert list(map(len, shards)) == [895, 894]
    assert set(shards[0]).isdisjoint(shards[1])
    final_cfg = raw_config(CONFIGS / 'DOMA/E1/final_m1m2/config.yaml')
    for mutation in ('comm_range', 'range', 'mapping', 'dataset'):
        wrong = copy.deepcopy(final_cfg)
        if mutation == 'comm_range':
            wrong['comm_range'] = 180
        elif mutation == 'range':
            wrong['model']['args']['lidar_range'] = [-204.8, -102.4, -3, 204.8, 102.4, 1]
        elif mutation == 'mapping':
            wrong['heter']['mapping_dict']['m3'] = 'm3'
        else:
            wrong['fusion']['core_method'] += 'infer'
        rejected(lambda: validate_protocol(wrong, 'final'))
    print(json.dumps(dict(status='CPU_COMPONENT_PASS', yaml_count=len(files),
          synthetic_stage1_losses=losses, object_stage2=['E1','E2','E3'],
          merge='two-source ownership + negative cases', validation_shards=[895,894],
          limitation='Synthetic BEV/object modules and checkpoint fixtures only; no real encoders, DAIR media, AP run or CUDA/DDP'), indent=2))


if __name__ == '__main__':
    main()
