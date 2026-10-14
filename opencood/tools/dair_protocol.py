"""Explicit DAIR protocol, checkpoint ownership and provenance (no dataset imports)."""
import hashlib
import json
from pathlib import Path

import torch
import yaml

RANGE = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
BACKEND = ('pyramid_backbone.', 'shrink_conv.', 'cls_head.', 'reg_head.', 'dir_head.')
FRONT1 = ('encoder_m1.', 'backbone_m1.', 'aligner_m1.')
FRONT2 = ('encoder_m2.', 'backbone_m2.', 'aligner_m2.',
          'doma_object_adapter_m2.', 'doma_context_adapter_m2.')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_config(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf8'))


def enabled(cfg):
    return cfg.get('dair_protocol', {}).get('enabled') is True


def validate_protocol(cfg, stage=None):
    """Fail on drift; never rewrite the loaded experiment configuration."""
    require(enabled(cfg), 'Explicit dair_protocol.enabled required')
    p = cfg['dair_protocol']
    actual = p['stage']
    require(actual in ('stage1', 'stage2', 'final'), 'Unknown DAIR stage')
    require(stage is None or actual == stage, 'Wrong DAIR stage')
    require(p['method'] in ('HEAL', 'DOMA'), 'Unknown method')
    require(p['evaluation_split'] == 'validation', 'DAIR evaluates validation')
    require(p['train_samples'] == 4811 and p['validation_samples'] == 1789,
            'Unexpected split sizes')
    require(cfg['seed'] == 303 and cfg['comm_range'] == 100, 'Seed/communication drift')
    require(cfg['fusion']['dataset'] == 'dairv2x', 'Expected DAIR Dataset')
    require(cfg['cav_lidar_range'] == RANGE and cfg['model']['args']['lidar_range'] == RANGE,
            'DAIR model range drift')
    require(cfg['preprocess']['cav_lidar_range'] == RANGE
            and cfg['postprocess']['gt_range'] == RANGE
            and cfg['postprocess']['anchor_args']['cav_lidar_range'] == RANGE,
            'DAIR preprocessing/GT/anchor range drift')
    require(cfg['test_dir'] == cfg['validate_dir'], 'Final must evaluate validation split')
    require(Path(cfg['root_dir']).name == 'train.json'
            and Path(cfg['validate_dir']).name == 'val.json', 'Unexpected split names')
    tp = cfg['train_params']
    require(tp['max_cav'] == 2 and tp['eval_freq'] == 2 and tp['save_freq'] == 2,
            'Agent count/evaluation cadence drift')
    require(tp['ddp_safe_validation']['enabled'] is True, 'Safe DDP must be enabled')
    require(tp['run_inference_after_training'] is False, 'Use separate DAIR Final inference')
    heter = cfg['heter']
    keys = {k for k in cfg['model']['args'] if k.startswith('m') and k[1:].isdigit()}
    expected = {'m1'} if actual == 'stage1' else ({'m2'} if actual == 'stage2' else {'m1', 'm2'})
    require(keys == expected and set(heter['modality_setting']) == expected, 'Wrong modality keys')
    mapping = {'m1': 'm1', 'm2': 'm1'} if actual == 'stage1' else (
        {'m1': 'm2', 'm2': 'm2'} if actual == 'stage2' else {'m1': 'm1', 'm2': 'm2'})
    require(heter['mapping_dict'] == mapping, 'Physical modality mapping drift')
    require(cfg['fusion']['core_method'] == ('lateheter' if actual == 'stage2' else 'intermediateheter'),
            'Do not use ordered/infer Dataset')
    require(cfg['label_type'] == ('camera' if actual == 'stage2' else 'lidar'), 'Label branch drift')
    if actual == 'final':
        require(heter['ego_modality'] == 'm1' and heter['assignment_path'] is None,
                'Final requires Vehicle=m1, Infrastructure=m2, native assignment')
    else:
        first = actual == 'stage1'
        require(tp['batch_size'] == 2 and tp['epoches'] == (40 if first else 30), 'Training budget drift')
        require(cfg['optimizer']['core_method'] == 'Adam'
                and cfg['optimizer']['lr'] == (.002 if first else .001), 'Optimizer drift')
        require(cfg['lr_scheduler']['step_size'] == ([15, 30] if first else [15, 25]), 'Milestone drift')
        require(float(cfg['optimizer']['args']['weight_decay']) == 1e-4
                and float(cfg['optimizer']['args']['eps']) == 1e-10, 'Adam arguments drift')
    if p['method'] == 'DOMA':
        from opencood.models.sub_modules.doma_config import validate_doma_config
        d = cfg['model']['args']['doma']
        validate_doma_config(d)
        require(d['mode'] == {'stage1': 'stage1_anchor', 'stage2': 'stage2_adapt', 'final': 'inference'}[actual],
                'Wrong DOMA training mode')
        require(actual != 'stage2' or d['active_modality'] == 'm2', 'Wrong Adapter owner')
        require(cfg['loss']['core_method'] == 'doma_point_pillar_pyramid_loss', 'DOMA loss required')
        require(cfg['optimizer']['doma_param_groups'] == {'enabled': True, 'weight_decay': 0.0},
                'V2R2 DOMA decay must be explicit')
    return cfg


def load_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()),
            'Expected a model state_dict, not an optimizer/trainer checkpoint')
    require(all(not v.is_floating_point() or torch.isfinite(v).all().item() for v in state.values()),
            'Non-finite checkpoint tensor')
    return state


def shared_keys(state):
    from opencood.tools.doma_tools import SHARED_PREFIXES
    return {k for k in state if k.startswith(BACKEND + SHARED_PREFIXES)}


def assert_shared_equal(source, candidate):
    keys = shared_keys(source)
    require(keys and keys == shared_keys(candidate), 'Shared detection/DOMA key set mismatch')
    bad = [k for k in keys if source[k].shape != candidate[k].shape
           or source[k].dtype != candidate[k].dtype or not torch.equal(source[k], candidate[k].cpu())]
    require(not bad, 'Frozen parameters/buffers differ: ' + str(bad))


def inherit_state(model, source):
    """The only allowed partial load: Stage1 m1 -> fresh Stage2 m2."""
    target = model.state_dict()
    missing, extra = set(target) - set(source), set(source) - set(target)
    require(all(k.startswith(FRONT2) for k in missing), 'Unexpected missing Stage1 keys: ' + str(sorted(missing)))
    require(all(k.startswith(FRONT1) for k in extra), 'Unexpected Stage1 keys: ' + str(sorted(extra)))
    require(not any(k.startswith(FRONT2) for k in source), 'Stage1 must not contain m2 frontends/adapters')
    require(shared_keys(source) == shared_keys(target), 'Required shared modules absent')
    for k in set(target) & set(source):
        require(target[k].shape == source[k].shape and target[k].dtype == source[k].dtype,
                'Checkpoint shape/dtype mismatch: ' + k)
    result = model.load_state_dict(source, strict=False)
    require(set(result.missing_keys) == missing and set(result.unexpected_keys) == extra, 'Load contract changed')
    assert_shared_equal(source, model.state_dict())


def verify_provenance(directory, cfg):
    path = Path(directory) / 'stage1_source.json'
    record = json.loads(path.read_text(encoding='utf8'))
    require(digest(Path(directory) / 'config.yaml') == record['stage2_config_sha256'],
            'Stage2 config changed after initialization; prepare a new run')
    for field in ('checkpoint', 'config'):
        source = Path(record[field])
        require(source.is_absolute() and source.is_file(), 'Missing absolute Stage1 source: ' + str(source))
        require(digest(source) == record[field + '_sha256'], 'Stage1 source changed: ' + str(source))
    s1 = validate_protocol(raw_config(record['config']), 'stage1')
    require(s1 == record['config_snapshot'], 'Stage1 config snapshot differs')
    require(s1['dair_protocol']['method'] == cfg['dair_protocol']['method'], 'Cannot inherit another method')
    if cfg['dair_protocol']['method'] == 'DOMA':
        from opencood.tools.doma_tools import doma_method_fingerprint
        require(doma_method_fingerprint(s1['model']['args']['doma']) ==
                doma_method_fingerprint(cfg['model']['args']['doma']), 'DOMA method fingerprint differs')
    return record


def load_training_state(model, state, directory, checkpoint, cfg):
    validate_protocol(cfg)
    require(all(torch.is_tensor(v) and (not v.is_floating_point() or torch.isfinite(v).all().item())
                for v in state.values()), 'Invalid/non-finite checkpoint tensors')
    if cfg['dair_protocol']['stage'] == 'stage2':
        record = verify_provenance(directory, cfg)
        source = load_state(record['checkpoint'])
        if Path(checkpoint).name == 'net_epoch1.pth':
            require(digest(checkpoint) == record['checkpoint_sha256'], 'Initialization copy differs from Stage1')
            inherit_state(model, source)
        else:
            model.load_state_dict(state, strict=True)
            assert_shared_equal(source, model.state_dict())
    else:
        model.load_state_dict(state, strict=True)


def parameter_counts(model):
    result = {}
    for group in ('all', 'base', 'doma'):
        params = [p for n, p in model.named_parameters()
                  if group == 'all' or n.startswith('doma_') == (group == 'doma')]
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        result[group] = dict(total=total, trainable=trainable, frozen=total-trainable)
    return result


def build_model(config):
    """Real model only: no stubs, mock encoders or replacement convolutions."""
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model
    cfg = validate_protocol(load_yaml(str(config)))
    model = create_model(cfg)
    return cfg, model
