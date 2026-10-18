"""Native DAIR validation for HEAL/DOMA; no ordered-assignment overrides."""
import argparse
import json
import re
from pathlib import Path

from opencood.tools.dair_protocol import digest, load_state, require, validate_protocol, raw_config


def _json_default(value):
    """Convert report values only; never rewrite the inference configuration."""
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Unsupported protocol value: %s' % type(value).__name__)


def _report_json(report):
    return json.dumps(report, indent=2, default=_json_default, allow_nan=False) + '\n'


def _field(cfg, path):
    value = cfg
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return {'present': False}
        value = value[key]
    return {'present': True, 'value': value}


def write_effective_protocol(cfg, config_path, directory, checkpoint, max_samples=None):
    """Persist parsed configuration evidence before any model/dataset construction.

    Values are configuration inputs, not measurements of a constructed model.
    Relative dataset paths are interpreted against the recorded working directory.
    """
    paths = [
        'dair_protocol.method', 'dair_protocol.experiment', 'fusion.dataset',
        'fusion.core_method', 'data_dir', 'root_dir', 'validate_dir', 'test_dir',
        'cav_lidar_range', 'preprocess.cav_lidar_range', 'postprocess.gt_range',
        'postprocess.anchor_args.cav_lidar_range', 'model.args.lidar_range',
        'comm_range', 'train_params.max_cav', 'input_source', 'label_type',
        'heter.ego_modality', 'heter.mapping_dict', 'model.core_method',
        'postprocess.anchor_args.W', 'postprocess.anchor_args.H',
        'postprocess.anchor_args.D', 'postprocess.anchor_args.feature_stride',
    ]
    modalities = sorted(set(cfg.get('heter', {}).get('modality_setting', {})) |
                        {k for k in cfg.get('model', {}).get('args', {})
                         if k.startswith('m') and k[1:].isdigit()})
    for modality in modalities:
        model = 'model.args.' + modality
        dataset = 'heter.modality_setting.' + modality
        paths.extend(model + suffix for suffix in (
            '.core_method', '.sensor_type', '.encoder_args.lidar_range',
            '.encoder_args.grid_conf.xbound', '.encoder_args.grid_conf.ybound'))
        paths.extend(dataset + suffix for suffix in (
            '.core_method', '.sensor_type', '.preprocess.cav_lidar_range',
            '.grid_conf.xbound', '.grid_conf.ybound'))
    report = dict(
        status='configuration_validated_before_model_construction',
        config_path=str(Path(config_path).resolve()), config_sha256=digest(config_path),
        model_directory=str(Path(directory).resolve()),
        checkpoint_path=str(Path(checkpoint).resolve()),
        working_directory=str(Path.cwd()),
        evaluation=dict(split='validation', mode='full' if max_samples is None else 'subset',
                        max_samples=max_samples, train=False, visualize=True,
                        batch_size=1, shuffle=False, drop_last=False),
        effective_fields={path: _field(cfg, path) for path in paths},
        raw_postprocess=_field(raw_config(config_path), 'postprocess'),
        effective_postprocess=_field(cfg, 'postprocess'),
    )
    payload = _report_json(report)
    destination = Path(directory) / 'protocol_effective.json'
    destination.write_text(payload, encoding='utf8')
    print(f'Effective protocol saved to {destination}', flush=True)
    return report


def _checkpoint_epoch(checkpoint):
    # Same filename epoch convention as HEAL load_saved_model; no source substitution.
    match = re.fullmatch(r'net_epoch(?:_bestval_at)?(\d+)\.pth',
                         str(checkpoint).replace('\\', '/').rsplit('/', 1)[-1])
    return match.group(1) if match else 'unknown'


def _infer_info(cfg, checkpoint):
    # DAIR uses the native heter Dataset, without ordered inference's use_cav control.
    fusion_method = cfg['fusion']['core_method'].removesuffix('heter')
    lidar_range = cfg['cav_lidar_range']
    return f"{fusion_method}_{lidar_range[3]}_{lidar_range[4]}_epoch{_checkpoint_epoch(checkpoint)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, help='Diagnostic subset only; never a full-validation result')
    opt = parser.parse_args()
    if opt.max_samples is not None:
        require(opt.max_samples > 0, 'max-samples must be positive')
    from opencood.hypes_yaml.yaml_utils import load_yaml

    directory = Path(opt.model_dir).resolve()
    config_path = directory / 'config.yaml'
    cfg = validate_protocol(load_yaml(str(config_path)), 'final')
    checkpoint = directory / 'net_epoch1.pth'
    write_effective_protocol(cfg, config_path, directory, checkpoint, opt.max_samples)

    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils, seed_utils, inference_utils
    from opencood.tools.inference_doma_heter_in_order import inference_intermediate_fusion_doma
    from opencood.utils import eval_utils

    seed = seed_utils.seed_from_hypes(cfg)
    provenance = json.loads((directory / 'merge_sources.json').read_text(encoding='utf8'))
    require(digest(config_path) == provenance['final_config_sha256'], 'Final config changed after merge')
    require(digest(checkpoint) == provenance['final_checkpoint_sha256'], 'Final checkpoint changed after merge')
    model = train_utils.create_model(cfg)
    model.load_state_dict(load_state(checkpoint), strict=True)
    infer_info = _infer_info(cfg, checkpoint)
    print(f'Loaded checkpoint: {checkpoint} (file epoch {_checkpoint_epoch(checkpoint)})', flush=True)
    source = provenance.get('stage2_checkpoint')
    if source is not None:
        print(f'Stage2 source checkpoint: {source} (file epoch {_checkpoint_epoch(source)})', flush=True)
    else:
        print('Stage2 source checkpoint: not recorded', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dataset = build_dataset(cfg, visualize=True, train=False)
    require(len(dataset) == cfg['dair_protocol']['validation_samples'], 'Unexpected validation size')
    selected = dataset if opt.max_samples is None else Subset(dataset, range(min(opt.max_samples, len(dataset))))
    loader = DataLoader(selected, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=opt.num_workers, collate_fn=dataset.collate_batch_test,
                        **seed_utils.dataloader_seed_kwargs(seed))
    stats = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
    processed, skipped = 0, 0
    agent_counts = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"{infer_info}_{i}", flush=True)
            if batch is None:
                skipped += 1
                continue
            modalities = batch['ego']['agent_modality_list']
            require(list(modalities) in (['m1'], ['m1', 'm2']), 'Unexpected physical/modality order')
            n = len(modalities)
            agent_counts[n] = agent_counts.get(n, 0) + 1
            batch = train_utils.to_device(batch, device)
            if cfg['dair_protocol']['method'] == 'DOMA':
                result = inference_intermediate_fusion_doma(batch, model, dataset)
            else:
                result = inference_utils.inference_intermediate_fusion(batch, model, dataset)
            for threshold in stats:
                eval_utils.caluclate_tp_fp(result['pred_box_tensor'], result['pred_score'],
                                          result['gt_box_tensor'], stats, threshold)
            processed += 1
    require(processed > 0 and stats[.3]['gt'] > 0, 'No evaluable predictions/GT frames')
    label = 'validation' if opt.max_samples is None else 'validation_subset_%d' % len(selected)
    if skipped:
        label += '_incomplete'
    ap = eval_utils.eval_final_results(stats, str(directory), label)
    report = dict(split='validation', full_split=opt.max_samples is None and skipped == 0,
                  processed=processed, skipped=skipped, agent_counts=agent_counts,
                  gt_count=stats[.3]['gt'], config_sha256=digest(config_path),
                  checkpoint_sha256=digest(checkpoint), split_sha256=digest(cfg['validate_dir']),
                  comm_range=cfg['comm_range'], lidar_range=cfg['cav_lidar_range'],
                  mapping=cfg['heter']['mapping_dict'], postprocess=cfg['postprocess'],
                  AP30=ap[0], AP50=ap[1], AP70=ap[2])
    report['raw_postprocess'] = _field(raw_config(config_path), 'postprocess')
    payload = _report_json(report)
    (directory / ('protocol_' + label + '.json')).write_text(payload, encoding='utf8')
    print(f"Validation protocol saved to {directory / ('protocol_' + label + '.json')}", flush=True)


if __name__ == '__main__':
    main()
