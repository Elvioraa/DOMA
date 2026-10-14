"""Prepare isolated DAIR runs, merge verified pairs, or report parameter details.

No operation chooses a Stage1 source automatically. Run from the repository root.
"""
import argparse
from contextlib import redirect_stdout
import json
import shutil
import sys
from pathlib import Path

import torch

from opencood.tools.dair_protocol import (
    FRONT1, FRONT2, assert_shared_equal, build_model, digest, inherit_state,
    load_state, parameter_counts, raw_config, require, validate_protocol, verify_provenance,
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf8')


def prepare(config, output, checkpoint=None, stage1_config=None):
    cfg = validate_protocol(raw_config(config))
    stage = cfg['dair_protocol']['stage']
    require(stage in ('stage1', 'stage2'), 'prepare only accepts training configs')
    out = Path(output).resolve()
    require(not out.exists(), 'Output already exists; never overwrite a run')
    record = None
    if stage == 'stage2':
        require(checkpoint and stage1_config, 'Explicit Stage1 checkpoint and config are required')
        cp, source_cfg = Path(checkpoint).resolve(), Path(stage1_config).resolve()
        source = validate_protocol(raw_config(source_cfg), 'stage1')
        require(source['dair_protocol']['method'] == cfg['dair_protocol']['method'], 'Wrong Stage1 method')
        if cfg['dair_protocol']['method'] == 'DOMA':
            from opencood.tools.doma_tools import doma_method_fingerprint
            require(doma_method_fingerprint(source['model']['args']['doma']) ==
                    doma_method_fingerprint(cfg['model']['args']['doma']), 'DOMA method differs')
        state = load_state(cp)
        _, source_model = build_model(source_cfg)
        source_model.load_state_dict(state, strict=True)
        del source_model
        _, target = build_model(config)
        inherit_state(target, state)
        record = dict(checkpoint=str(cp), checkpoint_sha256=digest(cp), config=str(source_cfg),
                      config_sha256=digest(source_cfg), config_snapshot=source,
                      stage2_config_sha256=digest(config), recovery='weights_only; init_epoch=1')
    else:
        require(not checkpoint and not stage1_config, 'Stage1 starts from scratch')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, out / 'config.yaml')
    if record:
        shutil.copyfile(record['checkpoint'], out / 'net_epoch1.pth')
        write_json(out / 'stage1_source.json', record)
    print(out)


def merge(config, stage2_dir, checkpoint, output):
    """Strict Final schema + source ownership; creates a fresh output directory."""
    final = validate_protocol(raw_config(config), 'final')
    directory = Path(stage2_dir).resolve()
    s2cfg = validate_protocol(raw_config(directory / 'config.yaml'), 'stage2')
    require(final['dair_protocol']['method'] == s2cfg['dair_protocol']['method']
            and final['dair_protocol']['experiment'] == s2cfg['dair_protocol']['experiment'],
            'Final method/experiment differs from Stage2')
    require(final['model']['args']['m2'] == s2cfg['model']['args']['m2'], 'Final m2 config differs')
    record = verify_provenance(directory, s2cfg)
    require(final['model']['args']['m1'] == record['config_snapshot']['model']['args']['m1'],
            'Final m1 config differs from Stage1')
    cp = Path(checkpoint).resolve()
    require(cp.parent == directory and cp.name != 'net_epoch1.pth',
            'Select a trained Stage2 checkpoint in its run directory, not the initialization')
    out = Path(output).resolve()
    require(not out.exists(), 'Final output already exists')
    stage1, stage2 = load_state(record['checkpoint']), load_state(cp)
    _, model2 = build_model(directory / 'config.yaml')
    model2.load_state_dict(stage2, strict=True)
    del model2
    assert_shared_equal(stage1, stage2)
    merged = dict(stage2)
    merged.update(stage1)
    if final['dair_protocol']['method'] == 'DOMA':
        from opencood.tools.doma_tools import apply_doma_merge_ownership, doma_method_fingerprint
        require(doma_method_fingerprint(final['model']['args']['doma']) ==
                doma_method_fingerprint(s2cfg['model']['args']['doma']), 'Final DOMA method differs')
        merged = apply_doma_merge_ownership(merged, [stage2, stage1])
    _, model = build_model(config)
    model.load_state_dict(merged, strict=True)
    for key in model.state_dict():
        owner = stage2 if key.startswith(FRONT2) else stage1
        require(key in owner and torch.equal(merged[key], owner[key]), 'Wrong merge owner: ' + key)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, out / 'config.yaml')
    torch.save(merged, out / 'net_epoch1.pth')
    write_json(out / 'merge_sources.json', dict(stage1=record, stage2_checkpoint=str(cp),
               stage2_sha256=digest(cp), stage2_config_sha256=digest(directory / 'config.yaml'),
               final_config_sha256=digest(config), final_checkpoint_sha256=digest(out / 'net_epoch1.pth')))
    print(out)


def counts(stage1, stage2, final):
    configs = [validate_protocol(raw_config(p), stage) for p, stage in
               zip((stage1, stage2, final), ('stage1', 'stage2', 'final'))]
    require(len({c['dair_protocol']['method'] for c in configs}) == 1, 'Mixed methods')
    require(configs[1]['dair_protocol']['experiment'] == configs[2]['dair_protocol']['experiment'], 'Mixed experiments')
    result = {}
    for name, path in zip(('stage1', 'stage2', 'final'), (stage1, stage2, final)):
        with redirect_stdout(sys.stderr):
            _, model = build_model(path)
        result[name] = dict(config=str(Path(path).resolve()), sha256=digest(path), counts=parameter_counts(model))
        del model
    result['two_stage_raw_sums'] = {
        field: result['stage1']['counts']['all'][field] + result['stage2']['counts']['all'][field]
        for field in ('total', 'trainable', 'frozen')}
    result['paper_sigma_m_params'] = None
    result['paper_status'] = ('UNCONFIRMED counting boundary: Table 3 accumulates collaboration-base '
                              'and integration model parameters; neither raw sum is asserted to reproduce it. '
                              'Final model count is reported separately; buffers/DDP replicas excluded.')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage1-checkpoint')
    p.add_argument('--stage1-config')
    p = sub.add_parser('merge')
    p.add_argument('--config', required=True)
    p.add_argument('--stage2-dir', required=True)
    p.add_argument('--stage2-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('counts')
    for name in ('stage1', 'stage2', 'final'):
        p.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config, args.output, args.stage1_checkpoint, args.stage1_config)
    elif args.command == 'merge':
        merge(args.config, args.stage2_dir, args.stage2_checkpoint, args.output)
    else:
        counts(args.stage1, args.stage2, args.final)


if __name__ == '__main__':
    main()
