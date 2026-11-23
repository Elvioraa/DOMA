"""Prepare, strictly merge and evaluate isolated V2XSet independent detectors.

No command launches training. Run the existing tools.train entry point after prepare.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'opencood/hypes_yaml/v2xset/Baselines'
ROOT = REPO / 'opencood/logs/V2XSET_BASELINES_SEED303'
NAMES = dict(zip(('m1', 'm2', 'm3', 'm4'),
                 ('m1_pointpillar', 'm2_lss_effnet', 'm3_second', 'm4_lss_resnet101')))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    return load_yaml(str(path))


def build(config):
    from opencood.models.heter_model_late import HeterModelLate
    if config['model']['core_method'] != 'heter_model_late':
        raise ValueError('Only independently trained HeterModelLate weights are accepted')
    return HeterModelLate(config['model']['args']).cpu()


def load_state(path):
    import torch
    # Do not deserialize arbitrary Python checkpoint objects.
    return torch.load(str(path), map_location='cpu', weights_only=True)


def merge_states(target, sources):
    """Check exact key ownership, shapes, dtypes and shared tensor equality."""
    import torch
    merged = {}
    prefixes = ('encoder_', 'backbone_', 'layers_', 'shrink_conv_',
                'cls_head_', 'reg_head_', 'dir_head_')
    for modality, state in sources.items():
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or key not in target:
                raise ValueError('Unexpected checkpoint key: ' + key)
            root = key.split('.')[0]
            if root.startswith(prefixes) and not root.endswith('_' + modality):
                raise ValueError('Wrong modality owner: ' + key)
            if tensor.shape != target[key].shape or tensor.dtype != target[key].dtype:
                raise ValueError('Shape/dtype mismatch: ' + key)
            if key in merged and not torch.equal(merged[key], tensor):
                raise ValueError('Conflicting shared parameter: ' + key)
            merged[key] = tensor
    missing = set(target) - set(merged)
    if missing:
        raise ValueError('Missing parameters: ' + ', '.join(sorted(missing)))
    return merged


def prepare(modality):
    source = CONFIG / (NAMES[modality] + '.yaml')
    destination = ROOT / 'single' / NAMES[modality]
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'config.yaml').open('x', encoding='utf-8') as stream:
        stream.write(source.read_text(encoding='utf-8'))
    print('Prepared fresh training directory:', destination)


def assignments():
    from opencood.utils.heter_utils import assign_modality_4, assign_modality_4_in_order
    data = REPO / 'dataset/V2XSET'
    for split in ('train', 'validate', 'test'):
        if not (data / split).is_dir():
            raise FileNotFoundError(data / split)
    destination = REPO / 'opencood/logs/heter_modality_assign'
    destination.mkdir(parents=True, exist_ok=True)
    for name, generator in (('v2xset_4modality.json', assign_modality_4),
                            ('v2xset_4modality_in_order.json', assign_modality_4_in_order)):
        path = destination / name
        if path.exists():
            assignment = json.loads(path.read_text())
            for split in (('test',) if 'in_order' in name else ('train', 'validate', 'test')):
                for scene in sorted((data / split).iterdir()):
                    if not scene.is_dir():
                        continue
                    cavs = sorted(p.name for p in scene.iterdir() if p.is_dir())
                    if 'in_order' in name and cavs and cavs[0] == '-1':
                        cavs = cavs[1:] + cavs[:1]
                    for i, cav in enumerate(cavs):
                        value = assignment[scene.name][cav]
                        if value not in NAMES or ('in_order' in name and value != f'm{i % 4 + 1}'):
                            raise ValueError('Invalid modality assignment: ' + str(scene / cav))
            print('Validated existing assignment; unchanged:', path)
            continue
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / name
            generator(str(data), str(generated))
            with path.open('x', encoding='utf-8') as stream:
                stream.write(generated.read_text())
        print('Generated assignment:', path)


def checkpoint(path):
    path = Path(path)
    if path.is_dir():
        choices = list(path.glob('net_epoch_bestval_at*.pth'))
        if len(choices) != 1:
            raise ValueError('Require exactly one best-validation-loss checkpoint: ' + str(path))
        return choices[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def merge(args):
    import torch
    final = read_config(CONFIG / 'final.yaml')
    model = build(final)
    sources, manifest = {}, {}
    for modality in NAMES:
        path = checkpoint(getattr(args, modality))
        trained_config = read_config(path.parent / 'config.yaml')
        expected = read_config(CONFIG / (NAMES[modality] + '.yaml'))
        if trained_config != expected:
            raise ValueError('Training config differs from the independent baseline: ' + str(path))
        state = load_state(path)
        single = build(expected)
        single.load_state_dict(state, strict=True)
        del single
        sources[modality] = state
        manifest[modality] = {'checkpoint': str(path.resolve()), 'sha256': sha256(path),
                              'config_sha256': sha256(path.parent / 'config.yaml')}
    state = merge_states(model.state_dict(), sources)
    model.load_state_dict(state, strict=True)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    weight = destination / 'net_epoch1.pth'
    with weight.open('xb') as stream:
        torch.save(state, stream)
    # Verify the serialized result as well, not just the in-memory dictionary.
    model.load_state_dict(load_state(weight), strict=True)
    with (destination / 'config.yaml').open('x', encoding='utf-8') as stream:
        stream.write((CONFIG / 'final.yaml').read_text(encoding='utf-8'))
    manifest.update(merged_sha256=sha256(weight), final_config_sha256=sha256(destination / 'config.yaml'),
                    strict_load=True, checkpoint_epoch_note='1 is a filename identifier, not a training epoch')
    with (destination / 'sources.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print('Strict merge and serialized reload PASS:', weight)


def infer(args):
    import torch
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import inference_utils, seed_utils, train_utils
    from opencood.tools.inference_heter_in_order import inference_late_fusion_heter_in_order
    from opencood.utils import eval_utils

    directory = Path(args.model_dir)
    config_path = directory / 'config.yaml'
    if read_config(config_path) != read_config(CONFIG / 'final.yaml'):
        raise ValueError('Final configuration differs from the approved baseline')
    config = read_config(config_path)
    config['validate_dir'] = config['test_dir']
    seed = seed_utils.seed_from_hypes(config)
    weight = directory / 'net_epoch1.pth'
    provenance = json.loads((directory / 'sources.json').read_text())
    if sha256(weight) != provenance['merged_sha256'] or sha256(config_path) != provenance['final_config_sha256']:
        raise ValueError('Final checkpoint/config hash differs from merge manifest')
    model = build(config)
    model.load_state_dict(load_state(weight), strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dataset = build_dataset(config, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=dataset.collate_batch_test,
                        **seed_utils.dataloader_seed_kwargs(seed))
    # Same complete dataset/GT is used for every prediction member count.
    counts = [1] if args.fusion_method == 'no' else [1, 2, 3, 4]
    if args.use_cav is not None:
        if args.fusion_method != 'late' or args.use_cav not in counts:
            raise ValueError('--use_cav is only valid for late with 1/2/3/4')
        counts = [args.use_cav]
    plans = [('no_fusion' if args.fusion_method == 'no' else f'late_use_cav{k}', k) for k in counts]
    for name, _ in plans:
        if (directory / name).exists():
            raise FileExistsError('Refusing to overwrite results: ' + str(directory / name))
    for name, count in plans:
        output = directory / name
        output.mkdir(exist_ok=False)
        stats = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (0.3, 0.5, 0.7)}
        protocol = {'seed': seed, 'test_dir': config['test_dir'], 'sample_count': len(dataset),
                    'gt_type': config['baseline_protocol']['gt'], 'label_type': dataset.label_type,
                    'gt_range': config['postprocess']['gt_range'], 'comm_range': config['comm_range'],
                    'max_cav': dataset.max_cav, 'ego_modality': dataset.ego_modality,
                    'assignment_sha256': sha256(config['heter']['assignment_path']),
                    'checkpoint_sha256': sha256(weight), 'config_sha256': sha256(config_path),
                    'camera_compatible': config['v2xset_camera_compatible'],
                    'use_cav': count, 'method': args.fusion_method, 'processed_samples': 0,
                    'single_vehicle_baseline': args.fusion_method == 'no'}
        print(json.dumps(protocol, indent=2))
        with (output / 'protocol.json').open('x') as stream:
            json.dump(protocol, stream, indent=2)
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    raise RuntimeError('Unexpected missing sample; refusing to alter sample protocol')
                batch = train_utils.to_device(batch, device)
                result = (inference_utils.inference_no_fusion(batch, model, dataset)
                          if args.fusion_method == 'no' else
                          inference_late_fusion_heter_in_order(batch, model, dataset, count))
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(result['pred_box_tensor'], result['pred_score'],
                                              result['gt_box_tensor'], stats, threshold)
                protocol['processed_samples'] += 1
        eval_utils.eval_final_results(stats, str(output))
        protocol['gt_count'] = stats[0.5]['gt']
        protocol['ap'] = yaml.safe_load((output / 'eval.yaml').read_text())
        with (output / 'completed.json').open('x') as stream:
            json.dump(protocol, stream, indent=2)
        print('Completed:', output)


def audit_data(args):
    """CPU-only real-data parity check, without model construction or forward."""
    import numpy as np
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import seed_utils
    from opencood.utils.common_utils import update_dict

    final = read_config(CONFIG / 'final.yaml')
    reference = read_config(REPO / 'opencood/hypes_yaml/v2xset/MoreModality/HEAL/final_infer/m1m2m3m4.yaml')
    reference = update_dict(reference, {key: final['postprocess']['gt_range']
                                       for key in ('cav_lidar_range', 'lidar_range', 'gt_range')})
    from opencood.hypes_yaml import yaml_utils
    reference = getattr(yaml_utils, reference['yaml_parser'])(reference)
    reference['comm_range'] = 180
    reference['fusion']['core_method'] = 'intermediateheterinfer'
    reference['heter']['assignment_path'] = final['heter']['assignment_path']
    reference['heter']['ego_modality'] = 'm1'
    reference['heter']['mapping_dict'] = {m: m for m in NAMES}
    reference['heter']['lidar_channels_dict'] = {}
    reference['use_cav'] = 4

    def frames(dataset):
        return [(Path(dataset.scenario_folders[i]).name,
                 next(iter(cavs)), tuple(k for k in next(iter(cavs.values())) if k != 'ego'))
                for i, cavs in dataset.scenario_database.items()]

    for split in ('train', 'validate', 'test'):
        signatures = None
        for name in list(NAMES.values()) + ['final']:
            config = read_config(CONFIG / (name + '.yaml'))
            config['root_dir'] = config['validate_dir'] = 'dataset/V2XSET/' + split
            if split != 'test':
                config['heter']['assignment_path'] = 'opencood/logs/heter_modality_assign/v2xset_4modality.json'
            seed_utils.seed_everything(303)
            # Evaluation ordering allows checking identical ego/timestamps on
            # all splits; training uses these same filtered scene folders.
            dataset = build_dataset(config, visualize=True, train=False)
            signature = frames(dataset)
            if signatures is None:
                signatures = signature
            if signature != signatures:
                raise AssertionError('Frame/ego mismatch: ' + name + '/' + split)
            print('FRAME PARITY PASS', split, name, len(dataset))
        if split == 'test':
            reference['validate_dir'] = reference['test_dir']
            seed_utils.seed_everything(303)
            heal = build_dataset(reference, visualize=True, train=False)
            if frames(heal) != signatures or len(heal) != len(dataset):
                raise AssertionError('HEAL test frames differ')
            for index in range(min(args.samples, len(dataset))):
                ours = dataset.collate_batch_test([dataset[index]])
                theirs = heal.collate_batch_test([heal[index]])
                gt = dataset.post_processor.generate_gt_bbx({'ego': ours['ego']['baseline_gt']}).numpy()
                gt_ref = heal.post_processor.generate_gt_bbx(theirs).numpy()
                def ordered(boxes):
                    flat = boxes.reshape(-1, 24)
                    return flat[np.lexsort(flat.T[::-1])]
                np.testing.assert_allclose(ordered(gt), ordered(gt_ref), rtol=1e-5, atol=1e-4)
                print('HEAL GT PARITY PASS', index, len(gt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('assignments')
    audit = commands.add_parser('audit-data')
    audit.add_argument('--samples', type=int, default=3)
    prep = commands.add_parser('prepare')
    prep.add_argument('--modality', choices=NAMES, required=True)
    merging = commands.add_parser('merge')
    for modality in NAMES:
        merging.add_argument('--' + modality, default=str(ROOT / 'single' / NAMES[modality]))
    merging.add_argument('--output_dir', default=str(ROOT / 'final_infer'))
    inference = commands.add_parser('infer')
    inference.add_argument('--model_dir', default=str(ROOT / 'final_infer'))
    inference.add_argument('--fusion_method', choices=('no', 'late'), required=True)
    inference.add_argument('--use_cav', type=int)
    inference.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'assignments':
        assignments()
    elif args.command == 'audit-data':
        audit_data(args)
    elif args.command == 'prepare':
        prepare(args.modality)
    elif args.command == 'merge':
        merge(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
