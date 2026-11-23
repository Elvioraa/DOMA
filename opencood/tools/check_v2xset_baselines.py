"""Read-only static/CPU acceptance checks; no dataset, training or inference."""

import argparse
import ast
import copy
import io
from pathlib import Path

import yaml

from opencood.tools.v2xset_baselines import CONFIG, NAMES, REPO, merge_states


def check_configs():
    originals = ('m1_pointpillar_pretrain.yaml', 'm2_LSSeff48_pretrain.yaml',
                 'm3_SECOND32_pretrain.yaml', 'm4_LSSres48_pretrain.yaml')
    for modality, original in zip(NAMES, originals):
        config = yaml.safe_load((CONFIG / (NAMES[modality] + '.yaml')).read_text())
        reference = yaml.safe_load((REPO / 'opencood/hypes_yaml/v2xset/Single' / original).read_text())
        assert config['optimizer'] == reference['optimizer']
        assert config['lr_scheduler'] == reference['lr_scheduler']
        assert config['train_params']['batch_size'] == reference['train_params']['batch_size']
        assert config['model'] == reference['model']
        assert config['baseline_protocol']['output_dir'].endswith('/single/' + NAMES[modality])
    for path in CONFIG.glob('*.yaml'):
        config = yaml.safe_load(path.read_text())
        assert config['seed'] == 303 and config['comm_range'] == 180
        assert config['label_type'] == 'lidar' and config['v2xset_camera_compatible'] is True
        assert config['fusion']['core_method'] == 'lateheterbaseline'
        assert config['fusion']['dataset'] == 'v2xset'
        for key, split in (('root_dir', 'train'), ('validate_dir', 'validate'), ('test_dir', 'test')):
            assert config[key] == 'dataset/V2XSET/' + split
        for key, value in (('epoches', 30), ('eval_freq', 2), ('save_freq', 2), ('max_cav', 5)):
            assert config['train_params'][key] == value
        assert config['train_params']['run_inference_after_training'] is False
    final = yaml.safe_load((CONFIG / 'final.yaml').read_text())
    assert final['postprocess']['gt_range'] == [-204.8, -102.4, -3, 204.8, 102.4, 1]
    assert final['heter']['ego_modality'] == 'm1'
    assert final['heter']['mapping_dict'] == {m: m for m in NAMES}
    assert final['heter']['assignment_path'].endswith('v2xset_4modality_in_order.json')
    assert final['input_source'] == ['lidar', 'camera']
    assert final['baseline_protocol']['output_dir'].endswith('/final_infer')
    for modality in ('m1', 'm3'):
        assert final['model']['args'][modality]['encoder_args']['lidar_range'] == final['postprocess']['gt_range']
    print('PASS: five YAMLs, unchanged optimizer/scheduler/batch/sensor models, final protocol')


def check_gt():
    """Exercise the actual factory with a small parent stub, without CUDA extensions."""
    import numpy as np
    import torch

    class BaseStub:
        def __init__(self, params, visualize, train):
            self.load_camera_file = params.get('camera', False)
            self.scenario_folders = ['kept', '2021_09_09_13_20_58']

        def reinitialize(self):
            pass

    base_source = REPO / 'opencood/data_utils/datasets/basedataset/v2xset_basedataset.py'
    base_tree = ast.parse(base_source.read_text())
    base_tree.body = [n for n in base_tree.body if isinstance(n, ast.ClassDef)]
    base_namespace = {'OPV2VBaseDataset': BaseStub}
    exec(compile(base_tree, str(base_source), 'exec'), base_namespace)
    base = base_namespace['V2XSETBaseDataset']
    assert len(base({}, False).scenario_folders) == 2  # default remains unchanged
    for options in ({'camera': True}, {'v2xset_camera_compatible': True}):
        assert base(options, False).scenario_folders == ['kept']

    class Parent:
        def __init__(self, params, visualize, train):
            self.params = params

        def get_item_test(self, data, idx):
            return {'ego': {}}

        def generate_object_center_lidar(self, cavs, pose):
            cav = cavs[0]
            assert pose == [0, 0]
            boxes = np.zeros((2, 7))
            boxes[0, 0] = cav['object']
            return boxes, np.array([1, 0]), [cav['object']]

    source = REPO / 'opencood/data_utils/datasets/late_heter_baseline_fusion_dataset.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace = {'np': np, 'torch': torch, 'getLateheterFusionDataset': lambda cls: Parent}
    exec(compile(tree, str(source), 'exec'), namespace)
    dataset_class = namespace['getLateheterbaselineFusionDataset'](Parent)
    params = {'fusion': {'dataset': 'v2xset'}, 'label_type': 'lidar',
              'v2xset_camera_compatible': True, 'comm_range': 180,
              'postprocess': {'max_num': 100}}
    dataset = dataset_class(params, True, False)
    assert namespace['_preserve_cav_order'](['10', '20', '-1'], 'scene') == ['10', '20', '-1']
    data = {}
    for i, distance in enumerate((0, 20, 40, 200)):
        data[i] = {'ego': i == 0, 'object': (1 if i == 2 else i),
                   'params': {'lidar_pose': [distance, 0], 'lidar_pose_clean': [0, 0]}}
    result = dataset.get_item_test(data, 0)['ego']['baseline_gt']
    assert set(result['object_ids']) == {0, 1}  # duplicate removed, distant member excluded
    assert int(result['object_bbx_mask'].sum()) == 2
    assert torch.equal(result['transformation_matrix_clean'], torch.eye(4))
    assert dataset.generate_object_center_single == dataset.generate_object_center_lidar
    invalid = copy.deepcopy(params)
    invalid['label_type'] = 'camera'
    try:
        dataset_class(invalid, True, False)
    except ValueError:
        pass
    else:
        raise AssertionError('Camera GT was accepted')
    print('PASS: opt-in scene filtering/default parity, GT membership/deduplication/clean-frame and LiDAR binding (stub parents)')


def check_merge():
    import torch
    target = {'cls_head_m1.weight': torch.ones(2, 3), 'shared': torch.ones(1)}
    sources = {'m1': dict(target), 'm2': {'shared': torch.ones(1)}}
    assert set(merge_states(target, sources)) == set(target)
    bad = []
    item = copy.deepcopy(sources); del item['m1']['cls_head_m1.weight']; bad.append(item)
    item = copy.deepcopy(sources); item['m1']['cls_head_m1.weight'] = torch.ones(3, 2); bad.append(item)
    item = copy.deepcopy(sources); item['m2']['shared'] = torch.zeros(1); bad.append(item)
    item = copy.deepcopy(sources); item['m2']['cls_head_m1.weight'] = torch.ones(2, 3); bad.append(item)
    item = copy.deepcopy(sources); item['m1']['extra'] = torch.zeros(1); bad.append(item)
    for item in bad:
        try:
            merge_states(target, item)
        except ValueError:
            continue
        raise AssertionError('Invalid merge accepted')
    buffer = io.BytesIO()
    torch.save(merge_states(target, sources), buffer)
    buffer.seek(0)
    assert set(torch.load(buffer, map_location='cpu', weights_only=True)) == set(target)
    print('PASS: merge rejects missing/shape/owner/extra/shared conflicts; serialization roundtrip')


def check_models():
    import gc
    from opencood.tools.v2xset_baselines import build, read_config
    final = build(read_config(CONFIG / 'final.yaml'))
    target = final.state_dict()
    sources = {}
    for modality in NAMES:
        single = build(read_config(CONFIG / (NAMES[modality] + '.yaml')))
        state = single.state_dict()
        assert all(k in target and v.shape == target[k].shape for k, v in state.items())
        sources[modality] = {k: target[k] for k in state}
        single.load_state_dict(sources[modality], strict=True)
        print('PASS: model construction and strict per-modality CPU parameter load', modality,
              sum(p.numel() for p in single.parameters()))
        del single, state
        gc.collect()
    final.load_state_dict(merge_states(target, sources), strict=True)
    print('PASS: Final construction and strict four-modality merge (synthetic weights)')
    print('NOTE: original camera constructor allocates its frustum on CUDA; no forward is run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Construct all five models; original camera constructor needs CUDA')
    args = parser.parse_args()
    check_configs()
    check_gt()
    check_merge()
    if args.models:
        check_models()
    print('UNVERIFIED: real dataset/GT parity, trained checkpoints, CUDA training and AP')


if __name__ == '__main__':
    main()
