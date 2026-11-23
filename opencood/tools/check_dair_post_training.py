"""CPU/Mock checks only; no DAIR media, GPU, or measured AP claims."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import dair_post_training as post, validation_detection as vd
from opencood.tools.dair_protocol import digest


class Dataset:
    def __len__(self):
        return 1789


def expect_failure(call):
    try:
        call()
    except RuntimeError as exc:
        assert 'Training completed, but DAIR bestval AP FAILED' in str(exc)
    else:
        raise AssertionError('Expected explicit post-training failure')


def main():
    assert not post.is_enabled({})
    assert not post.run_bestval_ap({}, None, None, None, None, None)
    configs = Path(__file__).resolve().parents[1] / 'hypes_yaml/dairv2x/MoreModality/DAIR_PROTOCOL'
    dataset = Dataset()
    metrics = dict(ap30=.7, ap50=.6, ap70=.5)  # Mock outputs, not measured AP.
    with tempfile.TemporaryDirectory(prefix='check-dair-post-') as temp:
        root = Path(temp)
        val = root / 'val.json'
        val.write_text('[]')
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(7)
        checkpoint = root / 'net_epoch_bestval_at8.pth'
        torch.save(model.state_dict(), checkpoint)
        for name in ('net_epoch1.pth', 'net_epoch30.pth', 'net_epoch_bestval_at12.pth'):
            torch.save({'unrelated': torch.tensor(-1)}, root / name)
        initial_files = {p.name: digest(p) for p in root.iterdir()}
        for method in ('HEAL', 'DOMA'):
            for stage, config in [('stage1', 'stage1/config.yaml'),
                                  ('stage2', 'E1/stage2_m2/config.yaml'),
                                  ('stage2', 'E2/stage2_m2/config.yaml'),
                                  ('stage2', 'E3/stage2_m2/config.yaml')]:
                cfg = load_yaml(str(configs / method / config))
                assert not post.is_enabled(cfg)
                cfg['train_params']['dair_bestval_ap'] = {'enabled': True}
                cfg['validate_dir'] = cfg['test_dir'] = str(val)
                original = copy.deepcopy(cfg)
                def evaluate(actual_model, loader, actual_dataset, device, fusion):
                    assert actual_model is model and actual_dataset is dataset
                    assert fusion == ('intermediate' if stage == 'stage1' else 'late')
                    assert float(model.weight.item()) == 7
                    assert sum(1 for batch in loader if batch is not None) == len(dataset)
                    return metrics
                def verified_load(actual_model, state, directory, selected, effective):
                    assert Path(selected) == checkpoint and effective is cfg
                    actual_model.load_state_dict(state, strict=True)
                with patch.object(vd, 'build_detection_validation_loader', return_value=[{}]*len(dataset)), \
                     patch.object(vd, 'evaluate_detection_ap', side_effect=evaluate), \
                     patch.object(post, 'load_training_state', side_effect=verified_load) as loading:
                    assert post.run_bestval_ap(cfg, root, 8, model, dataset, torch.device('cpu'))
                    assert loading.call_count == 1
                assert cfg == original
        successes=list(root.glob('dair_bestval_ap_*/result.json'))
        assert len(successes)==8
        for path in successes:
            report=json.loads(path.read_text())
            assert report['checkpoint']==str(checkpoint.resolve())
            assert report['checkpoint_sha256']==digest(checkpoint)
            assert report['evaluated_samples']==1789
            assert report['metrics']==metrics
            assert report['effective_config']['postprocess']['anchor_args']['W']==512
        # Exercise real Stage1 strict loading and refusal to fall back to epoch1/last.
        cfg=load_yaml(str(configs/'HEAL/stage1/config.yaml'))
        cfg['train_params']['dair_bestval_ap']={'enabled':True}
        cfg['validate_dir']=cfg['test_dir']=str(val)
        with patch.object(vd,'build_detection_validation_loader',return_value=[{}]*len(dataset)), \
             patch.object(vd,'evaluate_detection_ap',side_effect=evaluate):
            stage='stage1'
            assert post.run_bestval_ap(cfg,root,8,model,dataset,torch.device('cpu'))
            expect_failure(lambda: post.run_bestval_ap(cfg,root,99,model,dataset,'cpu'))
        with patch.object(vd,'build_detection_validation_loader',return_value=[None]*len(dataset)), \
             patch.object(vd,'evaluate_detection_ap',side_effect=lambda m,l,*a: (list(l),metrics)[1]):
            expect_failure(lambda: post.run_bestval_ap(cfg,root,8,model,dataset,'cpu'))
        with patch.object(post,'load_training_state',side_effect=RuntimeError('provenance rejected')):
            expect_failure(lambda: post.run_bestval_ap(cfg,root,8,model,dataset,'cpu'))
        assert len(list(root.glob('dair_bestval_ap_*/failure.json')))==3
        assert all(digest(root/name)==sha for name,sha in initial_files.items())
    print('PASS: CPU/Mock bestval selection, all 8 stage configs, provenance delegation, full-split counting, failure propagation, isolated records, default-off; AP values are mocked')


if __name__ == '__main__':
    main()
