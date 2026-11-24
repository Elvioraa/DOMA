"""CPU-only DOMA-R tests using synthetic BEV maps, boxes and poses."""
import itertools
import unittest
from unittest.mock import Mock

import numpy as np
import torch
import yaml

from opencood.models.sub_modules.doma_box_coder import boxes_hwl_to_corners_3d
from opencood.models.sub_modules.doma_object import (predict_scene_residuals,
                                                     refine_doma_detections,
                                                     uniform_geometry_consensus)
from opencood.models.sub_modules.domar_plugins import DOMARPlugins, freeze_backbone, plugin_loss
from opencood.tools.check_doma_functional import YAML_ROOT, _Holder
from opencood.tools.train_domar_plugin import frozen_candidates
from opencood.utils.pose_utils import add_noise_data_dict


class DOMARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.heads = DOMARPlugins(128, 32, 5)
        self.boxes = torch.tensor([[0., 0., -1., 1.5, 1.6, 4., 0.],
                                   [8., 3., -1., 1.5, 1.6, 4., .2]])

    def test_masks_and_identity(self):
        valid = torch.tensor([[True, False, True], [False, False, False]])
        r = torch.randn(2, 3, 8)
        inputs = (torch.randn(2, 3, 128), torch.randn(2, 3, 32),
                  torch.randn(2, 3, 4), r, valid)
        residual, q, weights = self.heads(inputs)
        expected, _ = uniform_geometry_consensus(r, valid)
        self.assertLess((residual - expected).abs().max(), 1e-6)
        self.assertTrue(torch.equal(weights, torch.tensor([[.5, 0., .5], [0., 0., 0.]])))
        scores = torch.tensor([0., .72])
        self.assertTrue(torch.equal(self.heads.calibration.blend(scores, q), scores))
        self.assertTrue(torch.equal(q, torch.full_like(q, .5)))
        self.assertLess(sum(p.numel() for p in self.heads.parameters()), 100000)

    def scene(self, modalities):
        with (YAML_ROOT / 'V2' / ('stage1/m1.yaml' if modalities[1] == 'm1'
                                   else 'final_infer/m1m2m3m4.yaml')).open() as f:
            model = _Holder(yaml.safe_load(f)['model']['args'])
        with torch.no_grad():
            model.doma_shared_object_refiner.network[-1].weight.normal_(std=.01)
        detail = torch.randn(2, 64, 32, 32)
        return model, dict(agent_features=detail, agent_support=torch.ones(2, 1, 32, 32),
                           agent_modalities=modalities, context_agent_features=torch.randn(2, 128, 16, 16),
                           context_agent_support=torch.ones(2, 1, 16, 16),
                           agent_positions=torch.tensor([[0., 0.], [4., 1.]]))

    def test_noisy_stage15_backward(self):
        model, scene = self.scene(('m1', 'm1'))
        freeze_backbone(model)
        poses = {i: {'ego': i == 0, 'params': {'lidar_pose': np.zeros(6)}} for i in range(2)}
        noisy = add_noise_data_dict(poses, {'eval_pose_noise': (.4, .4, 303)}, 0)
        scene['agent_positions'][1] += torch.tensor(noisy[1]['params']['lidar_pose'][:2])
        with torch.no_grad():
            result = predict_scene_residuals(model, scene, self.boxes, collect_domar=True)
        residual, q, _ = self.heads(result['domar_inputs'])
        loss, target = plugin_loss(self.boxes, self.boxes[:1], residual, q, result['any_valid'],
                                   'sin_cos_centered', 1.)
        self.assertEqual(target[1].item(), 0.)
        self.assertGreater(target[0].item(), .3)
        loss.backward()
        for head in (self.heads.quality, self.heads.calibration):
            self.assertGreater(head.net[-1].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_detector_top64_and_empty_frame(self):
        model, scene = self.scene(('m1', 'm1'))
        freeze_backbone(model)
        def frozen_forward(ego):
            self.assertFalse(torch.is_grad_enabled())
            return {'doma_context': {'scenes': [scene]}}
        model.forward = frozen_forward
        boxes = self.boxes[:1].repeat(70, 1)
        boxes[:, 0] = torch.arange(70) / 10.
        scores = torch.linspace(.1, .9, len(boxes))
        processor = Mock()
        processor.post_process.return_value = boxes_hwl_to_corners_3d(boxes), scores
        ego = {'anchor_box': torch.zeros(1), 'agent_modality_list': ['m1', 'm1'],
               'object_bbx_center': self.boxes[None], 'object_bbx_mask': torch.tensor([[1, 0]])}
        inputs, selected, confidence, gt = frozen_candidates(model, ego, processor)
        self.assertTrue(torch.allclose(selected, boxes.flip(0)[:64], atol=1e-6))
        self.assertTrue(torch.equal(confidence, scores.flip(0)[:64]))
        self.assertTrue(torch.equal(gt, self.boxes[:1]))
        self.assertTrue(all(not value.requires_grad for value in inputs))
        processor.post_process.return_value = None, None
        self.assertIsNone(frozen_candidates(model, ego, processor))

    def test_heterogeneous_assembly_and_ablations(self):
        model, scene = self.scene(('m1', 'm2'))
        model.eval()
        context = {'scenes': [scene]}
        corners = boxes_hwl_to_corners_3d(self.boxes)
        scores = torch.tensor([.9, .7])
        base, _ = refine_doma_detections(model, corners, scores, context)
        model.domar_plugins = DOMARPlugins.from_model(model, 5)
        model.domar_plugins.load_state_dict(self.heads.state_dict(), strict=True)
        for w, c in itertools.product((False, True), repeat=2):
            model.domar_quality, model.domar_calibration = w, c
            model.domar_gamma_override = None
            boxes, conf = refine_doma_detections(model, corners, scores, context)
            self.assertLess((boxes - base).abs().max(), 1e-6)
            self.assertTrue(torch.equal(conf, scores))
            model.domar_gamma_override = 1.
            _, changed = refine_doma_detections(model, corners, scores, context)
            self.assertTrue(torch.allclose(changed, torch.full_like(scores, .5) if c else scores))


if __name__ == '__main__':
    unittest.main()
