"""Synthetic DOMA-R tests and an optional native CUDA vs Shapely comparison."""
import itertools
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from opencood.models.sub_modules import domar_iou
from opencood.models.sub_modules.doma_box_coder import (boxes_hwl_to_corners_3d,
                                                        pairwise_rotated_bev_iou_hwl)
from opencood.models.sub_modules.doma_object import (predict_scene_residuals,
                                                     refine_doma_detections,
                                                     uniform_geometry_consensus)
from opencood.models.sub_modules.domar_plugins import DOMARPlugins, freeze_backbone, plugin_loss
from opencood.tools.check_doma_functional import YAML_ROOT, _Holder
from opencood.tools.train_domar_plugin import CollateScenes, frozen_candidates, parse_args, run_epoch
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


class Stage15TrainingTest(unittest.TestCase):
    @staticmethod
    def rotated_boxes():
        a = torch.tensor([[0., 0., -1., 1.5, 1.6, 4., .47],
                          [1.3, -.7, 3., .8, 1.1, 3.4, -.82],
                          [20., 0., 0., 2., 3., 6., 1.1]])
        b = a[[0, 0, 1, 0, 0]].clone()
        b[0, 2:4] = torch.tensor([10., 2.5])  # disjoint in height, identical in BEV
        b[1, :2] += 2 * torch.stack([a[0, 6].cos(), a[0, 6].sin()])
        b[2, :2] += torch.tensor([-.4, .6])
        b[3, 6] += torch.pi
        b[4, :2] += torch.tensor([.001, -.003])  # inside PCDet's unscaled corner margin
        return a, b

    def test_shapely_iou_matches_existing_box_convention(self):
        a, b = self.rotated_boxes()
        with patch.object(domar_iou, 'cuda_iou_ops', side_effect=AssertionError('CPU path')):
            iou = domar_iou.rotated_bev_iou(a, b)
        reference = pairwise_rotated_bev_iou_hwl(a, b)
        torch.testing.assert_close(iou, reference, atol=1e-4, rtol=0)
        torch.testing.assert_close(iou[0, [0, 1, 3]], torch.tensor([1., 1/3, 1.]),
                                   atol=1e-4, rtol=0)
        self.assertFalse(iou[2].any())
        self.assertEqual(domar_iou.rotated_bev_iou(a, b[:0]).shape, (3, 0))
        self.assertEqual(domar_iou.rotated_bev_iou(a[:0], b).shape, (0, 5))

    def test_refined_iou_uses_original_gt_assignment(self):
        p = torch.tensor([[0., 0., 0., 1., 1.5, 4., 0.]])
        gt = p.repeat(2, 1)
        gt[1, 0] = 8
        r = torch.zeros(1, 8)
        r[0, 0] = 2  # Move exactly onto the other GT.
        _, target = plugin_loss(p, gt, r, torch.ones(1), torch.tensor([True]),
                                'sin_cos_centered', 1.)
        self.assertEqual(target.item(), 0.)

    def test_nms_preserves_top1000_and_score_ties(self):
        boxes = self.rotated_boxes()[0][:1].repeat(1001, 1)
        boxes[-1, 0] = 20  # unique detection below the pre-NMS cut
        corners = boxes_hwl_to_corners_3d(boxes)
        scores = torch.ones(len(boxes))
        scores[-1] = .1
        expected = [scores.numpy().argsort()[::-1][0]]
        self.assertEqual(domar_iou.proposal_nms(corners, scores, .3).tolist(), expected)
        self.assertEqual(domar_iou.proposal_nms(corners[:0], scores[:0], .3).tolist(), [])

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a CUDA device')
    def test_cuda_iou_and_nms_match_shapely(self):
        if domar_iou.cuda_iou_ops() is None:
            self.skipTest('requires compiled PCDet CUDA ops')
        a, b = self.rotated_boxes()
        expected = domar_iou.rotated_bev_iou(a, b)
        with patch.object(domar_iou, '_shapely_bev_iou', side_effect=AssertionError('CUDA path')):
            actual = domar_iou.rotated_bev_iou(a.cuda(), b.cuda())
        self.assertTrue(actual.is_cuda)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=0)
        with patch.object(domar_iou, 'cuda_iou_ops', return_value=None):
            fallback = domar_iou.rotated_bev_iou(a.cuda(), b.cuda())
        self.assertEqual(fallback.device, actual.device)
        torch.testing.assert_close(fallback.cpu(), expected, atol=1e-4, rtol=0)
        corners = boxes_hwl_to_corners_3d(torch.cat([a, b]))
        scores = torch.tensor([.95, .8, .7, .95, .65, .55, .4, .6])
        keep = domar_iou.proposal_nms(corners, scores, .3)
        kept_cuda = domar_iou.proposal_nms(corners.cuda(), scores.cuda(), .3)
        self.assertEqual(kept_cuda.tolist(), keep.tolist())

    def test_trainer_cli(self):
        args = ['--model_dir', 'm1', '--checkpoint', 'm1.pth', '--output_dir', 'stage15']
        defaults = parse_args(args)
        self.assertEqual((defaults.log_interval, defaults.num_workers), (50, 4))
        opt = parse_args(args + ['--log_interval', '1', '--num_workers', '0'])
        self.assertEqual((opt.log_interval, opt.num_workers), (1, 0))
        self.assertEqual(parse_args(args + ['--workers', '2']).num_workers, 2)
        for option, value in [('--log_interval', '0'), ('--num_workers', '-1')]:
            with self.assertRaises(SystemExit):
                parse_args(args + [option, value])

    def test_progress_counts_empty_frames_and_flushes_both_splits(self):
        class Frames(list):
            post_processor = None

            @staticmethod
            def collate_batch_train(samples):
                return {'ego': samples[0]}

        frames = Frames([None, {}, {}])
        loader = torch.utils.data.DataLoader(frames, batch_size=2, collate_fn=CollateScenes(frames))
        model = Mock(doma_config={'refiner': {'yaw_mode': 'sin_cos_centered'}})
        heads = DOMARPlugins(4, 3, 2)
        optimizer = torch.optim.AdamW(heads.parameters(), lr=.001)
        box = self.rotated_boxes()[0][:1]
        inputs = (torch.ones(1, 2, 4), torch.ones(1, 2, 3), torch.ones(1, 2, 4),
                  torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.bool))
        data = inputs, box, torch.tensor([.8]), box
        for split, opt, interval, steps in [('Train', optimizer, 1, [1, 2, 3]),
                                            ('Val', None, 2, [2, 3])]:
            with patch('opencood.tools.train_domar_plugin.frozen_candidates', side_effect=[data, None]), \
                    patch('builtins.print') as logged:
                run_epoch(model, heads, loader, opt, 1., torch.device('cpu'),
                          epoch=2, epochs=8, log_interval=interval, lr=.001)
            self.assertEqual(logged.call_count, len(steps))
            for call, step in zip(logged.call_args_list, steps):
                self.assertIn(f'[Stage1.5][{split}][Epoch 2/8][{step}/3]', call.args[0])
                self.assertIn('lr=1.00e-03 elapsed=', call.args[0])
                self.assertIn(' eta=', call.args[0])
                self.assertTrue(call.kwargs['flush'])
            self.assertIn('loss=n/a', logged.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
