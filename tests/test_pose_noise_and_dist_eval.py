"""CPU-only integration of the production pose transforms and AP matcher."""
import argparse
import copy
import math
import unittest

import numpy as np
import torch

from opencood.utils import eval_extensions, pose_utils
from opencood.utils.transformation_utils import get_pairwise_transformation


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        eval_extensions.add_arguments(parser)
        self.options = parser.parse_args([])
        self.options.note = ''
        poses = torch.randn(3, 6, generator=torch.Generator().manual_seed(41))
        self.frame = {i: {'ego': i == 1, 'params': {'lidar_pose': pose.tolist()}}
                      for i, pose in enumerate(poses)}

    def test_zero_noise(self):
        hypes = {'noise_setting': {'add_noise': False}}
        eval_extensions.configure_noise(hypes, self.options)
        self.assertEqual(hypes, {'noise_setting': {'add_noise': False}})
        self.assertEqual(self.options.note, '')
        self.assertFalse(self.options.eval_distance_buckets)
        original = get_pairwise_transformation(self.frame, 3, False)
        result = pose_utils.add_noise_data_dict(copy.deepcopy(self.frame), hypes['noise_setting'], 7)
        actual = get_pairwise_transformation(result, 3, False)
        self.assertLess(np.abs(original - actual).max(), 1e-6)

    def test_seeded_noise(self):
        self.options.pose_noise_xy = self.options.pose_noise_yaw = 0.4
        hypes = {}
        eval_extensions.configure_noise(hypes, self.options)
        def sample(frame, idx):
            return pose_utils.add_noise_data_dict(copy.deepcopy(frame), hypes['noise_setting'], idx)
        cached = copy.deepcopy(self.frame)
        first = pose_utils.add_noise_data_dict(dict(cached), hypes['noise_setting'], 7)
        self.assertEqual(cached, self.frame)
        repeat = sample(dict(reversed(list(self.frame.items()))), 7)
        other = sample(self.frame, 8)
        for i, content in self.frame.items():
            clean = np.asarray(content['params']['lidar_pose'])
            pose = first[i]['params']['lidar_pose']
            np.testing.assert_array_equal(first[i]['params']['lidar_pose_clean'], clean)
            np.testing.assert_array_equal(repeat[i]['params']['lidar_pose'], pose)
            if content['ego']:
                np.testing.assert_array_equal(pose, clean)
            else:
                np.testing.assert_array_equal(pose[[2, 3, 5]], clean[[2, 3, 5]])
                self.assertTrue(np.all(pose[[0, 1, 4]] != clean[[0, 1, 4]]))
                self.assertFalse(np.array_equal(pose, other[i]['params']['lidar_pose']))
        self.assertFalse(np.array_equal(get_pairwise_transformation(first, 3, False),
                                        get_pairwise_transformation(self.frame, 3, False)))

    def test_distance_ap(self):
        corners = torch.tensor([[-1., -1., -1.], [1., -1., -1.],
                                [1., 1., -1.], [-1., 1., -1.]])
        corners = torch.cat([corners, corners + torch.tensor([0., 0., 2.])])
        centres = torch.tensor([[0., 0., 0.], [18., 24., 0.], [30., 0., 40.]])
        gt = corners + centres[:, None, :]
        predictions = gt.clone()
        predictions[1, :, 0] += 0.5  # IoU 0.6
        counts = eval_extensions.DistanceBucketAP()
        counts.update(predictions[:2], torch.tensor([0.9, 0.8]), gt)
        counts.update(None, None, gt[:0])
        result = counts.summary()
        self.assertEqual(list(result.values()), [{0.5: 1., 0.7: 1.}, {0.5: 1., 0.7: 0.},
                                                 {0.5: 0., 0.7: 0.}])
        empty = eval_extensions.DistanceBucketAP()
        empty.update(predictions, torch.ones(3), gt[:0])
        self.assertTrue(all(math.isnan(ap) for pair in empty.summary().values() for ap in pair.values()))


if __name__ == '__main__':
    unittest.main()
