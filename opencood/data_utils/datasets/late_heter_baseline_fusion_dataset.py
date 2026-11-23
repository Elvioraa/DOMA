"""Independent V2XSet detectors with the HEAL ordered evaluation GT protocol."""

import numpy as np
import torch

from opencood.data_utils.datasets.late_heter_fusion_dataset import getLateheterFusionDataset


def _preserve_cav_order(cav_list, scenario_name):
    # BaseDataset already sorts validation/test CAVs and rotates infrastructure
    # to the end. Mapping every CAV to one detector must not select a new ego.
    return cav_list


def getLateheterbaselineFusionDataset(cls):
    class LateHeterBaselineDataset(getLateheterFusionDataset(cls)):
        def __init__(self, params, visualize, train=True):
            if (params['fusion']['dataset'] != 'v2xset'
                    or params['label_type'] != 'lidar'
                    or params.get('v2xset_camera_compatible') is not True):
                raise ValueError('Baseline requires V2XSet, LiDAR GT and camera-compatible scenes')
            super().__init__(params, visualize, train)
            # This is the function actually called by get_item_single_car,
            # including camera training and validation-loss supervision.
            self.generate_object_center_single = self.generate_object_center_lidar

        def reinitialize(self):
            if hasattr(self, 'adaptor'):
                self.adaptor.reorder_cav_list = _preserve_cav_order
            super().reinitialize()

        def get_item_test(self, base_data_dict, idx):
            processed = super().get_item_test(base_data_dict, idx)
            ego = next(c for c in base_data_dict.values() if c['ego'])
            centers, ids = [], []
            # Match HEAL infer: all communication-eligible members contribute GT
            # in the clean ego frame, regardless of the prediction member count.
            for cav in base_data_dict.values():
                distance = np.linalg.norm(np.asarray(cav['params']['lidar_pose'][:2])
                                          - np.asarray(ego['params']['lidar_pose'][:2]))
                if distance > self.params['comm_range']:
                    continue
                boxes, mask, object_ids = self.generate_object_center_lidar(
                    [cav], ego['params']['lidar_pose_clean'])
                centers.append(boxes[mask == 1])
                ids.extend(object_ids)
            unique = [ids.index(x) for x in set(ids)]
            boxes = np.vstack(centers)[unique]
            limit = self.params['postprocess']['max_num']
            if len(boxes) > limit:
                raise ValueError('GT exceeds max_num; refusing to truncate evaluation GT')
            padded = np.zeros((limit, 7))
            mask = np.zeros(limit)
            padded[:len(boxes)] = boxes
            mask[:len(boxes)] = 1
            processed['ego']['baseline_gt'] = {
                'object_bbx_center': torch.from_numpy(padded),
                'object_bbx_mask': torch.from_numpy(mask),
                'object_ids': [ids[i] for i in unique],
                'transformation_matrix_clean': torch.eye(4),
            }
            return processed

        def collate_batch_test(self, batch):
            output = super().collate_batch_test(batch)
            output['ego']['baseline_gt'] = batch[0]['ego']['baseline_gt']
            return output

        def post_process(self, data_dict, output_dict):
            boxes, scores = self.post_processor.post_process(data_dict, output_dict)
            gt = self.post_processor.generate_gt_bbx({'ego': data_dict['ego']['baseline_gt']})
            return boxes, scores, gt

        def post_process_no_fusion(self, data_dict, output_dict_ego):
            boxes, scores = self.post_processor.post_process(
                {'ego': data_dict['ego']}, output_dict_ego)
            gt = self.post_processor.generate_gt_bbx({'ego': data_dict['ego']['baseline_gt']})
            return boxes, scores, gt

    return LateHeterBaselineDataset
