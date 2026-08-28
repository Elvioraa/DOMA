"""DOMA-only PyramidFusion extension exposing pre-fusion pyramid features."""

import torch

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion, weighted_fuse


class DOMAPyramidFusion(PyramidFusion):
    """Preserve Official HEAL behavior unless DOMA requests Context features."""

    def forward_single(self, spatial_features, return_pre_fusion_features=False):
        if not return_pre_fusion_features:
            return super().forward_single(spatial_features)
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for index in range(self.num_levels):
            occ_map_list.append(
                getattr(self, "single_head_%d" % index)(feature_list[index])
            )
        final_feature = self.decode_multiscale_feature(feature_list)
        return final_feature, occ_map_list, tuple(feature_list)

    def forward_collab(
        self,
        spatial_features,
        record_len,
        affine_matrix,
        agent_modality_list=None,
        cam_crop_info=None,
        return_pre_fusion_features=False,
    ):
        if not return_pre_fusion_features:
            return super().forward_collab(
                spatial_features,
                record_len,
                affine_matrix,
                agent_modality_list,
                cam_crop_info,
            )

        crop_mask_flag = cam_crop_info is not None and len(cam_crop_info) > 0
        if crop_mask_flag:
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [
                    1 if value == cam_modality else 0
                    for value in agent_modality_list
                ]
                cam_agent_mask_dict[cam_modality] = torch.tensor(
                    mask_list, dtype=torch.bool
                )

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for index in range(self.num_levels):
            occ_map = getattr(self, "single_head_%d" % index)(feature_list[index])
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, height, width = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_height = (
                        height
                        / cam_crop_info[cam_modality][
                            "crop_ratio_H_%s" % cam_modality
                        ]
                        - 4
                    )
                    crop_width = (
                        width
                        / cam_crop_info[cam_modality][
                            "crop_ratio_W_%s" % cam_modality
                        ]
                        - 4
                    )
                    start_h = int(height // 2 - crop_height // 2)
                    end_h = int(height // 2 + crop_height // 2)
                    start_w = int(width // 2 - crop_width // 2)
                    end_w = int(width // 2 + crop_width // 2)
                    mask = cam_agent_mask_dict[cam_modality].to(occ_map.device)
                    cam_crop_mask[mask, :, start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[mask] = 1 - cam_crop_mask[mask]
                score = score * cam_crop_mask

            fused_feature_list.append(
                weighted_fuse(
                    feature_list[index],
                    score,
                    record_len,
                    affine_matrix,
                    self.align_corners,
                )
            )
        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        return fused_feature, occ_map_list, tuple(feature_list)
