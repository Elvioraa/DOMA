"""Detached GT-jitter proposal sampling for canonical DOMA V1-V3."""

import math

import torch
import torch.nn as nn

from opencood.models.sub_modules.doma_box_coder import wrap_angle


class DOMATrainingProposalSampler(nn.Module):
    """Create detached hwl proposals and matching GT targets using torch RNG."""

    def __init__(self, config, max_proposals):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("training_proposals must be a mapping")
        if config.get("source") != "gt_jitter":
            raise ValueError("DOMA V1-V3 proposal source must be gt_jitter")
        self.include_gt = _require_bool(config, "include_gt")
        self.jitters_per_gt = _nonnegative_int(
            config.get("jitters_per_gt"), "jitters_per_gt"
        )
        self.center_xy_std_rel = _nonnegative_real(
            config.get("center_xy_std_rel"), "center_xy_std_rel"
        )
        self.center_z_std_rel = _nonnegative_real(
            config.get("center_z_std_rel"), "center_z_std_rel"
        )
        self.log_size_std = _nonnegative_real(
            config.get("log_size_std"), "log_size_std"
        )
        self.yaw_std_rad = math.radians(
            _nonnegative_real(config.get("yaw_std_deg"), "yaw_std_deg")
        )
        self.max_proposals = _positive_int(max_proposals, "max_proposals")
        if not self.include_gt and self.jitters_per_gt == 0:
            raise ValueError("proposal sampler would generate no proposals")

    def forward(self, gt_boxes, gt_mask, with_jitter=True):
        """Return ``(proposals, targets)`` in ``[x,y,z,h,w,l,yaw]`` order."""
        if type(with_jitter) is not bool:
            raise TypeError("with_jitter must be bool")
        if not torch.is_tensor(gt_boxes) or gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
            raise ValueError("gt_boxes must have shape [M,7] in hwl order")
        if not torch.is_floating_point(gt_boxes):
            raise TypeError("gt_boxes must use a floating-point dtype")
        if not torch.is_tensor(gt_mask) or gt_mask.ndim != 1:
            raise ValueError("gt_mask must have shape [M]")
        if gt_mask.shape[0] != gt_boxes.shape[0] or gt_mask.device != gt_boxes.device:
            raise ValueError("gt_mask must match gt_boxes on length and device")
        valid_gt = gt_boxes[gt_mask.to(dtype=torch.bool)]
        if valid_gt.numel() == 0:
            empty = gt_boxes.new_empty((0, 7))
            return empty.detach(), empty.detach()
        if not bool((valid_gt[:, 3:6] > 0).all()):
            raise ValueError("valid GT height, width, and length must be positive")
        if not bool(torch.isfinite(valid_gt).all()):
            raise ValueError("valid GT boxes must contain only finite values")

        proposal_parts = []
        target_parts = []
        if self.include_gt:
            proposal_parts.append(valid_gt.clone())
            target_parts.append(valid_gt)
        jitter_count = self.jitters_per_gt if with_jitter else 0
        for _ in range(jitter_count):
            jittered = valid_gt.clone()
            noise = torch.randn_like(valid_gt)
            jittered[:, 0] += noise[:, 0] * valid_gt[:, 5] * self.center_xy_std_rel
            jittered[:, 1] += noise[:, 1] * valid_gt[:, 4] * self.center_xy_std_rel
            jittered[:, 2] += noise[:, 2] * valid_gt[:, 3] * self.center_z_std_rel
            jittered[:, 3:6] *= torch.exp(noise[:, 3:6] * self.log_size_std)
            jittered[:, 6] = wrap_angle(
                valid_gt[:, 6] + noise[:, 6] * self.yaw_std_rad
            )
            proposal_parts.append(jittered)
            target_parts.append(valid_gt)

        proposals = torch.cat(proposal_parts, dim=0)[: self.max_proposals]
        targets = torch.cat(target_parts, dim=0)[: self.max_proposals]
        return proposals.detach(), targets.detach()


def _require_bool(config, key):
    value = config.get(key)
    if type(value) is not bool:
        raise TypeError("training_proposals.%s must be bool" % key)
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _nonnegative_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("%s must be a non-negative real number" % name)
    return float(value)
