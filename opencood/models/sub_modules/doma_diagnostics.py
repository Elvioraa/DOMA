"""Detached diagnostics for optional DOMA research instrumentation.

This module contains no parameters and is never imported by Official HEAL
models.  Its inference helper compares evaluator-ready post-NMS HEAL boxes
with the final DOMA output without changing either tensor.
"""

import math

import torch

from opencood.models.sub_modules.doma_box_coder import (
    aligned_rotated_bev_iou_hwl,
    corners_3d_to_boxes_hwl,
    pairwise_rotated_bev_iou_hwl,
)


DEFAULT_NEUTRAL_THRESHOLD = 1.0e-6
DOMA_DELTA_IOU_RESULT_KEY = "doma_delta_iou"


@torch.no_grad()
def compute_post_nms_delta_iou(original_boxes, refined_boxes, gt_boxes):
    """Return paired per-proposal Delta-IoU for final inference boxes.

    Every valid original proposal is matched once to the GT box with maximum
    rotated-BEV IoU.  The refined box at the same proposal index is evaluated
    against that fixed GT; it is deliberately not rematched after refinement.
    Empty proposal or GT sets have no defined comparisons and return an empty
    tensor.  Numerically invalid proposal rows are retained as NaN so the
    distribution summary can report them through ``nonfinite_count``.
    """
    if original_boxes is None or refined_boxes is None:
        if original_boxes is not None or refined_boxes is not None:
            raise ValueError(
                "original and refined post-NMS boxes must both be tensors or None"
            )
        return torch.empty((0,), dtype=torch.float32)

    _validate_corner_tensor(original_boxes, "original_boxes")
    _validate_corner_tensor(refined_boxes, "refined_boxes")
    if original_boxes.shape != refined_boxes.shape:
        raise ValueError(
            "original and refined post-NMS boxes must have identical shapes"
        )
    proposal_count = int(original_boxes.shape[0])
    if proposal_count == 0:
        return original_boxes.new_empty((0,), dtype=torch.float32).detach()

    if gt_boxes is None:
        return original_boxes.new_empty((0,), dtype=torch.float32).detach()
    _validate_corner_tensor(gt_boxes, "gt_boxes")
    if int(gt_boxes.shape[0]) == 0:
        return original_boxes.new_empty((0,), dtype=torch.float32).detach()

    device = original_boxes.device
    original = original_boxes.detach().to(device=device, dtype=torch.float32)
    refined = refined_boxes.detach().to(device=device, dtype=torch.float32)
    gt = gt_boxes.detach().to(device=device, dtype=torch.float32)

    original_boxes, original_valid = _safe_corner_rows_to_boxes(original)
    refined_boxes, refined_valid = _safe_corner_rows_to_boxes(refined)
    gt_boxes, gt_valid = _safe_corner_rows_to_boxes(gt)
    delta = original.new_full((proposal_count,), float("nan"))

    valid_gt_indices = gt_valid.nonzero(as_tuple=False).squeeze(-1)
    valid_original_indices = original_valid.nonzero(as_tuple=False).squeeze(-1)
    if not valid_gt_indices.numel() or not valid_original_indices.numel():
        return delta.detach()

    valid_gt_boxes = gt_boxes.index_select(0, valid_gt_indices)
    valid_original_boxes = original_boxes.index_select(0, valid_original_indices)
    original_to_gt_iou = pairwise_rotated_bev_iou_hwl(
        valid_original_boxes, valid_gt_boxes
    )
    original_iou, matched_gt_local_indices = original_to_gt_iou.max(dim=1)

    refined_is_valid = refined_valid.index_select(0, valid_original_indices)
    jointly_valid_local_indices = refined_is_valid.nonzero(
        as_tuple=False
    ).squeeze(-1)
    if not jointly_valid_local_indices.numel():
        return delta.detach()

    jointly_valid_proposal_indices = valid_original_indices.index_select(
        0, jointly_valid_local_indices
    )
    matched_gt_indices = matched_gt_local_indices.index_select(
        0, jointly_valid_local_indices
    )
    fixed_gt_boxes = valid_gt_boxes.index_select(0, matched_gt_indices)
    valid_refined_boxes = refined_boxes.index_select(
        0, jointly_valid_proposal_indices
    )
    refined_iou = aligned_rotated_bev_iou_hwl(
        valid_refined_boxes, fixed_gt_boxes
    )
    delta[jointly_valid_proposal_indices] = (
        refined_iou - original_iou.index_select(0, jointly_valid_local_indices)
    )
    return delta.detach()


class DeltaIoUDiagnosticAccumulator:
    """Collect detached CPU values and summarize one exact global distribution."""

    def __init__(self, neutral_threshold=DEFAULT_NEUTRAL_THRESHOLD):
        self.neutral_threshold = _validate_neutral_threshold(neutral_threshold)
        self._chunks = []

    def update(self, values):
        if not torch.is_tensor(values) or values.ndim != 1:
            raise ValueError("Delta-IoU values must be a one-dimensional tensor")
        self._chunks.append(
            values.detach().to(device="cpu", dtype=torch.float32).clone()
        )

    def values(self):
        if not self._chunks:
            return torch.empty((0,), dtype=torch.float32)
        return torch.cat(self._chunks, dim=0)

    def summary(self):
        return summarize_delta_iou(
            self.values(), neutral_threshold=self.neutral_threshold
        )


def summarize_delta_iou(values, neutral_threshold=DEFAULT_NEUTRAL_THRESHOLD):
    """Compute exact statistics from one concatenated global value tensor."""
    neutral_threshold = _validate_neutral_threshold(neutral_threshold)
    if not torch.is_tensor(values) or values.ndim != 1:
        raise ValueError("Delta-IoU values must be a one-dimensional tensor")

    values = values.detach().to(device="cpu", dtype=torch.float64)
    count = int(values.numel())
    finite = values[torch.isfinite(values)]
    finite_count = int(finite.numel())
    stats = {
        "doma_delta_iou_count": count,
        "doma_delta_iou_finite_count": finite_count,
        "doma_delta_iou_nonfinite_count": count - finite_count,
        "doma_delta_iou_mean": 0.0,
        "doma_delta_iou_std": 0.0,
        "doma_delta_iou_min": 0.0,
        "doma_delta_iou_max": 0.0,
        "doma_delta_iou_p10": 0.0,
        "doma_delta_iou_p25": 0.0,
        "doma_delta_iou_p50": 0.0,
        "doma_delta_iou_p75": 0.0,
        "doma_delta_iou_p90": 0.0,
        "doma_delta_iou_improve_ratio": 0.0,
        "doma_delta_iou_worsen_ratio": 0.0,
        "doma_delta_iou_neutral_ratio": 0.0,
        "doma_delta_iou_abs_gt_0_01_ratio": 0.0,
        "doma_delta_iou_abs_gt_0_05_ratio": 0.0,
        "doma_delta_iou_neutral_threshold": neutral_threshold,
    }
    if finite_count == 0:
        return stats

    quantiles = torch.quantile(
        finite, finite.new_tensor((0.10, 0.25, 0.50, 0.75, 0.90))
    )
    absolute = finite.abs()
    stats.update(
        {
            "doma_delta_iou_mean": float(finite.mean().item()),
            "doma_delta_iou_std": float(finite.std(unbiased=False).item()),
            "doma_delta_iou_min": float(finite.min().item()),
            "doma_delta_iou_max": float(finite.max().item()),
            "doma_delta_iou_p10": float(quantiles[0].item()),
            "doma_delta_iou_p25": float(quantiles[1].item()),
            "doma_delta_iou_p50": float(quantiles[2].item()),
            "doma_delta_iou_p75": float(quantiles[3].item()),
            "doma_delta_iou_p90": float(quantiles[4].item()),
            "doma_delta_iou_improve_ratio": float(
                (finite > neutral_threshold).to(dtype=torch.float64).mean().item()
            ),
            "doma_delta_iou_worsen_ratio": float(
                (finite < -neutral_threshold).to(dtype=torch.float64).mean().item()
            ),
            "doma_delta_iou_neutral_ratio": float(
                (absolute <= neutral_threshold)
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "doma_delta_iou_abs_gt_0_01_ratio": float(
                (absolute > 0.01).to(dtype=torch.float64).mean().item()
            ),
            "doma_delta_iou_abs_gt_0_05_ratio": float(
                (absolute > 0.05).to(dtype=torch.float64).mean().item()
            ),
        }
    )
    return stats


def _validate_neutral_threshold(neutral_threshold):
    if isinstance(neutral_threshold, bool) or not isinstance(
        neutral_threshold, (int, float)
    ):
        raise TypeError("neutral_threshold must be a real number")
    neutral_threshold = float(neutral_threshold)
    if not math.isfinite(neutral_threshold) or neutral_threshold < 0.0:
        raise ValueError("neutral_threshold must be finite and non-negative")
    return neutral_threshold


def _validate_corner_tensor(corners, name):
    if not torch.is_tensor(corners):
        raise TypeError("%s must be a torch.Tensor" % name)
    if not torch.is_floating_point(corners):
        raise TypeError("%s must use a floating-point dtype" % name)
    if corners.ndim != 3 or tuple(corners.shape[1:]) != (8, 3):
        raise ValueError("%s must have shape [N,8,3]" % name)


def _safe_corner_rows_to_boxes(corners):
    """Convert finite rows while representing invalid rows with NaN boxes."""
    count = int(corners.shape[0])
    boxes = corners.new_full((count, 7), float("nan"))
    valid = torch.zeros((count,), dtype=torch.bool, device=corners.device)
    finite_indices = torch.isfinite(corners).flatten(1).all(dim=1).nonzero(
        as_tuple=False
    ).squeeze(-1)
    if not finite_indices.numel():
        return boxes, valid

    converted = corners_3d_to_boxes_hwl(corners.index_select(0, finite_indices))
    converted_valid = (
        torch.isfinite(converted).all(dim=1)
        & (converted[:, 3:6] > 0.0).all(dim=1)
    )
    boxes[finite_indices] = converted
    valid[finite_indices] = converted_valid
    return boxes, valid
