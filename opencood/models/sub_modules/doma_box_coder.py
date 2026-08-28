"""Pure PyTorch box geometry utilities for DOMA HEAL V1-V3.

The repository's official OPV2V HEAL configuration uses boxes in
``[x, y, z, h, w, l, yaw]`` order.  Yaw is measured in radians and positive
yaw rotates the local +x (length) axis toward +y in the left-handed OPV2V
BEV convention.  All functions in this module preserve that contract.
"""

import math

import torch


VALID_YAW_MODES = ("sin_cos", "sin_cos_centered")


def wrap_angle(angle):
    """Wrap a tensor of angles to ``[-pi, pi)`` without leaving PyTorch."""
    if not torch.is_tensor(angle):
        raise TypeError("angle must be a torch.Tensor")
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def encode_box_residual(proposals, targets, yaw_mode, eps=1e-6):
    """Encode target boxes relative to proposals as periodic 8D residuals.

    Parameters
    ----------
    proposals, targets : torch.Tensor
        Matching floating tensors with shape ``[..., 7]`` and fields
        ``[x, y, z, h, w, l, yaw]``.
    yaw_mode : str
        ``"sin_cos"`` preserves the legacy yaw target ``[sin(dyaw),
        cos(dyaw)]``. ``"sin_cos_centered"`` uses the zero-identity target
        ``[sin(dyaw), cos(dyaw) - 1]``.

    Returns
    -------
    torch.Tensor
        ``[..., 8]`` containing ``dx, dy, dz, dlog_l, dlog_w, dlog_h`` and
        the two yaw values selected by ``yaw_mode``. Translation is normalized
        by proposal length, width, and height respectively.
    """
    _validate_boxes(proposals, "proposals")
    _validate_boxes(targets, "targets")
    if proposals.shape != targets.shape:
        raise ValueError("proposals and targets must have identical shapes")
    if proposals.device != targets.device:
        raise ValueError("proposals and targets must be on the same device")
    _validate_yaw_mode(yaw_mode)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive real number")

    targets = targets.to(dtype=proposals.dtype)
    proposal_h = proposals[..., 3].clamp_min(float(eps))
    proposal_w = proposals[..., 4].clamp_min(float(eps))
    proposal_l = proposals[..., 5].clamp_min(float(eps))
    target_h = targets[..., 3].clamp_min(float(eps))
    target_w = targets[..., 4].clamp_min(float(eps))
    target_l = targets[..., 5].clamp_min(float(eps))

    dyaw = wrap_angle(targets[..., 6] - proposals[..., 6])
    yaw_cos = torch.cos(dyaw)
    if yaw_mode == "sin_cos_centered":
        yaw_cos = yaw_cos - 1.0
    return torch.stack(
        (
            (targets[..., 0] - proposals[..., 0]) / proposal_l,
            (targets[..., 1] - proposals[..., 1]) / proposal_w,
            (targets[..., 2] - proposals[..., 2]) / proposal_h,
            torch.log(target_l / proposal_l),
            torch.log(target_w / proposal_w),
            torch.log(target_h / proposal_h),
            torch.sin(dyaw),
            yaw_cos,
        ),
        dim=-1,
    )


def decode_box_residual(proposals, residuals, yaw_mode):
    """Decode explicitly versioned periodic residuals into HEAL boxes.

    In ``sin_cos_centered`` mode, an exact zero yaw residual preserves the
    proposal yaw bit-for-bit instead of sending it through ``wrap_angle``.
    Nonzero corrections retain the repository's existing wrap contract.
    """
    _validate_boxes(proposals, "proposals")
    if not torch.is_tensor(residuals):
        raise TypeError("residuals must be a torch.Tensor")
    if not torch.is_floating_point(residuals):
        raise TypeError("residuals must use a floating-point dtype")
    if residuals.shape[:-1] != proposals.shape[:-1] or residuals.shape[-1] != 8:
        raise ValueError(
            "residuals must have shape [...,8] matching proposals; got %r"
            % (tuple(residuals.shape),)
        )
    if residuals.device != proposals.device:
        raise ValueError("proposals and residuals must be on the same device")
    _validate_yaw_mode(yaw_mode)

    residuals = residuals.to(dtype=proposals.dtype)
    proposal_h = proposals[..., 3]
    proposal_w = proposals[..., 4]
    proposal_l = proposals[..., 5]
    yaw_cos = residuals[..., 7]
    if yaw_mode == "sin_cos_centered":
        yaw_cos = yaw_cos + 1.0
    dyaw = torch.atan2(residuals[..., 6], yaw_cos)
    decoded_yaw = wrap_angle(proposals[..., 6] + dyaw)
    if yaw_mode == "sin_cos_centered":
        zero_yaw = (residuals[..., 6] == 0) & (residuals[..., 7] == 0)
        identity_yaw = proposals[..., 6] + residuals[..., 6]
        decoded_yaw = torch.where(zero_yaw, identity_yaw, decoded_yaw)
    return torch.stack(
        (
            proposals[..., 0] + residuals[..., 0] * proposal_l,
            proposals[..., 1] + residuals[..., 1] * proposal_w,
            proposals[..., 2] + residuals[..., 2] * proposal_h,
            proposal_h * torch.exp(residuals[..., 5]),
            proposal_w * torch.exp(residuals[..., 4]),
            proposal_l * torch.exp(residuals[..., 3]),
            decoded_yaw,
        ),
        dim=-1,
    )


def boxes_hwl_to_corners_3d(boxes):
    """Convert ``[N,7]`` hwl boxes to the official ``[N,8,3]`` corners."""
    _validate_boxes(boxes, "boxes")
    if boxes.ndim != 2:
        raise ValueError("boxes must have shape [N,7]")
    template = boxes.new_tensor(
        (
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (-1, -1, 1),
        )
    ) / 2.0
    # Convert hwl to local lwh before applying the repository corner template.
    local = template.unsqueeze(0) * boxes[:, None, [5, 4, 3]]
    cos_yaw = torch.cos(boxes[:, 6]).view(-1, 1)
    sin_yaw = torch.sin(boxes[:, 6]).view(-1, 1)
    x = local[..., 0] * cos_yaw - local[..., 1] * sin_yaw
    y = local[..., 0] * sin_yaw + local[..., 1] * cos_yaw
    corners = torch.stack((x, y, local[..., 2]), dim=-1)
    return corners + boxes[:, None, 0:3]


def corners_3d_to_boxes_hwl(corners):
    """Convert official ``[N,8,3]`` corners to hwl center boxes in PyTorch."""
    if not torch.is_tensor(corners):
        raise TypeError("corners must be a torch.Tensor")
    if not torch.is_floating_point(corners):
        raise TypeError("corners must use a floating-point dtype")
    if corners.ndim != 3 or tuple(corners.shape[1:]) != (8, 3):
        raise ValueError("corners must have shape [N,8,3]")
    if not bool(torch.isfinite(corners).all()):
        raise ValueError("corners must contain only finite values")

    center = corners.mean(dim=1)
    h = torch.abs((corners[:, 4:, 2] - corners[:, :4, 2]).mean(dim=1))
    l_edges = torch.stack(
        (
            corners[:, 0, :2] - corners[:, 3, :2],
            corners[:, 2, :2] - corners[:, 1, :2],
            corners[:, 4, :2] - corners[:, 7, :2],
            corners[:, 6, :2] - corners[:, 5, :2],
        ),
        dim=1,
    )
    w_edges = torch.stack(
        (
            corners[:, 0, :2] - corners[:, 1, :2],
            corners[:, 2, :2] - corners[:, 3, :2],
            corners[:, 4, :2] - corners[:, 5, :2],
            corners[:, 6, :2] - corners[:, 7, :2],
        ),
        dim=1,
    )
    l = torch.linalg.vector_norm(l_edges, dim=-1).mean(dim=1)
    w = torch.linalg.vector_norm(w_edges, dim=-1).mean(dim=1)
    length_axis = corners[:, 1, :2] - corners[:, 2, :2]
    yaw = wrap_angle(torch.atan2(length_axis[:, 1], length_axis[:, 0]))
    return torch.stack(
        (center[:, 0], center[:, 1], center[:, 2], h, w, l, yaw), dim=-1
    )


def boxes_hwl_to_lwh(boxes):
    """Convert repository ``xyzhwlr`` boxes to ``xyzlwhr`` boxes."""
    _validate_boxes(boxes, "boxes")
    return boxes[..., [0, 1, 2, 5, 4, 3, 6]]


def boxes_lwh_to_hwl(boxes):
    """Convert ``xyzlwhr`` boxes to repository ``xyzhwlr`` boxes."""
    _validate_lwh_boxes(boxes, "boxes")
    return boxes[..., [0, 1, 2, 5, 4, 3, 6]]


@torch.no_grad()
def pairwise_rotated_bev_iou_hwl(boxes_a, boxes_b):
    """Return exact pairwise rotated BEV IoU for repository-order boxes.

    The repository's compiled IoU implementation is used when available.  Its
    mathematically equivalent pure-PyTorch polygon fallback keeps CPU tests
    functional when the optional extension is absent.
    """
    _validate_boxes(boxes_a, "boxes_a")
    _validate_boxes(boxes_b, "boxes_b")
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("boxes_a and boxes_b must both have shape [N,7]")
    if boxes_a.device != boxes_b.device:
        raise ValueError("boxes_a and boxes_b must share a device")
    boxes_a_lwh = boxes_hwl_to_lwh(boxes_a).detach().float()
    boxes_b_lwh = boxes_hwl_to_lwh(boxes_b).detach().float()
    if boxes_a.is_cuda:
        try:
            from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import (
                boxes_iou_bev,
            )

            result = boxes_iou_bev(boxes_a_lwh, boxes_b_lwh)
            return result.to(dtype=boxes_a.dtype)
        except (ImportError, OSError, RuntimeError, AttributeError) as error:
            raise RuntimeError(
                "repository CUDA rotated-IoU is required for DOMA "
                "GPU training/inference but is unavailable"
            ) from error

    return _pairwise_rotated_iou_torch(boxes_a_lwh, boxes_b_lwh).to(
        device=boxes_a.device, dtype=boxes_a.dtype
    )


@torch.no_grad()
def aligned_rotated_bev_iou_hwl(boxes_a, boxes_b):
    """Return elementwise rotated BEV IoU for matching ``[N,7]`` boxes."""
    _validate_boxes(boxes_a, "boxes_a")
    _validate_boxes(boxes_b, "boxes_b")
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("boxes_a and boxes_b must both have shape [N,7]")
    if boxes_a.shape != boxes_b.shape:
        raise ValueError("boxes_a and boxes_b must have identical shapes")
    if boxes_a.shape[0] == 0:
        return boxes_a.new_empty((0,))
    return torch.diagonal(pairwise_rotated_bev_iou_hwl(boxes_a, boxes_b))


def _validate_boxes(boxes, name):
    if not torch.is_tensor(boxes):
        raise TypeError("%s must be a torch.Tensor" % name)
    if not torch.is_floating_point(boxes):
        raise TypeError("%s must use a floating-point dtype" % name)
    if boxes.ndim < 1 or boxes.shape[-1] != 7:
        raise ValueError("%s must have shape [...,7] in hwl order" % name)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("%s must contain only finite values" % name)
    if boxes.numel() and not bool((boxes[..., 3:6] > 0).all()):
        raise ValueError("%s height, width, and length must be positive" % name)


def _validate_yaw_mode(yaw_mode):
    if yaw_mode not in VALID_YAW_MODES:
        raise ValueError(
            "yaw_mode must be one of %s; got %r"
            % (VALID_YAW_MODES, yaw_mode)
        )


def _validate_lwh_boxes(boxes, name):
    if not torch.is_tensor(boxes):
        raise TypeError("%s must be a torch.Tensor" % name)
    if not torch.is_floating_point(boxes):
        raise TypeError("%s must use a floating-point dtype" % name)
    if boxes.ndim < 1 or boxes.shape[-1] != 7:
        raise ValueError("%s must have shape [...,7] in lwh order" % name)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("%s must contain only finite values" % name)
    if boxes.numel() and not bool((boxes[..., 3:6] > 0).all()):
        raise ValueError("%s length, width, and height must be positive" % name)


def _pairwise_rotated_iou_torch(boxes_a, boxes_b):
    corners_a = _boxes_to_ccw_corners(boxes_a)
    corners_b = _boxes_to_ccw_corners(boxes_b)
    result = boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    areas_a = boxes_a[:, 3] * boxes_a[:, 4]
    areas_b = boxes_b[:, 3] * boxes_b[:, 4]
    for index_a in range(boxes_a.shape[0]):
        subject = [point for point in corners_a[index_a]]
        for index_b in range(boxes_b.shape[0]):
            clipped = subject
            clip_polygon = corners_b[index_b]
            for edge_index in range(4):
                clipped = _clip_polygon(
                    clipped,
                    clip_polygon[edge_index],
                    clip_polygon[(edge_index + 1) % 4],
                )
                if not clipped:
                    break
            intersection = _polygon_area(clipped, boxes_a)
            union = areas_a[index_a] + areas_b[index_b] - intersection
            result[index_a, index_b] = intersection / union.clamp_min(1e-7)
    return result


def _boxes_to_ccw_corners(boxes):
    half_l = boxes[:, 3] * 0.5
    half_w = boxes[:, 4] * 0.5
    local_x = torch.stack((-half_l, half_l, half_l, -half_l), dim=1)
    local_y = torch.stack((-half_w, -half_w, half_w, half_w), dim=1)
    cosine = torch.cos(boxes[:, 6]).unsqueeze(1)
    sine = torch.sin(boxes[:, 6]).unsqueeze(1)
    x = boxes[:, 0].unsqueeze(1) + cosine * local_x - sine * local_y
    y = boxes[:, 1].unsqueeze(1) + sine * local_x + cosine * local_y
    return torch.stack((x, y), dim=-1)


def _clip_polygon(subject, edge_start, edge_end):
    if not subject:
        return []
    output = []
    previous = subject[-1]
    previous_inside = _inside_edge(previous, edge_start, edge_end)
    for current in subject:
        current_inside = _inside_edge(current, edge_start, edge_end)
        if current_inside:
            if not previous_inside:
                output.append(
                    _segment_line_intersection(previous, current, edge_start, edge_end)
                )
            output.append(current)
        elif previous_inside:
            output.append(
                _segment_line_intersection(previous, current, edge_start, edge_end)
            )
        previous = current
        previous_inside = current_inside
    return output


def _inside_edge(point, edge_start, edge_end):
    edge = edge_end - edge_start
    relative = point - edge_start
    cross = edge[0] * relative[1] - edge[1] * relative[0]
    return bool(cross >= -1e-7)


def _segment_line_intersection(start, end, edge_start, edge_end):
    segment = end - start
    edge = edge_end - edge_start
    denominator = edge[0] * segment[1] - edge[1] * segment[0]
    if bool(denominator.abs() < 1e-8):
        return end
    delta = edge_start - start
    numerator = edge[0] * delta[1] - edge[1] * delta[0]
    return start + (numerator / denominator) * segment


def _polygon_area(points, reference):
    if len(points) < 3:
        return reference.new_tensor(0.0)
    polygon = torch.stack(points)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1))
    )

