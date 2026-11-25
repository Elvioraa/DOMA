"""Rotated BEV IoU and proposal NMS for Stage 1.5 only."""
from functools import lru_cache

import numpy as np
import torch

from .doma_box_coder import boxes_hwl_to_corners_3d, corners_3d_to_boxes_hwl


@lru_cache(maxsize=1)
def cuda_iou_ops():
    """Load the optional PCDet extension once, including a failed import."""
    try:
        from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda
    except (ImportError, OSError):
        return None
    return iou3d_nms_cuda


def _pcdet_boxes(hwl):
    # Both coordinate systems rotate +x toward +y, in radians. PCDet expects
    # length along x and width along y, and its kernel reads contiguous floats.
    boxes = hwl[:, [0, 1, 2, 5, 4, 3, 6]].float().contiguous()
    # check_in_box2d in iou3d_nms_kernel.cu uses an absolute 0.01 margin.
    # IoU is scale invariant. In these units the margin is < 1e-5 metres,
    # avoiding false overlap from near-edge corners without rebuilding PCDet.
    boxes[:, :6].mul_(1024)
    return boxes


def _shapely_bev_iou(boxes_a, boxes_b):
    import shapely

    def footprints(boxes):
        corners = boxes_hwl_to_corners_3d(boxes.to(device='cpu', dtype=torch.float64))
        return shapely.polygons(corners[:, :4, :2].numpy())

    a, b = footprints(boxes_a), footprints(boxes_b)
    rows, cols = np.nonzero(shapely.intersects(a[:, None], b[None, :]))
    overlap = np.zeros((len(a), len(b)), dtype=np.float64)
    overlap[rows, cols] = shapely.area(shapely.intersection(a[rows], b[cols]))
    union = shapely.area(a)[:, None] + shapely.area(b)[None, :] - overlap
    return torch.from_numpy(overlap / union).to(boxes_a)


@torch.no_grad()
def rotated_bev_iou(boxes_a, boxes_b):
    """Pairwise IoU of [x,y,z,h,w,l,yaw] footprints, on the input device."""
    if any(boxes.ndim != 2 or boxes.shape[1] != 7 or not boxes.is_floating_point()
           for boxes in (boxes_a, boxes_b)):
        raise ValueError('Expected floating [N,7] and [M,7] hwl boxes')
    if boxes_a.device != boxes_b.device:
        raise ValueError('Both box tensors must be on the same device')
    shape = (len(boxes_a), len(boxes_b))
    if 0 in shape:
        return boxes_a.new_empty(shape)
    ops = cuda_iou_ops() if boxes_a.is_cuda else None
    if ops is None:
        return _shapely_bev_iou(boxes_a, boxes_b)

    with torch.cuda.device(boxes_a.device):
        result = boxes_a.new_empty(shape, dtype=torch.float32)
        ops.boxes_iou_bev_gpu(_pcdet_boxes(boxes_a), _pcdet_boxes(boxes_b), result)
    return result.clamp_(0, 1).to(dtype=boxes_a.dtype)


@torch.no_grad()
def proposal_nms(corners, scores, threshold):
    """Apply the voxel postprocessor's top-1000 rotated NMS in Stage 1.5."""
    if not len(corners):
        return scores.new_empty((0,), dtype=torch.long)
    ops = cuda_iou_ops() if corners.is_cuda else None
    if ops is None:
        from opencood.utils.box_utils import nms_rotated
        return nms_rotated(corners, scores, threshold)

    # Keep the original NumPy ordering even for tied detector scores at the cut.
    # Only scores and the native NMS keep indices need host access, never polygons.
    order_np = scores.detach().cpu().numpy().argsort()[::-1][:1000].copy()
    order = torch.from_numpy(order_np).to(corners.device)
    boxes = _pcdet_boxes(corners_3d_to_boxes_hwl(corners[order]))
    keep = torch.empty(len(order), dtype=torch.long, device='cpu')
    with torch.cuda.device(corners.device):
        n = ops.nms_gpu(boxes, keep, threshold)
    return order[keep[:n].to(order.device)]
