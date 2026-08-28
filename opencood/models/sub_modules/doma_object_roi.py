"""Memory-bounded rotated ROI sampling for DOMA.

Input feature maps are per-agent Common-BEV tensors already spatially warped
to the ego frame.  Boxes use the repository's ``[x,y,z,h,w,l,yaw]`` order.
BEV x maps to feature columns (W), y maps to rows (H), and positive yaw turns
the local +x/length axis toward +y.  Sampling uses bin centers,
``grid_sample(..., align_corners=False)``, and zero padding.

The implementation deliberately processes one agent and at most ``chunk_size``
proposals at a time.  It never materializes ``[N,A,C,H,W]`` full-map copies.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DOMABEVGeometry:
    """Metric x/y bounds represented by an ego-aligned Common-BEV map."""

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    def __post_init__(self):
        for name in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("%s must be a real number" % name)
            if not math.isfinite(float(value)):
                raise ValueError("%s must be finite" % name)
        if (
            self.x_max <= self.x_min
            or self.y_max <= self.y_min
            or self.z_max <= self.z_min
        ):
            raise ValueError("BEV maximum bounds must exceed minimum bounds")

    @classmethod
    def from_lidar_range(cls, lidar_range):
        """Create geometry from ``[xmin,ymin,zmin,xmax,ymax,zmax]``."""
        if not isinstance(lidar_range, (list, tuple)) or len(lidar_range) != 6:
            raise ValueError("lidar_range must contain six values")
        return cls(
            x_min=float(lidar_range[0]),
            y_min=float(lidar_range[1]),
            z_min=float(lidar_range[2]),
            x_max=float(lidar_range[3]),
            y_max=float(lidar_range[4]),
            z_max=float(lidar_range[5]),
        )

    def cell_size(self, height, width):
        """Return metres per feature row/column for a runtime map shape."""
        if height <= 0 or width <= 0:
            raise ValueError("feature height and width must be positive")
        return (
            (self.y_max - self.y_min) / float(height),
            (self.x_max - self.x_min) / float(width),
        )


class ChunkedRotatedBEVROISampler(nn.Module):
    """Sample ``[N,A,C,Rh,Rw]`` ROIs without proposal-by-full-map expansion.

    ``agent_features`` is ``[A,C,H,W]`` in ego coordinates.  ``proposals`` is
    ``[N,7]`` in hwl order.  Optional ``agent_support`` is a binary/fractional
    ``[A,1,H,W]`` map describing valid sensor support after spatial warping.

    Returns ROI features, a bool ``[N,A]`` valid mask, and ``[N,A]`` coverage.
    Coverage is the fraction of ROI bin centers that are inside normalized
    grid range ``[-1,1]`` and inside the optional support map.
    """

    def __init__(
        self,
        bev_geometry,
        output_size=5,
        chunk_size=32,
        min_coverage=0.5,
    ):
        super().__init__()
        if not isinstance(bev_geometry, DOMABEVGeometry):
            raise TypeError("bev_geometry must be DOMABEVGeometry")
        self.bev_geometry = bev_geometry
        self.output_size = _validate_output_size(output_size)
        self.chunk_size = _positive_int(chunk_size, "chunk_size")
        self.min_coverage = _coverage_threshold(min_coverage)

    def forward(self, agent_features, proposals, agent_support=None):
        """Run differentiable single-scale rotated ROI extraction."""
        _validate_inputs(agent_features, proposals, agent_support)
        agent_count, channels, height, width = agent_features.shape
        proposal_count = int(proposals.shape[0])
        roi_h, roi_w = self.output_size
        if proposal_count == 0:
            empty_rois = agent_features.new_empty(
                (0, agent_count, channels, roi_h, roi_w)
            )
            empty_valid = torch.empty(
                (0, agent_count), dtype=torch.bool, device=agent_features.device
            )
            return empty_rois, empty_valid, agent_features.new_empty((0, agent_count))

        proposals = proposals.to(dtype=agent_features.dtype)
        per_agent_rois = []
        per_agent_coverage = []
        for agent_index in range(agent_count):
            roi_chunks = []
            coverage_chunks = []
            for start in range(0, proposal_count, self.chunk_size):
                stop = min(start + self.chunk_size, proposal_count)
                chunk = proposals[start:stop]
                grid = self._build_grid(chunk, height, width)
                chunk_count = int(chunk.shape[0])
                packed_grid = grid.reshape(1, chunk_count * roi_h, roi_w, 2)
                sampled = F.grid_sample(
                    agent_features[agent_index:agent_index + 1],
                    packed_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled = sampled.reshape(
                    channels, chunk_count, roi_h, roi_w
                ).permute(1, 0, 2, 3)
                roi_chunks.append(sampled)

                inside = torch.logical_and(grid >= -1.0, grid <= 1.0).all(dim=-1)
                if agent_support is not None:
                    support = F.grid_sample(
                        agent_support[agent_index:agent_index + 1].to(
                            dtype=agent_features.dtype
                        ),
                        packed_grid,
                        mode="nearest",
                        padding_mode="zeros",
                        align_corners=False,
                    ).reshape(chunk_count, roi_h, roi_w)
                    inside = torch.logical_and(inside, support > 0.5)
                coverage_chunks.append(
                    inside.to(dtype=agent_features.dtype).mean(dim=(-2, -1))
                )
            per_agent_rois.append(torch.cat(roi_chunks, dim=0))
            per_agent_coverage.append(torch.cat(coverage_chunks, dim=0))

        roi_features = torch.stack(per_agent_rois, dim=1)
        coverage = torch.stack(per_agent_coverage, dim=1)
        valid_mask = coverage >= self.min_coverage
        return roi_features, valid_mask, coverage

    def _build_grid(self, proposals, height, width):
        roi_h, roi_w = self.output_size
        dtype = proposals.dtype
        device = proposals.device
        col = (
            (torch.arange(roi_w, dtype=dtype, device=device) + 0.5)
            / float(roi_w)
            - 0.5
        )
        row = (
            (torch.arange(roi_h, dtype=dtype, device=device) + 0.5)
            / float(roi_h)
            - 0.5
        )
        local_x = proposals[:, 5, None, None] * col[None, None, :]
        local_y = proposals[:, 4, None, None] * row[None, :, None]
        cos_yaw = torch.cos(proposals[:, 6, None, None])
        sin_yaw = torch.sin(proposals[:, 6, None, None])
        x = proposals[:, 0, None, None] + cos_yaw * local_x - sin_yaw * local_y
        y = proposals[:, 1, None, None] + sin_yaw * local_x + cos_yaw * local_y
        grid_x = 2.0 * (x - self.bev_geometry.x_min) / (
            self.bev_geometry.x_max - self.bev_geometry.x_min
        ) - 1.0
        grid_y = 2.0 * (y - self.bev_geometry.y_min) / (
            self.bev_geometry.y_max - self.bev_geometry.y_min
        ) - 1.0
        # Broadcasting above gives [N,Rh,Rw] without any full-map replication.
        grid_x = grid_x + torch.zeros(
            (proposals.shape[0], roi_h, roi_w), dtype=dtype, device=device
        )
        grid_y = grid_y + torch.zeros_like(grid_x)
        return torch.stack((grid_x, grid_y), dim=-1)


def _validate_inputs(agent_features, proposals, agent_support):
    if not torch.is_tensor(agent_features) or agent_features.ndim != 4:
        raise ValueError("agent_features must be a torch.Tensor [A,C,H,W]")
    if not torch.is_floating_point(agent_features):
        raise TypeError("agent_features must use a floating-point dtype")
    if any(int(size) <= 0 for size in agent_features.shape):
        raise ValueError("agent_features A,C,H,W dimensions must be positive")
    if not torch.is_tensor(proposals) or proposals.ndim != 2 or proposals.shape[1] != 7:
        raise ValueError("proposals must be a torch.Tensor [N,7] in hwl order")
    if not torch.is_floating_point(proposals):
        raise TypeError("proposals must use a floating-point dtype")
    if proposals.device != agent_features.device:
        raise ValueError("proposals and agent_features must share a device")
    if not bool(torch.isfinite(proposals).all()):
        raise ValueError("proposals must contain only finite values")
    if proposals.numel() and not bool((proposals[:, 3:6] > 0).all()):
        raise ValueError("proposal height, width, and length must be positive")
    if agent_support is not None:
        expected = (
            agent_features.shape[0],
            1,
            agent_features.shape[2],
            agent_features.shape[3],
        )
        if not torch.is_tensor(agent_support) or tuple(agent_support.shape) != expected:
            raise ValueError("agent_support must have shape %r" % (expected,))
        if agent_support.device != agent_features.device:
            raise ValueError("agent_support and agent_features must share a device")


def _validate_output_size(value):
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("output_size must be an int or (Rh,Rw)")
    return (
        _positive_int(value[0], "output_size[0]"),
        _positive_int(value[1], "output_size[1]"),
    )


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _coverage_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("min_coverage must be a real number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError("min_coverage must be in (0,1]")
    return value
