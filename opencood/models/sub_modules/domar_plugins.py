"""Modality-independent DOMA-R heads operating on proposal x view tensors."""
import torch
from torch import nn
from torch.nn import functional as F

from .doma_box_coder import (aligned_rotated_bev_iou_hwl, decode_box_residual,
                             encode_box_residual, pairwise_rotated_bev_iou_hwl)


def scalar_mlp(inputs):
    layers = nn.Sequential(nn.Linear(inputs, 32), nn.LayerNorm(32), nn.GELU(),
                           nn.Linear(32, 1))
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return layers


class QualityWeightHead(nn.Module):
    def __init__(self, z_dim, u_dim):
        super().__init__()
        self.net = scalar_mlp(z_dim + u_dim + 4 + 8)

    def forward(self, z, u, priors, residual, valid, mean):
        inputs = torch.cat((z, u, priors, residual - mean[:, None]), dim=-1)
        logits = self.net(inputs).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return logits.softmax(dim=1) * valid


class IoUCalibrationHead(nn.Module):
    def __init__(self, z_dim, u_dim):
        super().__init__()
        self.net = scalar_mlp(z_dim + u_dim + 8 + 2)
        # Fitted on m1 validation predictions. The two training losses do not use gamma.
        self.register_buffer('gamma', torch.tensor(0.0))

    def forward(self, z, u, residual, variance, fraction):
        inputs = torch.cat((z, u, residual, variance, fraction), dim=-1)
        return self.net(inputs).squeeze(-1).sigmoid()

    def blend(self, scores, quality, override=None):
        gamma = self.gamma.clamp(0, 1) if override is None else override
        if gamma == 0:
            return scores
        eps = torch.finfo(scores.dtype).tiny
        return ((1 - gamma) * scores.clamp_min(eps).log()
                + gamma * quality.clamp_min(eps).log()).exp()


class DOMARPlugins(nn.Module):
    def __init__(self, z_dim, u_dim, max_cav):
        super().__init__()
        self.quality = QualityWeightHead(z_dim, u_dim)
        self.calibration = IoUCalibrationHead(z_dim, u_dim)
        self.max_cav = max_cav
        if max_cav < 1 or sum(p.numel() for p in self.parameters()) >= 100000:
            raise ValueError('Invalid max_cav or DOMA-R parameter budget exceeded')

    @classmethod
    def from_model(cls, model, max_cav):
        cfg = model.doma_config
        if not model.doma_flags['consensus'] or cfg['consensus']['mode'] != 'uniform_geometry_mean':
            raise ValueError('DOMA-R requires the chapter 3 uniform geometry consensus baseline')
        return cls(cfg['object_encoder']['embedding_dim'], cfg['geometry']['hidden_dim'], max_cav)

    def forward(self, inputs, quality=True, calibration=True):
        z, u, priors, residual, valid = inputs
        count = valid.sum(dim=1, keepdim=True)
        uniform = valid.to(z.dtype) / count.clamp_min(1)
        mean = (residual * valid.unsqueeze(-1)).sum(1) / count.clamp_min(1)
        weights = self.quality(z, u, priors, residual, valid, mean) if quality else uniform
        # Center the update around DOMA's sum/count to preserve it exactly at equal weights.
        consensus = mean + ((weights - uniform).unsqueeze(-1) * residual).sum(1)
        q = None
        if calibration:
            w = weights.unsqueeze(-1)
            variance = (w * (residual - consensus[:, None]).square()).sum((1, 2))
            q = self.calibration((z * w).sum(1), (u * w).sum(1), consensus,
                                 variance[:, None], count.to(z.dtype) / self.max_cav)
        return consensus, q, weights


def plugin_loss(proposals, gt, residual, quality, valid, yaw_mode, lambda_iou):
    """Match once on initial rotated IoU. Retain unmatched/background calibration targets."""
    with torch.no_grad():
        refined = torch.where(valid[:, None], decode_box_residual(proposals, residual, yaw_mode),
                              proposals)
        q_target = torch.zeros_like(quality)
        positive = torch.zeros_like(valid)
        if len(gt):
            initial_iou, indices = pairwise_rotated_bev_iou_hwl(proposals, gt).max(dim=1)
            matched = gt[indices]
            q_target = aligned_rotated_bev_iou_hwl(refined, matched)
            positive = valid & (initial_iou >= 0.3)
            target = encode_box_residual(proposals[positive], matched[positive], yaw_mode)
    consensus_loss = (F.smooth_l1_loss(residual[positive], target) if positive.any()
                      else residual.sum() * 0)
    return consensus_loss + lambda_iou * F.smooth_l1_loss(quality, q_target), q_target


def freeze_backbone(model):
    model.requires_grad_(False)
    model.eval()
    model.domar_stage15 = True
