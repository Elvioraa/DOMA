"""Small CPU forward/loss/backward checks for DOMA-only object modules."""

from pathlib import Path

import torch
import torch.nn as nn
import yaml
import copy

from opencood.loss.doma_object_loss import compute_doma_object_loss
from opencood.models.sub_modules.doma_box_coder import boxes_hwl_to_corners_3d
from opencood.models.sub_modules.doma_object import (
    install_doma_modules,
    refine_doma_detections,
    run_doma_training,
)


ROOT = Path(__file__).resolve().parents[2]
YAML_ROOT = (
    ROOT
    / "opencood"
    / "hypes_yaml"
    / "opv2v"
    / "MoreModality"
    / "DOMA"
)


class _Holder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modality_name_list = [
            key for key in args if key.startswith("m") and key[1:].isdigit()
        ]
        self.sensor_type_dict = {
            key: args[key]["sensor_type"] for key in self.modality_name_list
        }
        install_doma_modules(self, args)


def main():
    torch.manual_seed(20260827)
    for version in ("V1", "V2", "V3"):
        with open(YAML_ROOT / version / "stage1" / "m1.yaml", "r") as stream:
            hypes = yaml.safe_load(stream)
        model = _Holder(hypes["model"]["args"])
        model.train()
        detail = torch.randn(1, 64, 32, 32, requires_grad=True)
        scene = {
            "agent_features": detail,
            "agent_support": detail.new_ones((1, 1, 32, 32)),
            "agent_modalities": ("m1",),
        }
        context_feature = None
        if model.doma_flags["context"]:
            context_feature = torch.randn(1, 128, 16, 16, requires_grad=True)
            scene["context_agent_features"] = context_feature
            scene["context_agent_support"] = context_feature.new_ones(
                (1, 1, 16, 16)
            )
        if model.doma_flags["quality"]:
            scene["agent_positions"] = detail.new_zeros((1, 2))
        context = {"scenes": (scene,), "box_order": "hwl", "aligned_to": "ego"}
        data = {
            "object_bbx_center": detail.new_tensor(
                [[[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]]
            ),
            "object_bbx_mask": torch.ones((1, 1), dtype=torch.bool),
        }
        payload = run_doma_training(model, context, data)
        object_loss, stats = compute_doma_object_loss(payload)
        assert object_loss.ndim == 0 and torch.isfinite(object_loss)
        object_loss.backward()
        assert detail.grad is not None
        assert model.doma_shared_object_refiner.network[-1].weight.grad is not None
        if context_feature is not None:
            assert context_feature.grad is not None
        if version == "V3":
            assert "doma_quality_loss" in stats
        else:
            assert "doma_quality_loss" not in stats
        print(
            "%s PASS loss=%.6f valid_ratio=%.3f"
            % (version, object_loss.item(), stats["doma_valid_object_ratio"])
        )
        _check_inference_contract(version)
    _check_disabled_consensus_loss()
    print("DOMA CPU functional acceptance: PASS")


def _check_inference_contract(version):
    with open(
        YAML_ROOT / version / "final_infer" / "m1m2m3m4.yaml", "r"
    ) as stream:
        hypes = yaml.safe_load(stream)
    model = _Holder(hypes["model"]["args"])
    model.eval()
    with torch.no_grad():
        model.doma_shared_object_refiner.network[-1].bias[0] = 0.1

    detail = torch.randn(2, 64, 32, 32)
    scene = {
        "agent_features": detail,
        "agent_support": detail.new_ones((2, 1, 32, 32)),
        "agent_modalities": ("m1", "m2"),
    }
    if model.doma_flags["context"]:
        context = torch.randn(2, 128, 16, 16)
        scene["context_agent_features"] = context
        scene["context_agent_support"] = context.new_ones((2, 1, 16, 16))
    if model.doma_flags["quality"]:
        scene["agent_positions"] = detail.new_tensor([[0.0, 0.0], [4.0, 1.0]])
    same_forward_context = {
        "scenes": (scene,),
        "box_order": "hwl",
        "aligned_to": "ego",
    }
    center_boxes = detail.new_tensor(
        [
            [0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.0],
            [8.0, 3.0, -1.0, 1.56, 1.6, 3.9, 0.2],
        ]
    )
    corners = boxes_hwl_to_corners_3d(center_boxes)
    scores = detail.new_tensor([0.9, 0.7])
    refined, returned_scores = refine_doma_detections(
        model, corners, scores, same_forward_context
    )
    assert refined.shape == corners.shape
    assert returned_scores is scores
    assert torch.equal(returned_scores, detail.new_tensor([0.9, 0.7]))
    assert not torch.equal(refined, corners)


def _check_disabled_consensus_loss():
    with open(YAML_ROOT / "V1" / "stage1" / "m1.yaml", "r") as stream:
        hypes = yaml.safe_load(stream)
    args = copy.deepcopy(hypes["model"]["args"])
    args["doma"]["ablation"] = True
    args["doma"]["consensus"]["enabled"] = False
    model = _Holder(args)
    model.train()
    detail = torch.randn(1, 64, 32, 32, requires_grad=True)
    context = {
        "scenes": (
            {
                "agent_features": detail,
                "agent_support": detail.new_ones((1, 1, 32, 32)),
                "agent_modalities": ("m1",),
            },
        ),
        "box_order": "hwl",
        "aligned_to": "ego",
    }
    data = {
        "object_bbx_center": detail.new_tensor(
            [[[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]]
        ),
        "object_bbx_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    payload = run_doma_training(model, context, data)
    _, stats = compute_doma_object_loss(payload)
    assert stats["doma_consensus_loss"] == 0.0


if __name__ == "__main__":
    main()
