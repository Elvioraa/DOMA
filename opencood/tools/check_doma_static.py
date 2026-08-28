"""CPU/static acceptance checks for the clean DOMA implementation."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from opencood.models.sub_modules.doma_box_coder import encode_box_residual
from opencood.models.sub_modules.doma_config import validate_doma_config
from opencood.models.sub_modules.doma_object import install_doma_modules


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOMA_YAML_ROOT = (
    REPOSITORY_ROOT
    / "opencood"
    / "hypes_yaml"
    / "opv2v"
    / "MoreModality"
    / "DOMA"
)
OFFICIAL_YAML_ROOT = DOMA_YAML_ROOT.parent / "HEAL"

YAML_MAP = {
    "stage1/m1.yaml": "stage1/m1_pyramid.yaml",
    "stage2/m2.yaml": "stage2/m2_single_pyramid.yaml",
    "stage2/m3.yaml": "stage2/m3_single_pyramid.yaml",
    "stage2/m4.yaml": "stage2/m4_single_pyramid.yaml",
    "final_infer/m1m2m3m4.yaml": "final_infer/m1m2m3m4.yaml",
}

EXPECTED_COUNTS = {
    "V1": {
        "stage1": (138248, 24),
        "merged": (163592, 42),
    },
    "V2": {
        "stage1": (315529, 41),
        "merged": (440713, 77),
    },
    "V3": {
        "stage1": (326026, 45),
        "merged": (451210, 81),
    },
}
EXPECTED_METHOD_SHA256 = {
    "V1": "407eb3245b12be1a872868b7c0f2851953d23e458bb41af3f26e9883df779dd0",
    "V2": "0f0429015bbaa2b1c00c05ed0388dd25404b846d901dcd37d6223b2046532e27",
    "V3": "0a2105e6459749cba55f2a22b69d6429990cfab2fb10cc20915e932aa0a312d2",
}


class _DOMAHolder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modality_name_list = [
            key
            for key in args
            if key.startswith("m") and key[1:].isdigit()
        ]
        self.sensor_type_dict = {
            key: args[key]["sensor_type"] for key in self.modality_name_list
        }


def main():
    results = {"yaml": {}, "counts": {}, "dynamic_names": {}}
    for version in ("V1", "V2", "V3"):
        results["yaml"][version] = _check_yaml_pack(version)
        stage1 = _load_yaml(DOMA_YAML_ROOT / version / "stage1" / "m1.yaml")
        final = _load_yaml(
            DOMA_YAML_ROOT / version / "final_infer" / "m1m2m3m4.yaml"
        )
        stage1_counts = _count_doma(stage1["model"]["args"])
        merged_counts = _count_doma(final["model"]["args"])
        assert stage1_counts[:2] == EXPECTED_COUNTS[version]["stage1"]
        assert merged_counts[:2] == EXPECTED_COUNTS[version]["merged"]
        _check_isolation(version, stage1_counts[3])
        _check_isolation(version, merged_counts[3])
        results["counts"][version] = {
            "stage1_parameters": stage1_counts[0],
            "stage1_state_tensors": stage1_counts[1],
            "stage1_trainable": stage1_counts[2],
            "stage1_modules": stage1_counts[3],
            "merged_parameters": merged_counts[0],
            "merged_state_tensors": merged_counts[1],
            "merged_trainable_before_mode_policy": merged_counts[2],
            "merged_modules": merged_counts[3],
        }

    results["dynamic_names"] = {
        "doma_heter_pyramid_collab": _check_dynamic_class(
            REPOSITORY_ROOT / "opencood" / "models" / "doma_heter_pyramid_collab.py",
            "doma_heter_pyramid_collab",
        ),
        "doma_heter_pyramid_single": _check_dynamic_class(
            REPOSITORY_ROOT / "opencood" / "models" / "doma_heter_pyramid_single.py",
            "doma_heter_pyramid_single",
        ),
        "doma_point_pillar_pyramid_loss": _check_dynamic_class(
            REPOSITORY_ROOT / "opencood" / "loss" / "doma_point_pillar_pyramid_loss.py",
            "doma_point_pillar_pyramid_loss",
        ),
    }
    _check_centered_yaw_identity()
    results["rng_isolation"] = _check_rng_isolation()
    results["ablations"] = _check_supported_ablations()
    _check_no_runtime_version_branch()
    print(json.dumps(results, indent=2, sort_keys=True))
    print("DOMA static acceptance: PASS")


def _check_yaml_pack(version):
    checked = []
    for doma_relative, official_relative in YAML_MAP.items():
        doma_path = DOMA_YAML_ROOT / version / doma_relative
        official_path = OFFICIAL_YAML_ROOT / official_relative
        doma_hypes = _load_yaml(doma_path)
        official_hypes = _load_yaml(official_path)
        doma_config = doma_hypes["model"]["args"]["doma"]
        validate_doma_config(doma_config)
        method_config = copy.deepcopy(doma_config)
        method_config.pop("mode", None)
        method_config.pop("active_modality", None)
        serialized = json.dumps(
            method_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        fingerprint = hashlib.sha256(serialized).hexdigest()
        if fingerprint != EXPECTED_METHOD_SHA256[version]:
            raise AssertionError(
                "%s DOMA method fingerprint changed: %s" % (doma_path, fingerprint)
            )
        normalized = copy.deepcopy(doma_hypes)
        normalized["name"] = official_hypes["name"]
        normalized["model"]["args"].pop("doma")
        normalized["model"]["core_method"] = official_hypes["model"][
            "core_method"
        ]
        if normalized["loss"]["core_method"] == "doma_point_pillar_pyramid_loss":
            normalized["loss"]["core_method"] = "point_pillar_pyramid_loss"
        if normalized != official_hypes:
            raise AssertionError(
                "%s does not reduce to Official HEAL after removing DOMA"
                % doma_path
            )
        checked.append(doma_relative)
    return checked


def _count_doma(args):
    holder = _DOMAHolder(args)
    install_doma_modules(holder, args)
    parameter_count = sum(parameter.numel() for parameter in holder.parameters())
    state_count = len(holder.state_dict())
    trainable_count = sum(
        parameter.numel()
        for parameter in holder.parameters()
        if parameter.requires_grad
    )
    modules = [
        name for name, _ in holder.named_children() if name.startswith("doma_")
    ]
    return parameter_count, state_count, trainable_count, modules


def _check_isolation(version, modules):
    if version == "V1":
        assert not any("context" in name for name in modules)
        assert not any("quality" in name for name in modules)
        assert not any("multigranularity" in name for name in modules)
    if version == "V2":
        assert not any("quality" in name for name in modules)


def _check_dynamic_class(path, core_method):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    expected = core_method.replace("_", "").lower()
    matches = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.lower() == expected
    ]
    if len(matches) != 1:
        raise AssertionError(
            "%s must expose exactly one dynamically discoverable class" % path
        )
    return matches[0]


def _check_centered_yaw_identity():
    proposal = torch.tensor([[1.0, 2.0, 0.0, 1.5, 1.6, 3.9, 0.4]])
    centered = encode_box_residual(
        proposal, proposal.clone(), yaw_mode="sin_cos_centered"
    )
    legacy = encode_box_residual(proposal, proposal.clone(), yaw_mode="sin_cos")
    assert torch.equal(centered, torch.zeros_like(centered))
    assert legacy[0, 7].item() == 1.0


def _check_rng_isolation():
    """Prove later-version construction does not perturb earlier parameters."""
    original_state = torch.random.get_rng_state()
    states = {}
    try:
        for version in ("V1", "V2", "V3"):
            torch.manual_seed(20260827)
            hypes = _load_yaml(DOMA_YAML_ROOT / version / "stage1" / "m1.yaml")
            holder = _DOMAHolder(hypes["model"]["args"])
            install_doma_modules(holder, hypes["model"]["args"])
            states[version] = holder.state_dict()
    finally:
        torch.random.set_rng_state(original_state)

    core_prefixes = (
        "doma_shared_object_encoder.",
        "doma_shared_geometry_encoder.",
        "doma_shared_object_refiner.",
    )
    context_prefixes = (
        "doma_shared_context_encoder.",
        "doma_shared_multigranularity_fusion.",
    )
    for key, value in states["V1"].items():
        if key.startswith(core_prefixes):
            assert torch.equal(value, states["V2"][key])
            assert torch.equal(value, states["V3"][key])
    for key, value in states["V2"].items():
        if key.startswith(context_prefixes):
            assert torch.equal(value, states["V3"][key])
    return {
        "v1_core_equal_in_v2_v3": True,
        "v2_context_equal_in_v3": True,
    }


def _check_no_runtime_version_branch():
    """Reject version-dependent ``if`` statements in the tensor runtime."""
    runtime_path = (
        REPOSITORY_ROOT / "opencood" / "models" / "sub_modules" / "doma_object.py"
    )
    source = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runtime_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            condition = ast.get_source_segment(source, node.test) or ""
            if "version" in condition:
                raise AssertionError("DOMA runtime must not branch on version")


def _check_supported_ablations():
    """Exercise the documented explicit ablations without version branches."""
    holders = {}

    v3_hypes = _load_yaml(DOMA_YAML_ROOT / "V3" / "stage1" / "m1.yaml")
    v3_no_quality = copy.deepcopy(v3_hypes["model"]["args"])
    v3_no_quality["doma"]["ablation"] = True
    v3_no_quality["doma"]["quality"] = {"enabled": False}
    v3_no_quality["doma"]["consensus"] = {
        "enabled": True,
        "mode": "uniform_geometry_mean",
        "fallback_to_original": True,
    }
    validate_doma_config(v3_no_quality["doma"])
    holders["v3_without_quality"] = _DOMAHolder(v3_no_quality)
    install_doma_modules(holders["v3_without_quality"], v3_no_quality)
    assert not hasattr(holders["v3_without_quality"], "doma_shared_quality_head")

    v3_no_refiner = copy.deepcopy(v3_no_quality)
    v3_no_refiner["doma"]["refiner"]["enabled"] = False
    v3_no_refiner["doma"]["consensus"]["enabled"] = False
    validate_doma_config(v3_no_refiner["doma"])
    holders["v3_without_refiner"] = _DOMAHolder(v3_no_refiner)
    install_doma_modules(holders["v3_without_refiner"], v3_no_refiner)
    assert not hasattr(holders["v3_without_refiner"], "doma_shared_object_refiner")

    v2_hypes = _load_yaml(DOMA_YAML_ROOT / "V2" / "stage1" / "m1.yaml")
    v2_no_geometry = copy.deepcopy(v2_hypes["model"]["args"])
    v2_no_geometry["doma"]["ablation"] = True
    v2_no_geometry["doma"]["geometry"]["enabled"] = False
    validate_doma_config(v2_no_geometry["doma"])
    holders["v2_without_geometry"] = _DOMAHolder(v2_no_geometry)
    install_doma_modules(holders["v2_without_geometry"], v2_no_geometry)
    assert not hasattr(
        holders["v2_without_geometry"], "doma_shared_geometry_encoder"
    )

    v2_no_context = copy.deepcopy(v2_hypes["model"]["args"])
    v2_no_context["doma"]["ablation"] = True
    v2_no_context["doma"]["multi_granularity"] = {"enabled": False}
    validate_doma_config(v2_no_context["doma"])
    holders["v2_without_context"] = _DOMAHolder(v2_no_context)
    install_doma_modules(holders["v2_without_context"], v2_no_context)
    assert not hasattr(holders["v2_without_context"], "doma_context_roi")

    return {name: True for name in holders}


def _load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


if __name__ == "__main__":
    main()
