"""CPU acceptance checks for DOMA V3 R3 Decoupled Quality Calibration.

The full-model probe uses the repository's real PointPillar, BEV backbone,
ConvNeXt aligner, pyramid, detector heads, and DOMA paths on a reduced spatial
grid.  It deliberately replaces the CUDA-only Stage2 camera encoder with the
existing m1 PointPillar encoder while retaining the real Stage2 m2 ownership
and trainability contract.  CUDA modality smoke tests remain a separate gate.
"""

import ast
import copy
import io
import json
import math
import shutil
import sys
import tempfile
import types
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from opencood.tools.check_doma_model_counts import _install_optional_import_stubs


_install_optional_import_stubs()
try:
    import opencood.utils.box_overlaps  # noqa: F401
except ImportError:
    box_overlaps = types.ModuleType("opencood.utils.box_overlaps")
    box_overlaps.bbox_overlaps = lambda *args, **kwargs: None
    sys.modules["opencood.utils.box_overlaps"] = box_overlaps

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.doma_object_loss import compute_doma_object_loss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.models.doma_heter_pyramid_single import DOMAHeterPyramidSingle
from opencood.models.sub_modules.doma_box_coder import boxes_hwl_to_corners_3d
from opencood.models.sub_modules.doma_config import (
    doma_feature_flags,
    validate_doma_config,
)
from opencood.models.sub_modules.doma_object import (
    DetachedFeatureAffineCalibrator,
    SharedObjectQualityHead,
    configure_doma_trainability,
    install_doma_modules,
    normalized_agent_object_distance,
    predict_scene_residuals,
    refine_doma_detections,
    route_quality_calibrators,
    run_doma_training,
)
from opencood.tools.check_doma_static import _DOMAHolder
from opencood.tools.doma_tools import (
    SHARED_PREFIXES,
    apply_doma_merge_ownership,
    doma_method_fingerprint,
    merge_and_save_final,
)
from opencood.tools.train_utils import setup_optimizer


ROOT = Path(__file__).resolve().parents[2]
YAML_ROOT = (
    ROOT
    / "opencood"
    / "hypes_yaml"
    / "opv2v"
    / "MoreModality"
    / "DOMA"
)
R2_ROOT = YAML_ROOT / "V3_R2"
R3_ROOT = YAML_ROOT / "V3_R3"
CONFIG_PATHS = (
    Path("stage1/m1.yaml"),
    Path("stage2/m2.yaml"),
    Path("stage2/m3.yaml"),
    Path("stage2/m4.yaml"),
    Path("final_infer/m1m2m3m4.yaml"),
)
CALIBRATION_BLOCK = {
    "enabled": True,
    "variant": "detached_affine_learned_v1",
}


def _safe_load(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _anchor_reference_range():
    """Read the anchor range from the pre-DQC Stage1 source of truth."""
    hypes = _safe_load(R2_ROOT / "stage1" / "m1.yaml")
    anchor_range = list(hypes["cav_lidar_range"])
    model_range = list(hypes["model"]["args"]["lidar_range"])
    if anchor_range != model_range:
        raise AssertionError(
            "V3_R2 Stage1 cav_lidar_range must equal model.args.lidar_range"
        )
    return anchor_range


def _distance_normalization_block():
    return {
        "enabled": True,
        "mode": "anchor_reference",
        "reference_range": _anchor_reference_range(),
    }


def _quality_dimensions(model):
    quality_head = model.doma_shared_quality_head
    return quality_head.learned_dim, quality_head.input_dim


def _build_holder(args, seed=303):
    torch.manual_seed(seed)
    holder = _DOMAHolder(args)
    install_doma_modules(holder, args)
    holder._doma_log_printed = True
    holder.train()
    configure_doma_trainability(holder)
    return holder


def _holder_from_path(path, seed=303):
    return _build_holder(_safe_load(path)["model"]["args"], seed=seed)


def _parameter_count(model, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _trainable_names(model):
    return {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _module_grad_l1(module):
    return sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _assert_tree_equal(left, right, path="root"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError("%s tensor/type mismatch" % path)
        if not torch.equal(left, right):
            raise AssertionError("%s tensor mismatch" % path)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise AssertionError("%s mapping/type mismatch" % path)
        if tuple(left) != tuple(right):
            raise AssertionError("%s mapping keys differ" % path)
        for key in left:
            _assert_tree_equal(left[key], right[key], "%s.%s" % (path, key))
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("%s sequence mismatch" % path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_tree_equal(
                left_item, right_item, "%s[%d]" % (path, index)
            )
        return
    if left != right:
        raise AssertionError("%s value mismatch: %r != %r" % (path, left, right))


def _state_equal(left, right):
    if tuple(left.state_dict()) != tuple(right.state_dict()):
        return False
    return all(
        torch.equal(left.state_dict()[key], right.state_dict()[key])
        for key in left.state_dict()
    )


def _holder_training_payload(model, modality):
    torch.manual_seed(1801)
    detail = torch.randn(1, 64, 32, 32)
    context_feature = torch.randn(1, 128, 16, 16)
    scene = {
        "agent_features": detail,
        "agent_support": detail.new_ones((1, 1, 32, 32)),
        "agent_modalities": (modality,),
        "context_agent_features": context_feature,
        "context_agent_support": context_feature.new_ones((1, 1, 16, 16)),
        "agent_positions": detail.new_zeros((1, 2)),
    }
    context = {"scenes": (scene,), "box_order": "hwl", "aligned_to": "ego"}
    data = {
        "object_bbx_center": detail.new_tensor(
            [[[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]]
        ),
        "object_bbx_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    torch.manual_seed(1802)
    return run_doma_training(model, context, data)


def _holder_scene(model, modality, distance=0.0, seed=1901):
    """Build a deterministic valid scene for direct DOMA prediction probes."""
    torch.manual_seed(seed)
    detail = torch.randn(1, 64, 32, 32)
    context_feature = torch.randn(1, 128, 16, 16)
    scene = {
        "agent_features": detail,
        "agent_support": detail.new_ones((1, 1, 32, 32)),
        "agent_modalities": (modality,),
        "context_agent_features": context_feature,
        "context_agent_support": context_feature.new_ones((1, 1, 16, 16)),
        "agent_positions": detail.new_zeros((1, 2)),
    }
    proposal = detail.new_tensor(
        [[float(distance), 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]
    )
    return scene, proposal


def _holder_prediction_payload(model, modality, distance=0.0):
    scene, proposals = _holder_scene(model, modality, distance=distance)
    return predict_scene_residuals(model, scene, proposals)


def _distance_fixture(
    geometry, distances=(10.0, 50.0, 100.0), dtype=torch.float64
):
    proposals = torch.zeros(len(distances), 7, dtype=dtype)
    proposals[:, 0] = torch.tensor(distances, dtype=proposals.dtype)
    scene = {"agent_positions": torch.zeros(1, 2, dtype=proposals.dtype)}
    return normalized_agent_object_distance(
        scene, proposals, geometry
    ).squeeze(1)


def _distance_oracle(reference_range, distances, dtype=torch.float64):
    """Independently hand-compute normalized distance from a YAML range."""
    if not isinstance(reference_range, (tuple, list)) or len(reference_range) != 6:
        raise AssertionError("distance oracle requires a six-value YAML range")
    x_span = float(reference_range[3]) - float(reference_range[0])
    y_span = float(reference_range[4]) - float(reference_range[1])
    diagonal = math.hypot(x_span, y_span)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise AssertionError("distance oracle requires a positive XY diagonal")
    normalized = [
        min(max(math.hypot(float(distance), 0.0) / diagonal, 0.0), 1.0)
        for distance in distances
    ]
    return torch.tensor(normalized, dtype=dtype), diagonal


class _LegacyV3QualityHeadReference(nn.Module):
    """Frozen pre-DQC V3 Quality Head, independent of the runtime class."""

    def __init__(
        self,
        embedding_dim,
        geometry_dim,
        hidden_dim,
        use_roi_coverage,
        use_agent_distance,
    ):
        super().__init__()
        self.use_roi_coverage = bool(use_roi_coverage)
        self.use_agent_distance = bool(use_agent_distance)
        input_dim = embedding_dim + geometry_dim
        input_dim += int(self.use_roi_coverage) + int(self.use_agent_distance)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _quality_scalar(value, reference, name):
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError("%s must have shape [M]" % name)
        if value.shape[0] != reference.shape[0]:
            raise ValueError("%s count must match object embeddings" % name)
        return value.to(
            device=reference.device, dtype=reference.dtype
        ).unsqueeze(-1)

    def forward(
        self,
        object_embedding,
        geometry_embedding,
        roi_coverage=None,
        agent_distance=None,
    ):
        values = [object_embedding, geometry_embedding]
        if self.use_roi_coverage:
            values.append(
                self._quality_scalar(
                    roi_coverage, object_embedding, "roi_coverage"
                )
            )
        if self.use_agent_distance:
            values.append(
                self._quality_scalar(
                    agent_distance, object_embedding, "agent_distance"
                )
            )
        return torch.sigmoid(self.network(torch.cat(values, dim=-1))).squeeze(-1)


def _legacy_v3_distance_reference(scene, proposals, geometry):
    """Frozen pre-DQC V3 normalized-distance implementation."""
    positions = scene.get("agent_positions")
    if not torch.is_tensor(positions) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("quality-aware scene requires agent_positions [A,2]")
    if positions.device != proposals.device:
        positions = positions.to(
            device=proposals.device, dtype=proposals.dtype
        )
    else:
        positions = positions.to(dtype=proposals.dtype)
    diagonal = (
        (geometry.x_max - geometry.x_min) ** 2
        + (geometry.y_max - geometry.y_min) ** 2
    ) ** 0.5
    delta = proposals[:, None, :2] - positions[None, :, :]
    return torch.linalg.vector_norm(delta, dim=-1).div(diagonal).clamp(0.0, 1.0)


def _capture_actual_quality_inputs(model, modality):
    """Capture pre-DQC, calibrator, and Shared Quality Head input tensors."""
    quality_head = model.doma_shared_quality_head
    calibrator = getattr(model, "doma_quality_calibrator_%s" % modality)
    before_dqc = []
    calibrator_inputs = []
    head_inputs = []
    original_builder = quality_head.build_input_features

    def capturing_builder(*args, **kwargs):
        value = original_builder(*args, **kwargs)
        before_dqc.append(value.detach().clone())
        return value

    quality_head.build_input_features = capturing_builder
    calibrator_handle = calibrator.register_forward_pre_hook(
        lambda module, inputs: calibrator_inputs.append(
            inputs[0].detach().clone()
        )
    )
    head_handle = quality_head.network[0].register_forward_pre_hook(
        lambda module, inputs: head_inputs.append(inputs[0].detach().clone())
    )
    try:
        payload = _holder_training_payload(model, modality)
    finally:
        quality_head.build_input_features = original_builder
        calibrator_handle.remove()
        head_handle.remove()
    return payload, before_dqc, calibrator_inputs, head_inputs


def _expect_error(callable_value, error_type, contains):
    try:
        callable_value()
    except error_type as error:
        if contains not in str(error):
            raise AssertionError(
                "error did not contain %r: %s" % (contains, error)
            )
        return str(error)
    raise AssertionError("expected %s" % error_type.__name__)


def _full_stage2_args(dqc_enabled, mode="stage2_adapt"):
    stage1 = _safe_load(R3_ROOT / "stage1" / "m1.yaml")
    stage2 = _safe_load(R3_ROOT / "stage2" / "m2.yaml")
    args = copy.deepcopy(stage1["model"]["args"])
    modality = args.pop("m1")
    modality["aligner_args"] = copy.deepcopy(
        stage2["model"]["args"]["m2"]["aligner_args"]
    )
    small_range = [-6.4, -6.4, -3.0, 6.4, 6.4, 1.0]
    args["lidar_range"] = small_range
    modality["encoder_args"]["lidar_range"] = small_range
    args["m2"] = modality
    args["ego_modality"] = "m2"
    args["fix_encoder"] = False
    args["doma"] = copy.deepcopy(stage2["model"]["args"]["doma"])
    args["doma"]["quality"]["stage2_calibration"]["enabled"] = bool(
        dqc_enabled
    )
    args["doma"]["mode"] = mode
    if mode == "stage2_adapt":
        args["doma"]["active_modality"] = "m2"
    else:
        args["doma"].pop("active_modality", None)
    return args


def _build_full_stage2(dqc_enabled, mode="stage2_adapt"):
    torch.manual_seed(303)
    model = DOMAHeterPyramidSingle(_full_stage2_args(dqc_enabled, mode=mode))
    model._doma_log_printed = True
    if mode == "stage2_adapt":
        model.train()
        with torch.no_grad():
            model.doma_shared_multigranularity_fusion.residual_scale.fill_(1.0)
            model.doma_shared_object_refiner.network[-1].weight.normal_(
                mean=0.0, std=0.05
            )
    else:
        model.eval()
    return model


def _full_input(include_gt=True):
    coordinates = torch.tensor(
        [
            [0, 0, y, x]
            for y in range(0, 32, 4)
            for x in range(0, 32, 4)
        ],
        dtype=torch.int32,
    )
    torch.manual_seed(404)
    voxel_count = int(coordinates.shape[0])
    data = {
        "inputs_m2": {
            "voxel_features": torch.randn(voxel_count, 4, 4),
            "voxel_coords": coordinates,
            "voxel_num_points": torch.full(
                (voxel_count,), 4, dtype=torch.int32
            ),
        },
    }
    if include_gt:
        data.update(
            {
                "object_bbx_center": torch.tensor(
                    [[[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]]
                ),
                "object_bbx_mask": torch.ones((1, 1), dtype=torch.bool),
            }
        )
    return data


def _detection_targets(output):
    batch, anchors, height, width = output["cls_preds"].shape
    positive = torch.zeros(batch, height, width, anchors)
    negative = torch.ones_like(positive)
    positive[0, height // 2, width // 2, 0] = 1.0
    negative[0, height // 2, width // 2, 0] = 0.0
    return {
        "pos_equal_one": positive,
        "neg_equal_one": negative,
        "targets": torch.zeros(batch, height, width, anchors * 7),
    }


def _test_config_pack():
    fingerprints = []
    method_blocks = []
    anchor_range = _anchor_reference_range()
    for relative in CONFIG_PATHS:
        r2 = _safe_load(R2_ROOT / relative)
        r3 = _safe_load(R3_ROOT / relative)
        expected = copy.deepcopy(r2)
        expected["name"] = expected["name"].replace("V3_R2", "V3_R3")
        expected_quality = expected["model"]["args"]["doma"]["quality"]
        expected_quality["stage2_calibration"] = copy.deepcopy(
            CALIBRATION_BLOCK
        )
        expected_quality["distance_normalization"] = (
            _distance_normalization_block()
        )
        if r3 != expected:
            raise AssertionError("R3 has an unintended delta in %s" % relative)
        if "checkpoint_selection" in r3:
            raise AssertionError("R3 must not mix checkpoint selection")
        config = r3["model"]["args"]["doma"]
        validate_doma_config(config)
        fingerprints.append(doma_method_fingerprint(config))
        quality = config["quality"]
        method_blocks.append(
            {
                "stage2_calibration": quality["stage2_calibration"],
                "distance_normalization": quality["distance_normalization"],
            }
        )
        if quality["distance_normalization"]["reference_range"] != anchor_range:
            raise AssertionError("R3 reference range differs from Stage1 m1")
    if len(set(fingerprints)) != 1:
        raise AssertionError("R3 method fingerprints differ")
    if len({json.dumps(block, sort_keys=True) for block in method_blocks}) != 1:
        raise AssertionError("R3 Quality-interface method blocks differ")
    return {
        "files": len(CONFIG_PATHS),
        "only_name_and_quality_interface_delta": True,
        "fingerprints_equal": True,
        "method_blocks_equal": True,
        "anchor_reference_range": anchor_range,
        "checkpoint_selection_absent": True,
    }


def _test_t1():
    result = {}
    for label, relative, modality in (
        ("stage1", Path("stage1/m1.yaml"), "m1"),
        ("stage2", Path("stage2/m2.yaml"), "m2"),
    ):
        old_hypes = _safe_load(R2_ROOT / relative)
        off_hypes = copy.deepcopy(old_hypes)
        off_quality = off_hypes["model"]["args"]["doma"]["quality"]
        off_quality["stage2_calibration"] = {"enabled": False}
        off_quality["distance_normalization"] = {"enabled": False}
        old_model = _build_holder(old_hypes["model"]["args"])
        off_model = _build_holder(off_hypes["model"]["args"])
        if not _state_equal(old_model, off_model):
            raise AssertionError("%s off state differs" % label)
        if _parameter_count(old_model) != _parameter_count(off_model):
            raise AssertionError("%s off parameter count differs" % label)
        if _trainable_names(old_model) != _trainable_names(off_model):
            raise AssertionError("%s off trainability differs" % label)
        old_payload = _holder_training_payload(old_model, modality)
        off_payload = _holder_training_payload(off_model, modality)
        _assert_tree_equal(old_payload, off_payload, label)
        result[label] = {
            "state_keys": len(old_model.state_dict()),
            "parameters": _parameter_count(old_model),
            "trainable_parameters": _parameter_count(
                old_model, trainable_only=True
            ),
            "forward_exact": True,
        }
    return result


def _test_t2():
    base = _safe_load(R3_ROOT / "stage2" / "m2.yaml")["model"]["args"][
        "doma"
    ]
    quality_off = copy.deepcopy(base)
    quality_off["ablation"] = True
    quality_off["quality"] = {
        "enabled": False,
        "stage2_calibration": copy.deepcopy(CALIBRATION_BLOCK),
        "distance_normalization": {"enabled": False},
    }
    _expect_error(
        lambda: validate_doma_config(quality_off),
        ValueError,
        "requires doma.quality.enabled=true",
    )
    unknown_on = copy.deepcopy(base)
    unknown_on["quality"]["stage2_calibration"]["variant"] = "unknown"
    _expect_error(
        lambda: validate_doma_config(unknown_on),
        ValueError,
        "variant must be one of",
    )
    unknown_off = copy.deepcopy(base)
    unknown_off["quality"]["stage2_calibration"] = {
        "enabled": False,
        "variant": "unknown",
    }
    validate_doma_config(unknown_off)
    if doma_feature_flags(unknown_off)["stage2_calibration"]:
        raise AssertionError("disabled unknown variant became active")
    distance_quality_off = copy.deepcopy(base)
    distance_quality_off["ablation"] = True
    distance_quality_off["quality"] = {
        "enabled": False,
        "stage2_calibration": {"enabled": False},
        "distance_normalization": _distance_normalization_block(),
    }
    _expect_error(
        lambda: validate_doma_config(distance_quality_off),
        ValueError,
        "requires doma.quality.enabled=true",
    )
    distance_unknown_on = copy.deepcopy(base)
    distance_unknown_on["quality"]["distance_normalization"][
        "mode"
    ] = "unknown"
    _expect_error(
        lambda: validate_doma_config(distance_unknown_on),
        ValueError,
        "mode must be one of",
    )
    distance_unknown_off = copy.deepcopy(base)
    distance_unknown_off["quality"]["distance_normalization"] = {
        "enabled": False,
        "mode": "unknown",
        "reference_range": "unused",
    }
    validate_doma_config(distance_unknown_off)
    if doma_feature_flags(distance_unknown_off)[
        "quality_distance_normalization"
    ]:
        raise AssertionError("disabled distance normalization became active")
    return {
        "quality_disabled_rejected": True,
        "enabled_unknown_rejected": True,
        "disabled_unknown_inert": True,
        "distance_without_quality_rejected": True,
        "distance_unknown_enabled_rejected": True,
        "distance_unknown_disabled_inert": True,
    }


def _test_t3():
    model = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    learned_dim, _ = _quality_dimensions(model)
    calibrator = DetachedFeatureAffineCalibrator(learned_dim)
    features = torch.randn(7, learned_dim, requires_grad=True)
    output = calibrator(features)
    error = float((output - features.detach()).abs().max().item())
    if error != 0.0:
        raise AssertionError("identity initialization is not exact")
    output.sum().backward()
    if features.grad is not None:
        raise AssertionError("calibrator did not detach its input")
    return {
        "shape": list(features.shape),
        "learned_dim": learned_dim,
        "max_abs_error": error,
    }


def _test_t4():
    model = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    learned_dim, input_dim = _quality_dimensions(model)
    payload, before_dqc, calibrator_inputs, head_inputs = (
        _capture_actual_quality_inputs(model, "m2")
    )
    expected_full = (2, input_dim)
    expected_learned = (2, learned_dim)
    if len(before_dqc) != 1 or tuple(before_dqc[0].shape) != expected_full:
        raise AssertionError("unexpected pre-DQC Quality input shape")
    if (
        len(calibrator_inputs) != 1
        or tuple(calibrator_inputs[0].shape) != expected_learned
    ):
        raise AssertionError("unexpected DQC learned-feature shape")
    if len(head_inputs) != 1 or tuple(head_inputs[0].shape) != expected_full:
        raise AssertionError("unexpected real Quality input shape")
    scene = payload["scenes"][0]
    if tuple(scene["individual_quality"].shape) != (2,):
        raise AssertionError("unexpected Quality output shape")
    if tuple(scene["quality_targets"].shape) != (2,):
        raise AssertionError("unexpected Quality target shape")
    return {
        "F_q_shape": list(head_inputs[0].shape),
        "calibrator_input_shape": list(calibrator_inputs[0].shape),
        "feature_axis": -1,
        "learned_dim": learned_dim,
        "input_dim": input_dim,
        "pair_level": True,
    }


def _test_t5_t6_t8():
    model = _build_full_stage2(True)
    output = model(_full_input())
    scene = output["doma_object"]["scenes"][0]
    quality_loss = F.smooth_l1_loss(
        scene["individual_quality"], scene["quality_targets"]
    )
    quality_loss.backward()
    names = (
        "encoder_m2",
        "backbone_m2",
        "aligner_m2",
        "doma_object_adapter_m2",
        "doma_context_adapter_m2",
        "doma_shared_quality_head",
        "doma_quality_calibrator_m2",
    )
    quality_grads = {
        name: _module_grad_l1(getattr(model, name)) for name in names
    }
    if quality_grads["doma_quality_calibrator_m2"] <= 0.0:
        raise AssertionError("L_quality did not train DQC")
    calibrator = model.doma_quality_calibrator_m2
    if _module_grad_l1(calibrator) <= 0.0:
        raise AssertionError("DQC gradient unexpectedly vanished")
    if calibrator.gamma.grad is None or calibrator.gamma.grad.abs().sum() <= 0:
        raise AssertionError("L_quality did not train DQC gamma")
    if calibrator.beta.grad is None or calibrator.beta.grad.abs().sum() <= 0:
        raise AssertionError("L_quality did not train DQC beta")
    for name in names[:-1]:
        if quality_grads[name] != 0.0:
            raise AssertionError("L_quality leaked into %s" % name)

    model.zero_grad(set_to_none=True)
    output = model(_full_input())
    hypes = load_yaml(str(R3_ROOT / "stage2" / "m2.yaml"))
    heal_loss = PointPillarPyramidLoss(copy.deepcopy(hypes["loss"]["args"]))(
        output, _detection_targets(output)
    )
    non_quality_payload = copy.copy(output["doma_object"])
    non_quality_payload["quality_enabled"] = False
    residual_loss, _ = compute_doma_object_loss(non_quality_payload)
    (heal_loss + residual_loss).backward()
    detection_grads = {
        name: _module_grad_l1(getattr(model, name)) for name in names
    }
    for name in names[:5]:
        if detection_grads[name] <= 0.0:
            raise AssertionError("detection/private gradient missing for %s" % name)
    if detection_grads["doma_quality_calibrator_m2"] != 0.0:
        raise AssertionError("non-Quality objective trained DQC")

    off_model = _holder_from_path(R2_ROOT / "stage2" / "m2.yaml")
    on_holder = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    parameter_delta = _parameter_count(on_holder) - _parameter_count(off_model)
    trainable_delta = _parameter_count(
        on_holder, trainable_only=True
    ) - _parameter_count(off_model, trainable_only=True)
    holder_calibrator = on_holder.doma_quality_calibrator_m2
    learned_dim, input_dim = _quality_dimensions(on_holder)
    calibrator_parameters = _parameter_count(holder_calibrator)
    expected_delta = 2 * learned_dim
    if parameter_delta != expected_delta or trainable_delta != expected_delta:
        raise AssertionError("DQC parameter delta must be 2*learned_dim")
    if calibrator_parameters != expected_delta:
        raise AssertionError("DQC parameter count does not match 2*learned_dim")
    new_trainable = _trainable_names(on_holder) - _trainable_names(off_model)
    if new_trainable != {
        "doma_quality_calibrator_m2.gamma",
        "doma_quality_calibrator_m2.beta",
    }:
        raise AssertionError("DQC introduced unexpected trainable parameters")

    optimizer = setup_optimizer(hypes, model)
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    calibrator_ids = {
        id(parameter)
        for parameter in model.doma_quality_calibrator_m2.parameters()
    }
    if set(optimizer_ids) != expected_ids or len(optimizer_ids) != len(set(optimizer_ids)):
        raise AssertionError("optimizer coverage is not exactly once")
    if not calibrator_ids.issubset(set(optimizer_ids)):
        raise AssertionError("optimizer omitted DQC")

    return (
        {
            "quality_loss": float(quality_loss.detach().item()),
            "gradient_l1": quality_grads,
        },
        {
            "heal_loss": float(heal_loss.detach().item()),
            "doma_residual_loss": float(residual_loss.detach().item()),
            "gradient_l1": detection_grads,
        },
        {
            "learned_dim": learned_dim,
            "quality_input_dim": input_dim,
            "calibrator_parameters": calibrator_parameters,
            "stage2_trainable_before": _parameter_count(
                off_model, trainable_only=True
            ),
            "stage2_trainable_after": _parameter_count(
                on_holder, trainable_only=True
            ),
            "parameter_delta": parameter_delta,
            "trainable_delta": trainable_delta,
            "optimizer_exact_coverage": True,
        },
        output,
    )


def _test_t7():
    model_path = (
        ROOT / "opencood" / "models" / "sub_modules" / "doma_object.py"
    )
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(model_path))
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        condition = ast.get_source_segment(source, node.test) or ""
        if any(('"%s"' % modality) in condition for modality in ("m2", "m3", "m4")):
            forbidden.append(condition)
    if forbidden:
        raise AssertionError("DQC runtime has modality special cases: %s" % forbidden)
    blocks = []
    classes = []
    for modality in ("m2", "m3", "m4"):
        hypes = _safe_load(R3_ROOT / "stage2" / (modality + ".yaml"))
        quality = hypes["model"]["args"]["doma"]["quality"]
        blocks.append(
            {
                "stage2_calibration": quality["stage2_calibration"],
                "distance_normalization": quality["distance_normalization"],
            }
        )
        model = _holder_from_path(R3_ROOT / "stage2" / (modality + ".yaml"))
        classes.append(
            type(getattr(model, "doma_quality_calibrator_%s" % modality))
        )
    if len({json.dumps(block, sort_keys=True) for block in blocks}) != 1:
        raise AssertionError("modality DQC configs differ")
    if len(set(classes)) != 1:
        raise AssertionError("modalities use different DQC classes")
    return {
        "runtime_special_cases": 0,
        "shared_class": classes[0].__name__,
        "identical_method_blocks": True,
    }


def _test_t9_t10_t11():
    old_stage2_hypes = _safe_load(R2_ROOT / "stage2" / "m2.yaml")
    old_model = _build_holder(old_stage2_hypes["model"]["args"])
    off_hypes = copy.deepcopy(old_stage2_hypes)
    off_hypes["model"]["args"]["doma"]["quality"][
        "stage2_calibration"
    ] = {"enabled": False}
    off_hypes["model"]["args"]["doma"]["quality"][
        "distance_normalization"
    ] = {"enabled": False}
    off_model = _build_holder(off_hypes["model"]["args"], seed=999)
    off_model.load_state_dict(old_model.state_dict(), strict=True)

    old_stage1 = _holder_from_path(R2_ROOT / "stage1" / "m1.yaml")
    dqc_stage2 = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml", seed=999)
    incompatible = dqc_stage2.load_state_dict(old_stage1.state_dict(), strict=False)
    calibrator_keys = {
        "doma_quality_calibrator_m2.gamma",
        "doma_quality_calibrator_m2.beta",
    }
    if not calibrator_keys.issubset(set(incompatible.missing_keys)):
        raise AssertionError("old Stage1 load did not identify new DQC keys")
    calibrator = dqc_stage2.doma_quality_calibrator_m2
    if torch.count_nonzero(calibrator.gamma) or torch.count_nonzero(calibrator.beta):
        raise AssertionError("old Stage1 anchor lost identity initialization")
    for key, value in old_stage1.state_dict().items():
        if key.startswith(SHARED_PREFIXES):
            if not torch.equal(dqc_stage2.state_dict()[key], value):
                raise AssertionError("old Stage1 shared parameter was not loaded")

    with torch.no_grad():
        calibrator.gamma.copy_(torch.linspace(-0.2, 0.2, calibrator.feature_dim))
        calibrator.beta.copy_(torch.linspace(0.3, -0.3, calibrator.feature_dim))
    expected_gamma = calibrator.gamma.detach().clone()
    expected_beta = calibrator.beta.detach().clone()
    buffer = io.BytesIO()
    torch.save(dqc_stage2.state_dict(), buffer)
    buffer.seek(0)
    saved_state = torch.load(buffer, map_location="cpu")
    resumed = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml", seed=1001)
    resumed.load_state_dict(saved_state, strict=True)
    configure_doma_trainability(resumed)
    if not torch.equal(resumed.doma_quality_calibrator_m2.gamma, expected_gamma):
        raise AssertionError("gamma did not survive checkpoint roundtrip")
    if not torch.equal(resumed.doma_quality_calibrator_m2.beta, expected_beta):
        raise AssertionError("beta did not survive checkpoint roundtrip")
    if not resumed.doma_quality_calibrator_m2.gamma.requires_grad:
        raise AssertionError("resumed DQC is not trainable")
    return (
        {
            "old_v3_off_strict_load": True,
            "old_stage1_to_dqc_stage2": True,
            "identity_after_anchor_load": True,
        },
        {"gamma_exact": True, "beta_exact": True},
        {"nonzero_preserved": True, "trainability_preserved": True},
    )


def _stage2_state_with_shared(stage2, stage1_state):
    state = copy.deepcopy(stage2.state_dict())
    for key, value in stage1_state.items():
        if key.startswith(SHARED_PREFIXES):
            state[key] = value.clone()
    return state


def _test_t12_t13():
    stage1 = _holder_from_path(R3_ROOT / "stage1" / "m1.yaml")
    stage1_state = copy.deepcopy(stage1.state_dict())
    dimension_source = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    dqc_dimensions = _quality_dimensions(dimension_source)
    values = {"m2": 0.01, "m3": 0.02, "m4": 0.03}
    stage2_states = []
    for modality in ("m2", "m3", "m4"):
        model = _holder_from_path(R3_ROOT / "stage2" / (modality + ".yaml"))
        state = _stage2_state_with_shared(model, stage1_state)
        state["doma_quality_calibrator_%s.gamma" % modality].fill_(
            values[modality]
        )
        state["doma_quality_calibrator_%s.beta" % modality].fill_(
            -values[modality]
        )
        stage2_states.append(state)
    ordered = stage2_states + [stage1_state]
    merged = apply_doma_merge_ownership(
        OrderedDict(),
        ordered,
        require_dqc_calibrators=True,
        dqc_dimensions=dqc_dimensions,
    )
    for modality, value in values.items():
        gamma = merged["doma_quality_calibrator_%s.gamma" % modality]
        beta = merged["doma_quality_calibrator_%s.beta" % modality]
        if not torch.equal(gamma, torch.full_like(gamma, value)):
            raise AssertionError("%s gamma ownership failed" % modality)
        if not torch.equal(beta, torch.full_like(beta, -value)):
            raise AssertionError("%s beta ownership failed" % modality)
    for key, value in stage1_state.items():
        if key.startswith(SHARED_PREFIXES) and not torch.equal(merged[key], value):
            raise AssertionError("merge changed Stage1 shared key %s" % key)
    final_model = _holder_from_path(
        R3_ROOT / "final_infer" / "m1m2m3m4.yaml"
    )
    if set(merged) != set(final_model.state_dict()):
        raise AssertionError("merged state does not match final model keys")

    with tempfile.TemporaryDirectory() as temporary_root:
        temporary_root = Path(temporary_root)
        model_dirs = []
        for index, (relative, state) in enumerate(
            zip(
                (
                    Path("stage2/m2.yaml"),
                    Path("stage2/m3.yaml"),
                    Path("stage2/m4.yaml"),
                    Path("stage1/m1.yaml"),
                ),
                ordered,
            )
        ):
            model_dir = temporary_root / ("source_%d" % index)
            model_dir.mkdir()
            shutil.copyfile(R3_ROOT / relative, model_dir / "config.yaml")
            torch.save(state, model_dir / "net_epoch1.pth")
            model_dirs.append(str(model_dir))
        output_dir = temporary_root / "merged"
        merged_path = merge_and_save_final(model_dirs, str(output_dir))
        merged_from_disk = torch.load(merged_path, map_location="cpu")
        for key, value in merged.items():
            if not torch.equal(merged_from_disk[key], value):
                raise AssertionError("merge_and_save changed %s" % key)

    missing_all = copy.deepcopy(ordered)
    for state in missing_all[:3]:
        for key in list(state):
            if key.startswith("doma_quality_calibrator_"):
                del state[key]
    _expect_error(
        lambda: apply_doma_merge_ownership(
            OrderedDict(),
            missing_all,
            require_dqc_calibrators=True,
            dqc_dimensions=dqc_dimensions,
        ),
        RuntimeError,
        "required=True present=False",
    )
    _expect_error(
        lambda: apply_doma_merge_ownership(
            OrderedDict(),
            ordered,
            require_dqc_calibrators=False,
            dqc_dimensions=dqc_dimensions,
        ),
        RuntimeError,
        "required=False present=True",
    )
    _expect_error(
        lambda: apply_doma_merge_ownership(
            OrderedDict(),
            [{}, {}, {}, {}],
            require_dqc_calibrators=True,
            dqc_dimensions=dqc_dimensions,
        ),
        RuntimeError,
        "requires DOMA and calibrator",
    )
    return (
        {
            "m2_gamma": values["m2"],
            "m3_gamma": values["m3"],
            "m4_gamma": values["m4"],
            "independent_private_ownership": True,
            "config_state_mismatch_rejected": True,
            "merge_and_save_final": True,
            "dqc_dimensions": list(dqc_dimensions),
        },
        {
            "stage1_shared_exact": True,
            "final_state_keys_exact": True,
            "shared_key_count": len(
                [key for key in stage1_state if key.startswith(SHARED_PREFIXES)]
            ),
        },
    )


def _test_t14():
    results = {}
    for enabled in (False, True):
        model = _build_full_stage2(enabled, mode="inference")
        calls = []
        handle = None
        if enabled:
            handle = model.doma_quality_calibrator_m2.register_forward_hook(
                lambda module, inputs, output: calls.append(int(output.shape[0]))
            )
            with torch.no_grad():
                model.doma_quality_calibrator_m2.gamma.fill_(0.05)
                model.doma_quality_calibrator_m2.beta.fill_(0.01)
        try:
            with torch.no_grad():
                output = model(_full_input(include_gt=False))
                boxes = torch.tensor(
                    [[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.1]]
                )
                corners = boxes_hwl_to_corners_3d(boxes)
                scores = torch.tensor([0.8])
                refined, returned_scores = refine_doma_detections(
                    model, corners, scores, output["doma_context"]
                )
        finally:
            if handle is not None:
                handle.remove()
        if refined.shape != corners.shape or returned_scores is not scores:
            raise AssertionError("inference output contract changed")
        if enabled and calls != [1]:
            raise AssertionError("inference did not route through DQC")
        if not enabled and hasattr(model, "doma_quality_calibrator_m2"):
            raise AssertionError("disabled inference instantiated DQC")
        results["enabled" if enabled else "disabled"] = {
            "full_forward": True,
            "refinement_forward": True,
            "calibrator_calls": sum(calls),
        }

    final_model = _holder_from_path(
        R3_ROOT / "final_infer" / "m1m2m3m4.yaml"
    )
    learned_dim, input_dim = _quality_dimensions(final_model)
    features = torch.randn(4, input_dim)
    with torch.no_grad():
        final_model.doma_quality_calibrator_m2.beta.fill_(0.02)
        final_model.doma_quality_calibrator_m3.beta.fill_(0.03)
        final_model.doma_quality_calibrator_m4.beta.fill_(0.04)
    routed = route_quality_calibrators(
        final_model, features, ("m1", "m2", "m3", "m4")
    )
    if not torch.equal(routed[0], features[0]):
        raise AssertionError("m1 inference path was calibrated")
    for index, offset in enumerate((0.02, 0.03, 0.04), start=1):
        if not torch.allclose(
            routed[index, :learned_dim],
            features[index, :learned_dim] + offset,
        ):
            raise AssertionError("non-m1 learned-feature routing failed")
        if not torch.equal(
            routed[index, learned_dim:], features[index, learned_dim:]
        ):
            raise AssertionError("inference changed semantic Quality scalars")
    results["final_routing"] = {
        "m1_identity": True,
        "m2_m3_m4_private": True,
        "semantic_scalars_exact": True,
    }
    return results


def _test_t15():
    old_hypes = _safe_load(R2_ROOT / "stage1" / "m1.yaml")
    enabled_hypes = _safe_load(R3_ROOT / "stage1" / "m1.yaml")
    old_model = _build_holder(old_hypes["model"]["args"])
    enabled_model = _build_holder(enabled_hypes["model"]["args"])
    if enabled_model.doma_flags["stage2_calibration"]:
        raise AssertionError("Stage1 activated DQC")
    if any("quality_calibrator" in key for key in enabled_model.state_dict()):
        raise AssertionError("Stage1 instantiated DQC")
    if not enabled_model.doma_flags["quality_distance_normalization"]:
        raise AssertionError("Stage1 did not enable anchor-reference distance")
    if enabled_model.doma_quality_distance_geometry != enabled_model.doma_bev_geometry:
        raise AssertionError("Stage1 anchor-reference geometry changed its range")
    if not _state_equal(old_model, enabled_model):
        raise AssertionError("Stage1 state changed")
    if _trainable_names(old_model) != _trainable_names(enabled_model):
        raise AssertionError("Stage1 trainability changed")
    _assert_tree_equal(
        _holder_training_payload(old_model, "m1"),
        _holder_training_payload(enabled_model, "m1"),
        "stage1_enabled_block",
    )
    return {
        "runtime_flag": False,
        "distance_runtime_flag": True,
        "calibrator_instances": 0,
        "state_exact": True,
        "trainability_exact": True,
        "forward_exact": True,
    }


def _test_t16():
    stage2 = {}
    reference_dimensions = None
    for modality in ("m2", "m3", "m4"):
        path = R3_ROOT / "stage2" / (modality + ".yaml")
        hypes = _safe_load(path)
        config = hypes["model"]["args"]["doma"]
        model = _holder_from_path(path)
        learned_dim, input_dim = _quality_dimensions(model)
        config_learned_dim = (
            config["object_encoder"]["embedding_dim"]
            + config["geometry"]["hidden_dim"]
        )
        if learned_dim != config_learned_dim:
            raise AssertionError("learned_dim is not config-derived")
        calibrator = getattr(model, "doma_quality_calibrator_%s" % modality)
        if tuple(calibrator.gamma.shape) != (learned_dim,):
            raise AssertionError("unexpected DQC gamma shape")
        if tuple(calibrator.beta.shape) != (learned_dim,):
            raise AssertionError("unexpected DQC beta shape")
        parameter_count = _parameter_count(calibrator)
        if parameter_count != 2 * learned_dim:
            raise AssertionError("DQC must contain 2*learned_dim parameters")
        dimensions = (learned_dim, input_dim)
        if reference_dimensions is None:
            reference_dimensions = dimensions
        elif dimensions != reference_dimensions:
            raise AssertionError("Stage2 modalities disagree on Quality dimensions")
        stage2[modality] = {
            "learned_dim": learned_dim,
            "input_dim": input_dim,
            "parameters": parameter_count,
        }

    final_model = _holder_from_path(
        R3_ROOT / "final_infer" / "m1m2m3m4.yaml"
    )
    final_learned_dim, final_input_dim = _quality_dimensions(final_model)
    final_calibrator_parameters = sum(
        parameter.numel()
        for name, parameter in final_model.named_parameters()
        if name.startswith("doma_quality_calibrator_")
    )
    expected_final = 3 * 2 * final_learned_dim
    if final_calibrator_parameters != expected_final:
        raise AssertionError("final inference DQC parameter count is wrong")
    if (final_learned_dim, final_input_dim) != reference_dimensions:
        raise AssertionError("final inference Quality dimensions differ")

    perturbed_hypes = copy.deepcopy(
        _safe_load(R3_ROOT / "stage2" / "m2.yaml")
    )
    perturbed_args = perturbed_hypes["model"]["args"]
    perturbed_config = perturbed_args["doma"]
    perturbed_config["object_encoder"]["embedding_dim"] += 17
    perturbed_config["geometry"]["hidden_dim"] += 9
    expected_perturbed_learned_dim = (
        perturbed_config["object_encoder"]["embedding_dim"]
        + perturbed_config["geometry"]["hidden_dim"]
    )
    expected_perturbed_input_dim = expected_perturbed_learned_dim + int(
        perturbed_config["quality"]["use_roi_coverage"]
    ) + int(perturbed_config["quality"]["use_agent_distance"])
    if expected_perturbed_learned_dim == reference_dimensions[0]:
        raise AssertionError("dimension perturbation did not change learned_dim")
    perturbed_model = _build_holder(perturbed_args, seed=1616)
    perturbed_learned_dim, perturbed_input_dim = _quality_dimensions(
        perturbed_model
    )
    perturbed_calibrator = perturbed_model.doma_quality_calibrator_m2
    if (
        perturbed_learned_dim != expected_perturbed_learned_dim
        or perturbed_input_dim != expected_perturbed_input_dim
    ):
        raise AssertionError("perturbed Quality dimensions are not config-derived")
    if tuple(perturbed_calibrator.gamma.shape) != (
        expected_perturbed_learned_dim,
    ) or tuple(perturbed_calibrator.beta.shape) != (
        expected_perturbed_learned_dim,
    ):
        raise AssertionError("perturbed DQC shape is not learned_dim-derived")
    perturbed_parameter_count = _parameter_count(perturbed_calibrator)
    if perturbed_parameter_count != 2 * expected_perturbed_learned_dim:
        raise AssertionError("perturbed DQC count is not 2*learned_dim")
    return {
        "stage2": stage2,
        "final_calibrator_parameters": final_calibrator_parameters,
        "expected_final_parameters": expected_final,
        "perturbed_noncanonical": {
            "embedding_dim": perturbed_config["object_encoder"][
                "embedding_dim"
            ],
            "geometry_hidden_dim": perturbed_config["geometry"][
                "hidden_dim"
            ],
            "learned_dim": perturbed_learned_dim,
            "input_dim": perturbed_input_dim,
            "gamma_shape": list(perturbed_calibrator.gamma.shape),
            "beta_shape": list(perturbed_calibrator.beta.shape),
            "parameters": perturbed_parameter_count,
            "config_derived": True,
        },
    }


def _test_t17():
    model = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    quality_head = model.doma_shared_quality_head
    learned_dim, input_dim = _quality_dimensions(model)
    semantic_dim = input_dim - learned_dim
    expected_semantic_dim = int(quality_head.use_roi_coverage) + int(
        quality_head.use_agent_distance
    )
    if semantic_dim != expected_semantic_dim or semantic_dim != 2:
        raise AssertionError("unexpected semantic Quality scalar contract")

    torch.manual_seed(1717)
    learned = torch.randn(3, learned_dim, requires_grad=True)
    coverage = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    distance = torch.tensor([0.1, 0.4, 0.7], requires_grad=True)
    semantic = torch.stack((coverage, distance), dim=-1)
    features = torch.cat((learned, semantic), dim=-1)
    calibrator = model.doma_quality_calibrator_m2
    with torch.no_grad():
        calibrator.gamma.fill_(0.25)
        calibrator.beta.fill_(-0.125)
    routed = route_quality_calibrators(model, features, ("m2",) * 3)
    expected_learned = learned.detach() * 1.25 - 0.125
    if not torch.equal(routed[:, :learned_dim], expected_learned):
        raise AssertionError("learned-only affine formula differs")
    if not torch.equal(routed[:, learned_dim], coverage.detach()):
        raise AssertionError("DQC changed coverage")
    if not torch.equal(routed[:, learned_dim + 1], distance.detach()):
        raise AssertionError("DQC changed distance")

    routed.square().sum().backward()
    if learned.grad is not None or coverage.grad is not None or distance.grad is not None:
        raise AssertionError("DQC did not detach the complete Quality input")
    if calibrator.gamma.grad is None or calibrator.gamma.grad.abs().sum() <= 0:
        raise AssertionError("semantic preservation test lost gamma gradient")
    if calibrator.beta.grad is None or calibrator.beta.grad.abs().sum() <= 0:
        raise AssertionError("semantic preservation test lost beta gradient")
    return {
        "learned_affine_applied": True,
        "coverage_before": coverage.detach().tolist(),
        "coverage_after": routed[:, learned_dim].detach().tolist(),
        "distance_before": distance.detach().tolist(),
        "distance_after": routed[:, learned_dim + 1].detach().tolist(),
        "semantic_scalars_exact": True,
        "all_source_features_detached": True,
    }


def _test_t18():
    model = _holder_from_path(R3_ROOT / "stage2" / "m2.yaml")
    learned_dim, input_dim = _quality_dimensions(model)
    _, before_dqc, calibrator_inputs, head_inputs = (
        _capture_actual_quality_inputs(model, "m2")
    )
    if len(before_dqc) != 1 or len(calibrator_inputs) != 1 or len(head_inputs) != 1:
        raise AssertionError("unexpected Quality-interface call count")
    if tuple(before_dqc[0].shape) != (2, input_dim):
        raise AssertionError("pre-DQC F_q shape changed")
    if tuple(calibrator_inputs[0].shape) != (2, learned_dim):
        raise AssertionError("calibrator did not receive only learned features")
    if tuple(head_inputs[0].shape) != (2, input_dim):
        raise AssertionError("Shared Quality Head input shape changed")
    if not torch.equal(before_dqc[0], head_inputs[0]):
        raise AssertionError("identity-initialized DQC changed F_q values")
    return {
        "before_shape": list(before_dqc[0].shape),
        "calibrator_shape": list(calibrator_inputs[0].shape),
        "after_shape": list(head_inputs[0].shape),
        "identity_full_input_exact": True,
    }


def _test_t19():
    paths = (
        ("m1", Path("stage1/m1.yaml")),
        ("m2", Path("stage2/m2.yaml")),
        ("m3", Path("stage2/m3.yaml")),
        ("m4", Path("stage2/m4.yaml")),
        ("final", Path("final_infer/m1m2m3m4.yaml")),
    )
    anchor_range = _anchor_reference_range()
    physical_distances = (10.0, 50.0, 100.0)
    oracle_values, anchor_diagonal = _distance_oracle(
        anchor_range, physical_distances
    )
    reference_values = None
    values = {}
    for label, relative in paths:
        hypes = _safe_load(R3_ROOT / relative)
        distance_config = hypes["model"]["args"]["doma"]["quality"][
            "distance_normalization"
        ]
        if distance_config["reference_range"] != anchor_range:
            raise AssertionError("%s does not use the Stage1 anchor range" % label)
        model = _holder_from_path(R3_ROOT / relative)
        if not model.doma_flags["quality_distance_normalization"]:
            raise AssertionError("%s anchor normalization is inactive" % label)
        normalized = _distance_fixture(
            model.doma_quality_distance_geometry,
            distances=physical_distances,
        )
        if not torch.equal(normalized, oracle_values):
            raise AssertionError(
                "%s anchor distance differs from independent oracle" % label
            )
        if reference_values is None:
            reference_values = normalized
        elif not torch.equal(normalized, reference_values):
            raise AssertionError("anchor-normalized distance differs for %s" % label)
        values[label] = normalized.tolist()
    return {
        "physical_distances_m": list(physical_distances),
        "normalized_by_config": values,
        "oracle_values": oracle_values.tolist(),
        "all_exact": True,
        "anchor_reference_range": anchor_range,
        "anchor_diagonal_m": anchor_diagonal,
    }


def _test_t20():
    old_stage1 = _holder_from_path(R2_ROOT / "stage1" / "m1.yaml")
    new_stage1 = _holder_from_path(R3_ROOT / "stage1" / "m1.yaml")
    torch.manual_seed(2020)
    distances = (torch.rand(31, dtype=torch.float64) * 320.0).tolist()
    old_values = _distance_fixture(old_stage1.doma_bev_geometry, distances)
    new_values = _distance_fixture(
        new_stage1.doma_quality_distance_geometry, distances
    )
    oracle_values, anchor_diagonal = _distance_oracle(
        _anchor_reference_range(), distances
    )
    old_new_max_abs_error = float(
        (old_values - new_values).abs().max().item()
    )
    old_oracle_max_abs_error = float(
        (old_values - oracle_values).abs().max().item()
    )
    new_oracle_max_abs_error = float(
        (new_values - oracle_values).abs().max().item()
    )
    if not torch.equal(old_values, new_values):
        raise AssertionError("Stage1 anchor distance changed numerically")
    if not torch.equal(old_values, oracle_values):
        raise AssertionError("legacy Stage1 distance differs from independent oracle")
    if not torch.equal(new_values, oracle_values):
        raise AssertionError("R3 Stage1 distance differs from independent oracle")

    mismatched_hypes = _safe_load(R3_ROOT / "stage1" / "m1.yaml")
    non_anchor_range = _safe_load(R2_ROOT / "stage2" / "m2.yaml")[
        "model"
    ]["args"]["lidar_range"]
    mismatched_hypes["model"]["args"]["doma"]["quality"][
        "distance_normalization"
    ]["reference_range"] = non_anchor_range
    _expect_error(
        lambda: _build_holder(mismatched_hypes["model"]["args"]),
        ValueError,
        "must match model.args.lidar_range",
    )
    return {
        "random_distance_count": len(distances),
        "old_new_max_abs_error": old_new_max_abs_error,
        "old_oracle_max_abs_error": old_oracle_max_abs_error,
        "new_oracle_max_abs_error": new_oracle_max_abs_error,
        "anchor_diagonal_m": anchor_diagonal,
        "mismatched_stage1_reference_rejected": True,
    }


def _test_t21():
    model = _holder_from_path(
        R3_ROOT / "final_infer" / "m1m2m3m4.yaml"
    )
    _, input_dim = _quality_dimensions(model)
    captured = []
    handle = model.doma_shared_quality_head.network[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        payload = _holder_prediction_payload(model, "m2", distance=50.0)
    finally:
        handle.remove()
    if not bool(payload["any_valid"].all()) or len(captured) != 1:
        raise AssertionError("final inference Quality path was not exercised")
    if tuple(captured[0].shape) != (1, input_dim):
        raise AssertionError("final inference Quality input shape changed")
    expected = _distance_fixture(
        model.doma_quality_distance_geometry,
        distances=(50.0,),
        dtype=captured[0].dtype,
    )[0]
    oracle_values, anchor_diagonal = _distance_oracle(
        _anchor_reference_range(),
        distances=(50.0,),
        dtype=captured[0].dtype,
    )
    oracle_expected = oracle_values[0]
    current_bev_value = _distance_fixture(
        model.doma_bev_geometry,
        distances=(50.0,),
        dtype=captured[0].dtype,
    )[0]
    actual = captured[0][0, -1]
    if not torch.equal(actual, expected):
        raise AssertionError("final inference did not use anchor distance")
    if not torch.equal(expected, oracle_expected):
        raise AssertionError(
            "final inference anchor runtime differs from independent oracle"
        )
    if not torch.equal(actual, oracle_expected):
        raise AssertionError(
            "final inference captured distance differs from independent oracle"
        )
    if torch.equal(actual, current_bev_value):
        raise AssertionError("final inference fell back to final-BEV diagonal")
    return {
        "physical_distance_m": 50.0,
        "actual": float(actual.item()),
        "anchor_expected": float(expected.item()),
        "oracle_expected": float(oracle_expected.item()),
        "final_bev_value": float(current_bev_value.item()),
        "anchor_diagonal_m": anchor_diagonal,
        "uses_anchor_reference": True,
    }


def _test_t22():
    results = {}
    legacy_hypes = _safe_load(R2_ROOT / "stage1" / "m1.yaml")
    legacy_config = legacy_hypes["model"]["args"]["doma"]
    head_kwargs = {
        "embedding_dim": legacy_config["object_encoder"]["embedding_dim"],
        "geometry_dim": legacy_config["geometry"]["hidden_dim"],
        "hidden_dim": legacy_config["quality"]["hidden_dim"],
        "use_roi_coverage": legacy_config["quality"]["use_roi_coverage"],
        "use_agent_distance": legacy_config["quality"]["use_agent_distance"],
    }
    torch.manual_seed(2223)
    current_head = SharedObjectQualityHead(**head_kwargs)
    current_head_rng = torch.get_rng_state().clone()
    torch.manual_seed(2223)
    legacy_head = _LegacyV3QualityHeadReference(**head_kwargs)
    legacy_head_rng = torch.get_rng_state().clone()
    if not torch.equal(current_head_rng, legacy_head_rng):
        raise AssertionError("Quality Head refactor changed initialization RNG")
    if not _state_equal(current_head, legacy_head):
        raise AssertionError("Quality Head refactor changed initialized state")
    if _parameter_count(current_head) != _parameter_count(legacy_head):
        raise AssertionError("Quality Head refactor changed parameter count")
    torch.manual_seed(2224)
    pair_count = 7
    object_embedding = torch.randn(pair_count, head_kwargs["embedding_dim"])
    geometry_embedding = torch.randn(pair_count, head_kwargs["geometry_dim"])
    roi_coverage = torch.rand(pair_count)
    agent_distance = torch.rand(pair_count)
    current_quality = current_head(
        object_embedding,
        geometry_embedding,
        roi_coverage=roi_coverage,
        agent_distance=agent_distance,
    )
    legacy_quality = legacy_head(
        object_embedding,
        geometry_embedding,
        roi_coverage=roi_coverage,
        agent_distance=agent_distance,
    )
    if not torch.equal(current_quality, legacy_quality):
        raise AssertionError("Quality Head refactor changed legacy forward")
    results["legacy_quality_head_reference"] = {
        "construction_rng_exact": True,
        "state_dict_exact": True,
        "parameter_count_exact": True,
        "forward_exact": True,
        "pair_count": pair_count,
        "input_dim": current_head.input_dim,
    }
    cases = (
        ("stage1", Path("stage1/m1.yaml"), "m1", "training"),
        ("stage2", Path("stage2/m2.yaml"), "m2", "training"),
        (
            "final_inference",
            Path("final_infer/m1m2m3m4.yaml"),
            "m2",
            "inference",
        ),
    )
    for label, relative, modality, forward_kind in cases:
        old_hypes = _safe_load(R2_ROOT / relative)
        off_hypes = copy.deepcopy(old_hypes)
        off_quality = off_hypes["model"]["args"]["doma"]["quality"]
        off_quality["stage2_calibration"] = {
            "enabled": False,
            "variant": "unused",
        }
        off_quality["distance_normalization"] = {
            "enabled": False,
            "mode": "unused",
            "reference_range": "unused",
        }
        validate_doma_config(off_hypes["model"]["args"]["doma"])

        old_model = _build_holder(old_hypes["model"]["args"], seed=2222)
        old_rng = torch.get_rng_state().clone()
        off_model = _build_holder(off_hypes["model"]["args"], seed=2222)
        off_rng = torch.get_rng_state().clone()
        if not torch.equal(old_rng, off_rng):
            raise AssertionError("%s disabled construction changed RNG" % label)
        if not _state_equal(old_model, off_model):
            raise AssertionError("%s disabled state_dict changed" % label)
        if _parameter_count(old_model) != _parameter_count(off_model):
            raise AssertionError("%s disabled parameter count changed" % label)
        if _trainable_names(old_model) != _trainable_names(off_model):
            raise AssertionError("%s disabled trainability changed" % label)
        if hasattr(off_model, "doma_quality_distance_geometry"):
            raise AssertionError("%s disabled distance geometry was installed" % label)
        if any("quality_calibrator" in key for key in off_model.state_dict()):
            raise AssertionError("%s disabled DQC parameters were installed" % label)

        if forward_kind == "training":
            old_payload = _holder_training_payload(old_model, modality)
            old_forward_rng = torch.get_rng_state().clone()
            off_payload = _holder_training_payload(off_model, modality)
            off_forward_rng = torch.get_rng_state().clone()
            if not torch.equal(old_forward_rng, off_forward_rng):
                raise AssertionError("%s disabled forward changed RNG" % label)
        else:
            old_payload = _holder_prediction_payload(
                old_model, modality, distance=10.0
            )
            off_payload = _holder_prediction_payload(
                off_model, modality, distance=10.0
            )
        _assert_tree_equal(old_payload, off_payload, label)

        old_distance = _distance_fixture(old_model.doma_bev_geometry)
        off_distance = _distance_fixture(off_model.doma_bev_geometry)
        if not torch.equal(old_distance, off_distance):
            raise AssertionError("%s disabled distance scalar changed" % label)
        distance_proposals = torch.zeros(3, 7, dtype=torch.float64)
        distance_proposals[:, 0] = torch.tensor(
            (10.0, 50.0, 100.0), dtype=distance_proposals.dtype
        )
        distance_scene = {
            "agent_positions": torch.zeros(
                1, 2, dtype=distance_proposals.dtype
            )
        }
        legacy_distance = _legacy_v3_distance_reference(
            distance_scene,
            distance_proposals,
            old_model.doma_bev_geometry,
        ).squeeze(1)
        if not torch.equal(old_distance, legacy_distance):
            raise AssertionError(
                "%s runtime distance differs from frozen legacy math" % label
            )
        if not torch.equal(off_distance, legacy_distance):
            raise AssertionError(
                "%s disabled distance differs from frozen legacy math" % label
            )
        results[label] = {
            "state_keys": len(old_model.state_dict()),
            "parameters": _parameter_count(old_model),
            "trainable_parameters": _parameter_count(
                old_model, trainable_only=True
            ),
            "construction_rng_exact": True,
            "forward_exact": True,
            "distance_exact": True,
            "legacy_distance_math_exact": True,
        }
    return results


def main():
    torch.set_num_threads(1)
    report = {"config_pack": _test_config_pack()}
    report["T1"] = _test_t1()
    report["T2"] = _test_t2()
    report["T3"] = _test_t3()
    report["T4"] = _test_t4()
    t5, t6, t8, _ = _test_t5_t6_t8()
    report["T5"] = t5
    report["T6"] = t6
    report["T7"] = _test_t7()
    report["T8"] = t8
    t9, t10, t11 = _test_t9_t10_t11()
    report["T9"] = t9
    report["T10"] = t10
    report["T11"] = t11
    t12, t13 = _test_t12_t13()
    report["T12"] = t12
    report["T13"] = t13
    report["T14"] = _test_t14()
    report["T15"] = _test_t15()
    report["T16"] = _test_t16()
    report["T17"] = _test_t17()
    report["T18"] = _test_t18()
    report["T19"] = _test_t19()
    report["T20"] = _test_t20()
    report["T21"] = _test_t21()
    report["T22"] = _test_t22()
    for key in ("T%d" % index for index in range(1, 23)):
        report[key]["status"] = "PASS"
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DOMA DQC T1-T22 acceptance: PASS")


if __name__ == "__main__":
    main()
