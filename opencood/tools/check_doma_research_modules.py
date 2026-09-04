"""CPU acceptance checks for opt-in DOMA OPA and QAR research modules.

The checks use DOMA-only lightweight holders and in-memory copies of the
packaged YAMLs.  They do not require spconv, a dataset, or a full HEAL model
forward.
"""

import contextlib
import copy
import io
import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from opencood.loss.doma_object_loss import (
    _compute_delta_iou_diagnostics,
    compute_doma_object_loss,
)
from opencood.models.sub_modules.doma_box_coder import (
    boxes_hwl_to_corners_3d,
    corners_3d_to_boxes_hwl,
)
from opencood.models.sub_modules.doma_config import (
    doma_feature_flags,
    validate_doma_config,
)
from opencood.models.sub_modules.doma_diagnostics import (
    DeltaIoUDiagnosticAccumulator,
    compute_post_nms_delta_iou,
)
from opencood.models.sub_modules.doma_object import (
    SHARED_DOMA_PREFIXES,
    _detached_delta_iou,
    assert_doma_qar_checkpoint_ready,
    configure_doma_trainability,
    install_doma_modules,
    refine_doma_detections,
    run_doma_training,
)
from opencood.tools import train_utils
from opencood.tools.doma_tools import (
    SHARED_PREFIXES,
    _validate_config_fingerprints,
    apply_doma_merge_ownership,
    doma_method_fingerprint,
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


def _opa_config(loss_type="mse", weight=1.0, detail=True, context=True):
    return {
        "enabled": True,
        "method": "consistency_proxy",
        "apply_to": "stage2_adapt",
        "hook": "shared_encoder_outputs",
        "target_type": "gt_proposal_consistency",
        "loss_type": loss_type,
        "weight": weight,
        "branches": {"detail": detail, "context": context},
    }


def _qar_config(
    threshold=0.0,
    loss_weight=1.0,
    training_loss=True,
    inference_gate=True,
    apply_to=None,
    detach_features=True,
):
    config = {
        "enabled": True,
        "target_type": "delta_iou",
        "hidden_dim": 64,
        "detach_residual": True,
        "zero_init_output": True,
        "training_loss": {"enabled": False},
        "inference_gate": {"enabled": False},
    }
    if training_loss:
        if apply_to is None:
            apply_to = ["stage1_anchor"]
        config["training_loss"] = {
            "enabled": True,
            "apply_to": (
                list(apply_to)
                if isinstance(apply_to, (tuple, list))
                else apply_to
            ),
            "loss_type": "smooth_l1",
            "weight": loss_weight,
            "detach_target": True,
            "detach_features": detach_features,
        }
    if inference_gate:
        config["inference_gate"] = {
            "enabled": True,
            "mode": "hard",
            "threshold": threshold,
        }
    return config


class _Holder(nn.Module):
    """Construction-only stand-in containing the DOMA modules and a base probe."""

    def __init__(self, args):
        super().__init__()
        self.base_probe = nn.Linear(1, 1)
        self.modality_name_list = [
            key for key in args if key.startswith("m") and key[1:].isdigit()
        ]
        self.sensor_type_dict = {
            key: args[key]["sensor_type"] for key in self.modality_name_list
        }
        install_doma_modules(self, args)
        self._doma_log_printed = True
        configure_doma_trainability(self)


class _ValidityOverrideROI(nn.Module):
    """Test-only wrapper that preserves ROI values but forces validity."""

    def __init__(self, inner, is_valid):
        super().__init__()
        self.inner = inner
        self.is_valid = is_valid

    def forward(self, *args, **kwargs):
        features, valid, coverage = self.inner(*args, **kwargs)
        if type(self.is_valid) is bool:
            override = torch.full_like(valid, self.is_valid)
        else:
            override = torch.as_tensor(
                self.is_valid, dtype=torch.bool, device=valid.device
            )
            if override.ndim != 1 or override.shape[0] != valid.shape[0]:
                raise ValueError("validity override must have one value per proposal")
            override = override[:, None].expand_as(valid)
        return features, override, coverage


def _load_args(relative_path):
    with open(YAML_ROOT / relative_path, "r") as stream:
        return copy.deepcopy(yaml.safe_load(stream)["model"]["args"])


def _research_args(relative_path, opa=False, qar=False, diagnostics=False):
    args = _load_args(relative_path)
    if opa or qar:
        args["doma"]["ablation"] = True
    if opa:
        args["doma"]["object_protocol_alignment"] = _opa_config()
    if qar:
        args["doma"]["quality_aware_refinement"] = _qar_config()
    if diagnostics:
        args["doma"]["delta_iou_diagnostics"] = {
            "enabled": True,
            "apply_to": [args["doma"]["mode"]],
            "neutral_threshold": 1.0e-6,
        }
    return args


def _expect_raises(error_type, callable_value, text):
    try:
        callable_value()
    except error_type as error:
        if text not in str(error):
            raise AssertionError(
                "expected error containing %r, received %r" % (text, str(error))
            )
        return str(error)
    raise AssertionError("expected %s" % error_type.__name__)


def _set_training(model):
    model.train()
    configure_doma_trainability(model)


def _training_inputs(model, modality, seed, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    detail = torch.randn(
        1,
        model.doma_common_bev_channels,
        32,
        32,
        generator=generator,
    ).requires_grad_(requires_grad)
    scene = {
        "agent_features": detail,
        "agent_support": detail.new_ones((1, 1, 32, 32)),
        "agent_modalities": (modality,),
    }
    context_feature = None
    if model.doma_flags["context"]:
        context_feature = torch.randn(
            1,
            model.doma_context_bev_channels,
            16,
            16,
            generator=generator,
        ).requires_grad_(requires_grad)
        scene["context_agent_features"] = context_feature
        scene["context_agent_support"] = context_feature.new_ones(
            (1, 1, 16, 16)
        )
    context = {
        "scenes": (scene,),
        "box_order": "hwl",
        "aligned_to": "ego",
    }
    data = {
        "object_bbx_center": detail.new_tensor(
            [[[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.2]]]
        ),
        "object_bbx_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    return context, data, detail, context_feature


def _run_training(model, modality, seed, requires_grad=False):
    _set_training(model)
    context, data, detail, context_feature = _training_inputs(
        model, modality, seed, requires_grad=requires_grad
    )
    torch.manual_seed(seed)
    payload = run_doma_training(model, context, data)
    return payload, detail, context_feature, data


def _state_max_abs_diff(left, right):
    if set(left) != set(right):
        raise AssertionError("state key sets differ")
    maximum = 0.0
    for key in left:
        if left[key].dtype == torch.bool or not torch.is_floating_point(left[key]):
            if not torch.equal(left[key], right[key]):
                raise AssertionError("non-floating state differs at %s" % key)
            continue
        if left[key].numel():
            difference = float((left[key] - right[key]).abs().max().item())
            maximum = max(maximum, difference)
    return maximum


def _payload_max_abs_diff(left, right, path="payload"):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            raise AssertionError("tensor structure differs at %s" % path)
        if left.dtype != right.dtype or left.device != right.device:
            raise AssertionError("tensor metadata differs at %s" % path)
        if left.dtype == torch.bool or not torch.is_floating_point(left):
            if not torch.equal(left, right):
                raise AssertionError("non-floating tensor differs at %s" % path)
            return 0.0
        if not left.numel():
            return 0.0
        return float((left - right).abs().max().item())
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError("mapping keys differ at %s" % path)
        return max(
            (
                _payload_max_abs_diff(left[key], right[key], "%s.%s" % (path, key))
                for key in left
            ),
            default=0.0,
        )
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("sequence structure differs at %s" % path)
        return max(
            (
                _payload_max_abs_diff(value, right[index], "%s[%d]" % (path, index))
                for index, value in enumerate(left)
            ),
            default=0.0,
        )
    if left != right:
        raise AssertionError("value differs at %s: %r != %r" % (path, left, right))
    return 0.0


def _test_config_contract():
    baseline = _load_args("V2/stage2/m2.yaml")["doma"]
    snapshot = copy.deepcopy(baseline)
    assert validate_doma_config(baseline) is baseline
    assert baseline == snapshot
    assert doma_feature_flags(baseline)["object_protocol_alignment"] is False
    assert doma_feature_flags(baseline)["quality_aware_refinement"] is False
    assert doma_feature_flags(baseline)["delta_iou_diagnostics"] is False

    for name in (
        "object_protocol_alignment",
        "quality_aware_refinement",
        "delta_iou_diagnostics",
    ):
        config = copy.deepcopy(baseline)
        config[name] = {"enabled": False}
        snapshot = copy.deepcopy(config)
        assert validate_doma_config(config) is config
        assert config == snapshot
        assert doma_feature_flags(config)[name] is False

    invalid_count = 0
    for name in (
        "object_protocol_alignment",
        "quality_aware_refinement",
        "delta_iou_diagnostics",
    ):
        config = copy.deepcopy(baseline)
        config[name] = {"enabled": 1}
        _expect_raises(
            TypeError,
            lambda config=config: validate_doma_config(config),
            ".enabled must be bool",
        )
        invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config(weight=0.0)
    _expect_raises(
        ValueError, lambda: validate_doma_config(config), "weight must be positive"
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config()
    config["object_protocol_alignment"]["method"] = "m1_distillation"
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "current OPA is not m1 distillation",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config(
        detail=False, context=False
    )
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "at least one enabled consistency-proxy branch",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config()
    config["object_protocol_alignment"]["branches"]["detail"] = 1
    _expect_raises(
        TypeError,
        lambda: validate_doma_config(config),
        "branches.detail must be bool",
    )
    invalid_count += 1

    config = _load_args("V1/stage2/m2.yaml")["doma"]
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config(
        detail=False, context=True
    )
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "branches.context requires the DOMA Context path",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["active_modality"] = "m1"
    config["object_protocol_alignment"] = _opa_config()
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "requires a non-anchor Stage2 active_modality",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config()
    for key in (
        "center_xy_std_rel",
        "center_z_std_rel",
        "log_size_std",
        "yaw_std_deg",
    ):
        config["training_proposals"][key] = 0.0
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "requires at least one non-zero",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["quality_aware_refinement"] = _qar_config(loss_weight=0.0)
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "training_loss.weight must be positive",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["object_protocol_alignment"] = _opa_config()
    config["object_protocol_alignment"]["loss_type"] = "invalid"
    _expect_raises(
        ValueError, lambda: validate_doma_config(config), "loss_type must be one of"
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["quality_aware_refinement"] = _qar_config()
    config["quality_aware_refinement"]["inference_gate"]["mode"] = "soft"
    _expect_raises(
        ValueError, lambda: validate_doma_config(config), "mode must be hard"
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["quality_aware_refinement"] = _qar_config(threshold=1.01)
    _expect_raises(
        ValueError, lambda: validate_doma_config(config), "threshold must be in"
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["quality_aware_refinement"] = _qar_config(
        training_loss=False, inference_gate=False
    )
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "requires training_loss or inference_gate",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["ablation"] = True
    config["quality_aware_refinement"] = _qar_config()
    del config["quality_aware_refinement"]["training_loss"]
    _expect_raises(
        TypeError,
        lambda: validate_doma_config(config),
        "training_loss must be a mapping",
    )
    invalid_count += 1

    config = copy.deepcopy(baseline)
    config["delta_iou_diagnostics"] = {"enabled": False, "extra": True}
    _expect_raises(
        ValueError,
        lambda: validate_doma_config(config),
        "delta_iou_diagnostics has unknown keys",
    )
    invalid_count += 1

    for block_name, block in (
        ("object_protocol_alignment", _opa_config()),
        ("quality_aware_refinement", _qar_config()),
    ):
        config = copy.deepcopy(baseline)
        config[block_name] = block
        _expect_raises(
            ValueError, lambda config=config: validate_doma_config(config), "ablation=true"
        )
        invalid_count += 1

    opa_stage2 = copy.deepcopy(baseline)
    opa_stage2["ablation"] = True
    opa_stage2["object_protocol_alignment"] = _opa_config()
    validate_doma_config(opa_stage2)
    assert doma_feature_flags(opa_stage2)["object_protocol_alignment"] is True
    opa_stage1 = _load_args("V2/stage1/m1.yaml")["doma"]
    opa_stage1["ablation"] = True
    opa_stage1["object_protocol_alignment"] = _opa_config()
    validate_doma_config(opa_stage1)
    assert doma_feature_flags(opa_stage1)["object_protocol_alignment"] is False

    loss_only = _load_args("V2/stage1/m1.yaml")["doma"]
    loss_only["ablation"] = True
    loss_only["quality_aware_refinement"] = _qar_config(inference_gate=False)
    validate_doma_config(loss_only)
    loss_only_flags = doma_feature_flags(loss_only)
    assert loss_only_flags["quality_aware_refinement"] is True
    assert loss_only_flags["quality_aware_refinement_training_loss"] is True
    assert loss_only_flags["quality_aware_refinement_inference_gate"] is False

    gate_only = copy.deepcopy(baseline)
    gate_only["ablation"] = True
    gate_only["quality_aware_refinement"] = _qar_config(training_loss=False)
    validate_doma_config(gate_only)
    gate_only_flags = doma_feature_flags(gate_only)
    assert gate_only_flags["quality_aware_refinement"] is True
    assert gate_only_flags["quality_aware_refinement_training_loss"] is False
    assert gate_only_flags["quality_aware_refinement_inference_gate"] is False

    diagnostic_only = copy.deepcopy(baseline)
    diagnostic_only["delta_iou_diagnostics"] = {
        "enabled": True,
        "apply_to": ["stage2_adapt"],
        "neutral_threshold": 1.0e-6,
    }
    validate_doma_config(diagnostic_only)
    diagnostic_flags = doma_feature_flags(diagnostic_only)
    assert diagnostic_flags["delta_iou_diagnostics"] is True
    assert diagnostic_flags["quality_aware_refinement"] is False
    return {
        "missing_blocks_disabled": True,
        "explicit_false_blocks_valid": True,
        "input_not_mutated": True,
        "invalid_cases_rejected": invalid_count,
        "opa_runtime_limited_to_stage2": True,
        "opa_proxy_method_explicit": True,
        "qar_loss_gate_flags_independent": True,
        "diagnostic_without_qar_head": True,
    }


def _test_qar_apply_to():
    def flags(relative_path, apply_to, detach_features):
        config = _load_args(relative_path)["doma"]
        config["ablation"] = True
        config["quality_aware_refinement"] = _qar_config(
            inference_gate=False,
            apply_to=apply_to,
            detach_features=detach_features,
        )
        validate_doma_config(config)
        return doma_feature_flags(config)

    stage1_only_stage1 = flags(
        "V2/stage1/m1.yaml", ["stage1_anchor"], True
    )
    stage1_only_stage2 = flags(
        "V2/stage2/m2.yaml", ["stage1_anchor"], True
    )
    assert stage1_only_stage1["quality_aware_refinement_training_loss"] is True
    assert stage1_only_stage2["quality_aware_refinement_training_loss"] is False

    both_stage1 = flags(
        "V2/stage1/m1.yaml",
        ["stage1_anchor", "stage2_adapt"],
        False,
    )
    both_stage2 = flags(
        "V2/stage2/m2.yaml",
        ["stage1_anchor", "stage2_adapt"],
        False,
    )
    assert both_stage1["quality_aware_refinement_training_loss"] is True
    assert both_stage2["quality_aware_refinement_training_loss"] is True

    inference_flags = flags(
        "V2/final_infer/m1m2m3m4.yaml",
        ["stage1_anchor", "stage2_adapt"],
        False,
    )
    assert inference_flags["quality_aware_refinement_training_loss"] is False

    invalid_cases = (
        (["stage2_adapt"], False, "cannot be stage2-only"),
        (["stage1_anchor", "stage2_adapt"], True, "must be false"),
        ([], True, "must not be empty"),
        (["stage1_anchor", "stage1_anchor"], True, "duplicate"),
        (["inference"], True, "entries must be one of"),
        ("stage1_anchor", True, "must be a list"),
    )
    for apply_to, detach_features, expected in invalid_cases:
        config = _load_args("V2/stage1/m1.yaml")["doma"]
        config["ablation"] = True
        config["quality_aware_refinement"] = _qar_config(
            inference_gate=False,
            apply_to=apply_to,
            detach_features=detach_features,
        )
        _expect_raises(
            (TypeError if expected == "must be a list" else ValueError),
            lambda config=config: validate_doma_config(config),
            expected,
        )

    for missing_key, expected in (
        ("apply_to", "apply_to must be a list"),
        ("detach_features", "detach_features must be bool"),
    ):
        config = _load_args("V2/stage1/m1.yaml")["doma"]
        config["ablation"] = True
        config["quality_aware_refinement"] = _qar_config(
            inference_gate=False
        )
        del config["quality_aware_refinement"]["training_loss"][missing_key]
        _expect_raises(
            TypeError,
            lambda config=config: validate_doma_config(config),
            expected,
        )

    return {
        "stage1_only_stage1": True,
        "stage1_only_stage2": False,
        "stage1_and_stage2": True,
        "inference_training_loss": False,
        "stage2_only_rejected": True,
        "invalid_cases_rejected": len(invalid_cases) + 2,
    }


def _stage2_disabled_parity():
    old_args = _load_args("V2/stage2/m2.yaml")
    disabled_args = copy.deepcopy(old_args)
    disabled_args["doma"]["object_protocol_alignment"] = {"enabled": False}
    disabled_args["doma"]["quality_aware_refinement"] = {"enabled": False}
    disabled_args["doma"]["delta_iou_diagnostics"] = {"enabled": False}

    torch.manual_seed(20260908)
    old_model = _Holder(old_args)
    old_construction_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260908)
    disabled_model = _Holder(disabled_args)
    disabled_construction_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_construction_rng, disabled_construction_rng)
    state_difference = _state_max_abs_diff(
        old_model.state_dict(), disabled_model.state_dict()
    )
    assert state_difference == 0.0
    assert sum(parameter.numel() for parameter in old_model.parameters()) == sum(
        parameter.numel() for parameter in disabled_model.parameters()
    )
    legacy_checkpoint = copy.deepcopy(old_model.state_dict())

    old_payload, old_detail, old_context, _ = _run_training(
        old_model, "m2", 4901, requires_grad=True
    )
    old_forward_rng = torch.random.get_rng_state().clone()
    disabled_payload, disabled_detail, disabled_context, _ = _run_training(
        disabled_model, "m2", 4901, requires_grad=True
    )
    disabled_forward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_forward_rng, disabled_forward_rng)
    payload_difference = _payload_max_abs_diff(old_payload, disabled_payload)
    assert payload_difference == 0.0
    old_loss, old_stats = compute_doma_object_loss(old_payload)
    disabled_loss, disabled_stats = compute_doma_object_loss(disabled_payload)
    assert old_stats == disabled_stats
    loss_difference = float((old_loss - disabled_loss).abs().item())
    assert loss_difference == 0.0

    old_model.zero_grad(set_to_none=True)
    disabled_model.zero_grad(set_to_none=True)
    torch.manual_seed(4902)
    old_loss.backward()
    old_backward_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(4902)
    disabled_loss.backward()
    disabled_backward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_backward_rng, disabled_backward_rng)
    gradient_difference = 0.0
    for (old_name, old_parameter), (disabled_name, disabled_parameter) in zip(
        old_model.named_parameters(), disabled_model.named_parameters()
    ):
        assert old_name == disabled_name
        assert (old_parameter.grad is None) == (disabled_parameter.grad is None)
        if old_parameter.grad is not None:
            gradient_difference = max(
                gradient_difference,
                float(
                    (old_parameter.grad - disabled_parameter.grad)
                    .abs()
                    .max()
                    .item()
                ),
            )
    assert gradient_difference == 0.0
    input_gradient_difference = 0.0
    for old_input, disabled_input in (
        (old_detail, disabled_detail),
        (old_context, disabled_context),
    ):
        assert (old_input is None) == (disabled_input is None)
        if old_input is None:
            continue
        assert (old_input.grad is None) == (disabled_input.grad is None)
        if old_input.grad is not None:
            input_gradient_difference = max(
                input_gradient_difference,
                float((old_input.grad - disabled_input.grad).abs().max().item()),
            )
    assert input_gradient_difference == 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        old_optimizer = train_utils.setup_optimizer(_optimizer_hypes(), old_model)
        disabled_optimizer = train_utils.setup_optimizer(
            _optimizer_hypes(), disabled_model
        )
    old_signature = [
        (
            len(group["params"]),
            sum(parameter.numel() for parameter in group["params"]),
            float(group["lr"]),
            float(group["weight_decay"]),
        )
        for group in old_optimizer.param_groups
    ]
    disabled_signature = [
        (
            len(group["params"]),
            sum(parameter.numel() for parameter in group["params"]),
            float(group["lr"]),
            float(group["weight_decay"]),
        )
        for group in disabled_optimizer.param_groups
    ]
    assert old_signature == disabled_signature
    old_optimizer.step()
    disabled_optimizer.step()
    optimizer_difference = _payload_max_abs_diff(
        old_optimizer.state_dict(),
        disabled_optimizer.state_dict(),
        "stage2_optimizer",
    )
    assert optimizer_difference == 0.0
    post_step_state_difference = _state_max_abs_diff(
        old_model.state_dict(), disabled_model.state_dict()
    )
    assert post_step_state_difference == 0.0

    checkpoint_model = _Holder(disabled_args)
    incompatible = checkpoint_model.load_state_dict(
        legacy_checkpoint, strict=False
    )
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    checkpoint_difference = _state_max_abs_diff(
        legacy_checkpoint, checkpoint_model.state_dict()
    )
    assert checkpoint_difference == 0.0
    assert_doma_qar_checkpoint_ready(checkpoint_model, resume_epoch=0)
    return {
        "state_max_abs_diff": state_difference,
        "construction_rng_equal": True,
        "forward_rng_equal": True,
        "payload_max_abs_diff": payload_difference,
        "loss_max_abs_diff": loss_difference,
        "backward_rng_equal": True,
        "gradient_max_abs_diff": gradient_difference,
        "input_gradient_max_abs_diff": input_gradient_difference,
        "optimizer_signature_equal": True,
        "optimizer_state_max_abs_diff": optimizer_difference,
        "post_optimizer_state_max_abs_diff": post_step_state_difference,
        "legacy_checkpoint_max_abs_diff": checkpoint_difference,
    }


def _test_disabled_parity():
    old_args = _load_args("V2/stage1/m1.yaml")
    disabled_args = copy.deepcopy(old_args)
    disabled_args["doma"]["object_protocol_alignment"] = {"enabled": False}
    disabled_args["doma"]["quality_aware_refinement"] = {"enabled": False}
    disabled_args["doma"]["delta_iou_diagnostics"] = {"enabled": False}

    torch.manual_seed(20260904)
    old_model = _Holder(old_args)
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260904)
    disabled_model = _Holder(disabled_args)
    disabled_rng = torch.random.get_rng_state().clone()
    old_state = old_model.state_dict()
    disabled_state = disabled_model.state_dict()
    state_difference = _state_max_abs_diff(old_state, disabled_state)
    assert state_difference == 0.0
    assert torch.equal(old_rng, disabled_rng)
    old_parameter_count = sum(parameter.numel() for parameter in old_model.parameters())
    disabled_parameter_count = sum(
        parameter.numel() for parameter in disabled_model.parameters()
    )
    assert old_parameter_count == disabled_parameter_count
    legacy_checkpoint = copy.deepcopy(old_state)

    old_payload, old_detail, old_context, _ = _run_training(
        old_model, "m1", 4101, requires_grad=True
    )
    old_forward_rng = torch.random.get_rng_state().clone()
    disabled_payload, disabled_detail, disabled_context, _ = _run_training(
        disabled_model, "m1", 4101, requires_grad=True
    )
    disabled_forward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_forward_rng, disabled_forward_rng)
    payload_difference = _payload_max_abs_diff(old_payload, disabled_payload)
    assert payload_difference == 0.0
    old_loss, old_stats = compute_doma_object_loss(old_payload)
    disabled_loss, disabled_stats = compute_doma_object_loss(disabled_payload)
    assert set(old_stats) == set(disabled_stats)
    assert torch.equal(old_loss, disabled_loss)
    assert old_stats == disabled_stats
    old_model.zero_grad(set_to_none=True)
    disabled_model.zero_grad(set_to_none=True)
    torch.manual_seed(4102)
    old_loss.backward()
    old_backward_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(4102)
    disabled_loss.backward()
    disabled_backward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_backward_rng, disabled_backward_rng)
    gradient_difference = 0.0
    for (old_name, old_parameter), (disabled_name, disabled_parameter) in zip(
        old_model.named_parameters(), disabled_model.named_parameters()
    ):
        assert old_name == disabled_name
        assert (old_parameter.grad is None) == (disabled_parameter.grad is None)
        if old_parameter.grad is not None:
            gradient_difference = max(
                gradient_difference,
                float(
                    (old_parameter.grad - disabled_parameter.grad)
                    .abs()
                    .max()
                    .item()
                ),
            )
    assert gradient_difference == 0.0
    input_gradient_difference = float(
        (old_detail.grad - disabled_detail.grad).abs().max().item()
    )
    assert input_gradient_difference == 0.0
    if old_context is not None:
        assert disabled_context is not None
        input_gradient_difference = max(
            input_gradient_difference,
            float(
                (old_context.grad - disabled_context.grad).abs().max().item()
            ),
        )
    assert input_gradient_difference == 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        old_optimizer = train_utils.setup_optimizer(_optimizer_hypes(), old_model)
        disabled_optimizer = train_utils.setup_optimizer(
            _optimizer_hypes(), disabled_model
        )
    optimizer_signature = lambda optimizer: [
        (
            len(group["params"]),
            sum(parameter.numel() for parameter in group["params"]),
            float(group["lr"]),
            float(group["weight_decay"]),
        )
        for group in optimizer.param_groups
    ]
    assert optimizer_signature(old_optimizer) == optimizer_signature(
        disabled_optimizer
    )
    old_optimizer.step()
    disabled_optimizer.step()
    optimizer_state_difference = _payload_max_abs_diff(
        old_optimizer.state_dict(), disabled_optimizer.state_dict(), "optimizer"
    )
    assert optimizer_state_difference == 0.0
    post_step_state_difference = _state_max_abs_diff(
        old_model.state_dict(), disabled_model.state_dict()
    )
    assert post_step_state_difference == 0.0

    checkpoint_model = _Holder(disabled_args)
    incompatible = checkpoint_model.load_state_dict(
        legacy_checkpoint, strict=False
    )
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert _state_max_abs_diff(
        legacy_checkpoint, checkpoint_model.state_dict()
    ) == 0.0

    old_infer_args = _load_args("V2/final_infer/m1m2m3m4.yaml")
    disabled_infer_args = copy.deepcopy(old_infer_args)
    disabled_infer_args["doma"]["object_protocol_alignment"] = {"enabled": False}
    disabled_infer_args["doma"]["quality_aware_refinement"] = {"enabled": False}
    disabled_infer_args["doma"]["delta_iou_diagnostics"] = {"enabled": False}
    torch.manual_seed(20260905)
    old_infer_model = _Holder(old_infer_args)
    old_infer_construction_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260905)
    disabled_infer_model = _Holder(disabled_infer_args)
    disabled_infer_construction_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_infer_construction_rng, disabled_infer_construction_rng)
    assert _state_max_abs_diff(
        old_infer_model.state_dict(), disabled_infer_model.state_dict()
    ) == 0.0
    assert sum(parameter.numel() for parameter in old_infer_model.parameters()) == sum(
        parameter.numel() for parameter in disabled_infer_model.parameters()
    )
    old_infer_model.eval()
    disabled_infer_model.eval()
    infer_centers = torch.tensor(
        [
            [-4.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.0],
            [4.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.1],
        ]
    )
    infer_boxes = boxes_hwl_to_corners_3d(infer_centers)
    infer_scores = torch.tensor([0.8, 0.4])
    infer_context = _inference_context(old_infer_model)
    torch.manual_seed(4103)
    old_output, old_scores = refine_doma_detections(
        old_infer_model, infer_boxes, infer_scores, infer_context
    )
    old_infer_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(4103)
    disabled_output, disabled_scores = refine_doma_detections(
        disabled_infer_model, infer_boxes, infer_scores, infer_context
    )
    disabled_infer_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_infer_rng, disabled_infer_rng)
    inference_difference = float((old_output - disabled_output).abs().max().item())
    assert inference_difference == 0.0
    assert old_scores is infer_scores and disabled_scores is infer_scores
    inference_score_difference = float(
        (old_scores - disabled_scores).abs().max().item()
    )
    assert inference_score_difference == 0.0
    return {
        "stage2_m2": _stage2_disabled_parity(),
        "state_key_count": len(old_state),
        "parameter_count": old_parameter_count,
        "state_max_abs_diff": state_difference,
        "construction_rng_equal": True,
        "forward_rng_equal": True,
        "payload_keys_equal": set(old_payload) == set(disabled_payload),
        "payload_max_abs_diff": payload_difference,
        "loss_keys_equal": set(old_stats) == set(disabled_stats),
        "loss_max_abs_diff": float((old_loss - disabled_loss).abs().item()),
        "backward_rng_equal": True,
        "gradient_max_abs_diff": gradient_difference,
        "input_gradient_max_abs_diff": input_gradient_difference,
        "optimizer_signature_equal": True,
        "optimizer_state_max_abs_diff": optimizer_state_difference,
        "post_optimizer_state_max_abs_diff": post_step_state_difference,
        "legacy_checkpoint_load": True,
        "inference_max_abs_diff": inference_difference,
        "inference_score_max_abs_diff": inference_score_difference,
        "inference_rng_equal": True,
        "inference_score_identity": True,
    }


def _test_opa_stage2():
    args = _research_args("V2/stage2/m2.yaml", opa=True)
    args["doma"]["loss"]["object_loss_weight"] = 0.0
    model = _Holder(args)
    payload, _, _, data = _run_training(model, "m2", 4202, requires_grad=True)

    torch.manual_seed(4202)
    proposals, targets, metadata = model.doma_training_proposal_sampler(
        data["object_bbx_center"][0],
        data["object_bbx_mask"][0],
        with_jitter=True,
        return_metadata=True,
    )
    assert metadata["target_indices"].tolist() == [0, 0]
    assert metadata["is_clean"].tolist() == [True, False]
    assert torch.equal(targets[0], targets[1])
    assert not torch.equal(proposals[0], proposals[1])
    assert not targets.requires_grad

    pairs = payload["scenes"][0]["protocol_pairs"]
    assert set(pairs) == {"detail", "context"}
    for pair in pairs.values():
        assert pair["prediction"].shape[0] > 0
        assert pair["prediction"].requires_grad
        assert not pair["target"].requires_grad
        assert pair["target"].grad_fn is None

    loss, stats = compute_doma_object_loss(payload)
    assert stats["doma_object_loss"] == 0.0
    assert stats["doma_protocol_pair_count"] > 0
    assert loss.item() > 0.0
    model.zero_grad(set_to_none=True)
    loss.backward()
    detail_grad = model.doma_object_adapter_m2.delta[-1].weight.grad
    context_grad = model.doma_context_adapter_m2.delta[-1].weight.grad
    assert detail_grad is not None and float(detail_grad.abs().sum().item()) > 0.0
    assert context_grad is not None and float(context_grad.abs().sum().item()) > 0.0
    shared = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("doma_shared_")
    ]
    assert shared
    assert all(not parameter.requires_grad for _, parameter in shared)
    assert all(parameter.grad is None for _, parameter in shared)
    return {
        "protocol_branches": sorted(pairs),
        "protocol_pair_count": stats["doma_protocol_pair_count"],
        "target_detached": True,
        "detail_adapter_final_grad_l1": float(detail_grad.abs().sum().item()),
        "context_adapter_final_grad_l1": float(context_grad.abs().sum().item()),
        "shared_modules_frozen": True,
    }


def _test_opa_branch_specific_validity():
    def run_case(
        name,
        detail_valid,
        context_valid,
        detail_enabled,
        context_enabled,
        expected_pairs,
        seed,
    ):
        args = _research_args("V2/stage2/m2.yaml", opa=True)
        args["doma"]["object_protocol_alignment"] = _opa_config(
            detail=detail_enabled,
            context=context_enabled,
        )
        args["doma"]["loss"]["object_loss_weight"] = 0.0
        model = _Holder(args)
        model.doma_object_roi = _ValidityOverrideROI(
            model.doma_object_roi, detail_valid
        )
        model.doma_context_roi = _ValidityOverrideROI(
            model.doma_context_roi, context_valid
        )
        payload, _, _, _ = _run_training(model, "m2", seed, requires_grad=True)
        scene = payload["scenes"][0]
        proposal_count = int(scene["valid_mask"].shape[0])
        detail_values = (
            [detail_valid] * proposal_count
            if type(detail_valid) is bool
            else list(detail_valid)
        )
        context_values = (
            [context_valid] * proposal_count
            if type(context_valid) is bool
            else list(context_valid)
        )
        expected_joint_count = sum(
            int(detail and context)
            for detail, context in zip(detail_values, context_values)
        ) * int(scene["valid_mask"].shape[1])
        assert int(scene["valid_mask"].sum().item()) == expected_joint_count
        pairs = payload["scenes"][0]["protocol_pairs"]
        expected_branches = {
            branch
            for branch, enabled in (
                ("detail", detail_enabled),
                ("context", context_enabled),
            )
            if enabled
        }
        assert set(pairs) == expected_branches
        loss, stats = compute_doma_object_loss(payload)
        assert torch.isfinite(loss)
        assert stats["doma_protocol_pair_count"] == expected_pairs
        if expected_pairs:
            model.zero_grad(set_to_none=True)
            loss.backward()
        detail_grad = model.doma_object_adapter_m2.delta[-1].weight.grad
        context_grad = model.doma_context_adapter_m2.delta[-1].weight.grad
        detail_l1 = (
            0.0 if detail_grad is None else float(detail_grad.abs().sum().item())
        )
        context_l1 = (
            0.0 if context_grad is None else float(context_grad.abs().sum().item())
        )
        if expected_pairs:
            assert (detail_l1 > 0.0) is bool(
                detail_enabled and any(detail_values)
            )
            assert (context_l1 > 0.0) is bool(
                context_enabled and any(context_values)
            )
        else:
            assert float(loss.item()) == 0.0
            assert detail_l1 == 0.0 and context_l1 == 0.0
        return {
            "name": name,
            "protocol_branches": sorted(pairs),
            "pair_count": stats["doma_protocol_pair_count"],
            "main_joint_valid_count": int(scene["valid_mask"].sum().item()),
            "detail_adapter_grad_l1": detail_l1,
            "context_adapter_grad_l1": context_l1,
        }

    results = {
        "A_detail_only": run_case(
            "A_detail_only", True, False, True, False, 1, 4211
        ),
        "A_context_only": run_case(
            "A_context_only", True, False, False, True, 0, 4212
        ),
        "B_context_only": run_case(
            "B_context_only", False, True, False, True, 1, 4213
        ),
        "B_detail_only": run_case(
            "B_detail_only", False, True, True, False, 0, 4214
        ),
        "C_both": run_case("C_both", True, True, True, True, 2, 4215),
        "D_neither": run_case(
            "D_neither", False, False, True, True, 0, 4216
        ),
        "E_joint_exclusive_reverse_order": run_case(
            "E_joint_exclusive_reverse_order",
            [True, True],
            [False, True],
            True,
            False,
            1,
            4217,
        ),
    }
    return results


def _test_qar_stage1():
    args = _research_args("V2/stage1/m1.yaml", qar=True)
    args["doma"]["quality_aware_refinement"] = _qar_config(
        inference_gate=False
    )
    args["doma"]["loss"]["object_loss_weight"] = 0.0
    model = _Holder(args)
    with torch.no_grad():
        model.doma_shared_object_refiner.network[-1].bias[0] = 0.25
    payload, _, _, _ = _run_training(model, "m1", 4303, requires_grad=True)
    scene = payload["scenes"][0]
    assert scene["qar_predictions"].requires_grad
    assert not scene["qar_targets"].requires_grad
    assert scene["qar_targets"].grad_fn is None
    assert "delta_iou" not in scene
    assert float(scene["qar_targets"].abs().sum().item()) > 0.0

    loss, stats = compute_doma_object_loss(payload)
    model.zero_grad(set_to_none=True)
    loss.backward()
    head_grad = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in model.doma_shared_qar_head.parameters()
        if parameter.grad is not None
    )
    assert head_grad > 0.0
    assert "doma_qar_loss" in stats and "doma_qar_target_mean" in stats
    return {
        "delta_target_detached": True,
        "delta_target_abs_sum": float(scene["qar_targets"].abs().sum().item()),
        "qar_head_grad_l1": head_grad,
        "qar_loss": stats["doma_qar_loss"],
    }


def _module_grad_l1(module):
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _test_qar_feature_detach():
    results = {}
    for detach_features, seed in ((True, 4311), (False, 4312)):
        args = _research_args("V2/stage1/m1.yaml", qar=True)
        args["doma"]["quality_aware_refinement"] = _qar_config(
            inference_gate=False,
            apply_to=["stage1_anchor"],
            detach_features=detach_features,
        )
        args["doma"]["loss"]["object_loss_weight"] = 0.0
        torch.manual_seed(4300)
        model = _Holder(args)
        with torch.no_grad():
            model.doma_shared_object_refiner.network[-1].bias[0] = 0.25
            model.doma_shared_multigranularity_fusion.residual_scale.fill_(1.0)
            qar_output = model.doma_shared_qar_head.network[-1]
            qar_output.weight.fill_(1.0e-3)
            qar_output.bias.fill_(0.3)
        payload, detail, context_feature, _ = _run_training(
            model, "m1", seed, requires_grad=True
        )
        loss, stats = compute_doma_object_loss(payload)
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {
            "head": _module_grad_l1(model.doma_shared_qar_head),
            "object_encoder": _module_grad_l1(
                model.doma_shared_object_encoder
            ),
            "context_encoder": _module_grad_l1(
                model.doma_shared_context_encoder
            ),
            "geometry_encoder": _module_grad_l1(
                model.doma_shared_geometry_encoder
            ),
            "fusion": _module_grad_l1(
                model.doma_shared_multigranularity_fusion
            ),
            "refiner": _module_grad_l1(model.doma_shared_object_refiner),
            "detail_input": (
                0.0 if detail.grad is None else float(detail.grad.abs().sum().item())
            ),
            "context_input": (
                0.0
                if context_feature.grad is None
                else float(context_feature.grad.abs().sum().item())
            ),
        }
        assert gradients["head"] > 0.0
        representation_names = (
            "object_encoder",
            "context_encoder",
            "geometry_encoder",
            "fusion",
            "detail_input",
            "context_input",
        )
        if detach_features:
            assert all(gradients[name] == 0.0 for name in representation_names)
        else:
            assert all(gradients[name] > 0.0 for name in representation_names)
        assert gradients["refiner"] == 0.0
        results["detach_%s" % str(detach_features).lower()] = {
            "qar_loss": stats["doma_qar_loss"],
            "grad_l1": gradients,
        }

    paired = []
    for detach_features in (True, False):
        args = _research_args("V2/stage1/m1.yaml", qar=True)
        args["doma"]["quality_aware_refinement"] = _qar_config(
            inference_gate=False,
            apply_to=["stage1_anchor"],
            detach_features=detach_features,
        )
        torch.manual_seed(4320)
        model = _Holder(args)
        payload, detail, context_feature, _ = _run_training(
            model, "m1", 4321, requires_grad=True
        )
        object_only_payload = dict(payload)
        object_only_payload.pop(
            "quality_aware_refinement_training_loss_enabled"
        )
        object_only_payload.pop(
            "quality_aware_refinement_training_loss_config"
        )
        object_loss, object_stats = compute_doma_object_loss(
            object_only_payload
        )
        paired.append(
            {
                "model": model,
                "payload": payload,
                "detail": detail,
                "context": context_feature,
                "loss": object_loss,
                "stats": object_stats,
            }
        )
    assert _state_max_abs_diff(
        paired[0]["model"].state_dict(), paired[1]["model"].state_dict()
    ) == 0.0
    for key in (
        "individual_residuals",
        "fused_residuals",
        "refined_boxes",
        "qar_predictions",
        "qar_targets",
    ):
        assert torch.equal(
            paired[0]["payload"]["scenes"][0][key],
            paired[1]["payload"]["scenes"][0][key],
        )
    assert torch.equal(paired[0]["loss"], paired[1]["loss"])
    assert paired[0]["stats"] == paired[1]["stats"]
    for item in paired:
        item["model"].zero_grad(set_to_none=True)
        item["loss"].backward()
    object_gradient_difference = 0.0
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
        paired[0]["model"].named_parameters(),
        paired[1]["model"].named_parameters(),
    ):
        assert left_name == right_name
        assert (left_parameter.grad is None) == (right_parameter.grad is None)
        if left_parameter.grad is not None:
            object_gradient_difference = max(
                object_gradient_difference,
                float(
                    (left_parameter.grad - right_parameter.grad)
                    .abs()
                    .max()
                    .item()
                ),
            )
    object_input_gradient_difference = 0.0
    for input_name in ("detail", "context"):
        left_input = paired[0][input_name]
        right_input = paired[1][input_name]
        assert (left_input is None) == (right_input is None)
        if left_input is None:
            continue
        assert (left_input.grad is None) == (right_input.grad is None)
        if left_input.grad is not None:
            object_input_gradient_difference = max(
                object_input_gradient_difference,
                float((left_input.grad - right_input.grad).abs().max().item()),
            )
    assert object_gradient_difference == 0.0
    assert object_input_gradient_difference == 0.0
    results["object_loss_parity"] = {
        "loss_max_abs_diff": float(
            (paired[0]["loss"] - paired[1]["loss"]).abs().item()
        ),
        "parameter_gradient_max_abs_diff": object_gradient_difference,
        "input_gradient_max_abs_diff": object_input_gradient_difference,
    }

    stage2_args = _research_args("V2/stage2/m2.yaml", qar=True)
    stage2_args["doma"]["quality_aware_refinement"] = _qar_config(
        inference_gate=False,
        apply_to=["stage1_anchor", "stage2_adapt"],
        detach_features=False,
    )
    stage2_args["doma"]["loss"]["object_loss_weight"] = 0.0
    stage2_model = _Holder(stage2_args)
    with torch.no_grad():
        stage2_model.doma_shared_object_refiner.network[-1].bias[0] = 0.25
        stage2_model.doma_shared_multigranularity_fusion.residual_scale.fill_(1.0)
        stage2_qar_output = stage2_model.doma_shared_qar_head.network[-1]
        stage2_qar_output.weight.fill_(1.0e-3)
        stage2_qar_output.bias.fill_(0.3)
    stage2_model.load_state_dict(
        copy.deepcopy(stage2_model.state_dict()), strict=False
    )
    stage2_payload, _, _, _ = _run_training(
        stage2_model, "m2", 4313, requires_grad=True
    )
    stage2_loss, stage2_stats = compute_doma_object_loss(stage2_payload)
    stage2_model.zero_grad(set_to_none=True)
    stage2_loss.backward()
    stage2_detail_grad = _module_grad_l1(stage2_model.doma_object_adapter_m2)
    stage2_context_grad = _module_grad_l1(stage2_model.doma_context_adapter_m2)
    stage2_head_grad = _module_grad_l1(stage2_model.doma_shared_qar_head)
    assert stage2_detail_grad > 0.0 and stage2_context_grad > 0.0
    assert stage2_head_grad == 0.0
    results["stage2_apply"] = {
        "qar_loss": stage2_stats["doma_qar_loss"],
        "detail_adapter_grad_l1": stage2_detail_grad,
        "context_adapter_grad_l1": stage2_context_grad,
        "frozen_head_grad_l1": stage2_head_grad,
    }
    return results


def _test_delta_iou_diagnostics():
    synthetic = _compute_delta_iou_diagnostics(
        (
            {
                "delta_iou": torch.tensor(
                    [-1.0, -0.5, 0.0, 0.25, 1.0, float("nan"), float("inf")]
                )
            },
        )
    )
    assert synthetic["doma_delta_iou_count"] == 7
    assert synthetic["doma_delta_iou_finite_count"] == 5
    assert synthetic["doma_delta_iou_nonfinite_count"] == 2
    assert abs(synthetic["doma_delta_iou_mean"] + 0.05) < 1.0e-6
    assert synthetic["doma_delta_iou_p50"] == 0.0
    assert abs(synthetic["doma_delta_iou_improve_ratio"] - 0.4) < 1.0e-6
    assert abs(synthetic["doma_delta_iou_zero_ratio"] - 0.2) < 1.0e-6
    assert abs(synthetic["doma_delta_iou_worsen_ratio"] - 0.4) < 1.0e-6
    empty = _compute_delta_iou_diagnostics(
        ({"delta_iou": torch.empty((0,))},)
    )
    assert empty["doma_delta_iou_count"] == 0
    assert empty["doma_delta_iou_std"] == 0.0

    probe_boxes = torch.tensor(
        [
            [0.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0],
            [1.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0],
            [2.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0],
        ]
    )
    invalid_refined = probe_boxes.clone()
    invalid_refined[1, 0] = float("nan")
    invalid_refined[2, 3] = 0.0
    invalid_delta = _detached_delta_iou(
        probe_boxes, invalid_refined, probe_boxes
    )
    assert invalid_delta.shape == (3,)
    assert invalid_delta[0] == 0.0
    assert bool(torch.isnan(invalid_delta[1:]).all())
    invalid_stats = _compute_delta_iou_diagnostics(
        ({"delta_iou": invalid_delta},)
    )
    assert invalid_stats["doma_delta_iou_finite_count"] == 1
    assert invalid_stats["doma_delta_iou_nonfinite_count"] == 2

    baseline_args = _load_args("V2/stage1/m1.yaml")
    diagnostic_args = _research_args(
        "V2/stage1/m1.yaml", diagnostics=True
    )
    torch.manual_seed(20260906)
    baseline_model = _Holder(baseline_args)
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260906)
    diagnostic_model = _Holder(diagnostic_args)
    diagnostic_rng = torch.random.get_rng_state().clone()
    assert _state_max_abs_diff(
        baseline_model.state_dict(), diagnostic_model.state_dict()
    ) == 0.0
    assert torch.equal(baseline_rng, diagnostic_rng)
    assert not hasattr(diagnostic_model, "doma_shared_qar_head")

    baseline_payload, _, _, _ = _run_training(
        baseline_model, "m1", 4606, requires_grad=True
    )
    baseline_forward_rng = torch.random.get_rng_state().clone()
    diagnostic_payload, _, _, _ = _run_training(
        diagnostic_model, "m1", 4606, requires_grad=True
    )
    diagnostic_forward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(baseline_forward_rng, diagnostic_forward_rng)
    assert _state_max_abs_diff(
        baseline_model.state_dict(), diagnostic_model.state_dict()
    ) == 0.0
    baseline_loss, baseline_stats = compute_doma_object_loss(baseline_payload)
    diagnostic_loss, diagnostic_stats = compute_doma_object_loss(
        diagnostic_payload
    )
    assert torch.equal(baseline_loss, diagnostic_loss)
    assert "doma_delta_iou_count" not in baseline_stats
    assert diagnostic_stats["doma_delta_iou_count"] > 0
    delta = diagnostic_payload["scenes"][0]["delta_iou"]
    assert not delta.requires_grad and delta.grad_fn is None

    divergent_args = _research_args(
        "V2/stage1/m1.yaml", diagnostics=True
    )
    divergent_model = _Holder(divergent_args)
    with torch.no_grad():
        divergent_model.doma_shared_object_refiner.network[-1].bias[3] = 1000.0
    divergent_payload, _, _, _ = _run_training(
        divergent_model, "m1", 4607
    )
    _, divergent_stats = compute_doma_object_loss(divergent_payload)
    assert divergent_stats["doma_delta_iou_finite_count"] == 0
    assert divergent_stats["doma_delta_iou_nonfinite_count"] > 0

    baseline_model.zero_grad(set_to_none=True)
    diagnostic_model.zero_grad(set_to_none=True)
    baseline_loss.backward()
    baseline_backward_rng = torch.random.get_rng_state().clone()
    diagnostic_loss.backward()
    diagnostic_backward_rng = torch.random.get_rng_state().clone()
    assert torch.equal(baseline_backward_rng, diagnostic_backward_rng)
    assert _state_max_abs_diff(
        baseline_model.state_dict(), diagnostic_model.state_dict()
    ) == 0.0
    gradient_difference = 0.0
    for (baseline_name, baseline_parameter), (
        diagnostic_name,
        diagnostic_parameter,
    ) in zip(
        baseline_model.named_parameters(), diagnostic_model.named_parameters()
    ):
        assert baseline_name == diagnostic_name
        baseline_grad = baseline_parameter.grad
        diagnostic_grad = diagnostic_parameter.grad
        assert (baseline_grad is None) == (diagnostic_grad is None)
        if baseline_grad is not None:
            gradient_difference = max(
                gradient_difference,
                float((baseline_grad - diagnostic_grad).abs().max().item()),
            )
    assert gradient_difference == 0.0
    return {
        "synthetic_finite_nonfinite_distribution": True,
        "empty_distribution": True,
        "invalid_boxes_recorded_as_nonfinite": True,
        "divergent_end_to_end_diagnostic": True,
        "diagnostic_only_has_no_qar_head": True,
        "state_and_rng_unchanged": True,
        "post_forward_state_and_rng_unchanged": True,
        "post_backward_state_and_rng_unchanged": True,
        "loss_max_abs_diff": float(
            (baseline_loss - diagnostic_loss).abs().item()
        ),
        "gradient_max_abs_diff": gradient_difference,
        "target_detached": True,
        "batch_diagnostic_count": diagnostic_stats["doma_delta_iou_count"],
    }


def _test_inference_delta_iou_diagnostics():
    gt_centers = torch.tensor(
        [
            [0.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
            [20.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
        ]
    )
    original_centers = torch.tensor(
        [
            [0.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
            [1.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
            [20.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
            [30.0, 0.0, -1.0, 1.5, 1.6, 4.0, 0.0],
        ]
    )
    refined_centers = original_centers.clone()
    refined_centers[0] = gt_centers[1]
    refined_centers[1] = gt_centers[0]
    original_boxes = boxes_hwl_to_corners_3d(original_centers)
    refined_boxes = boxes_hwl_to_corners_3d(refined_centers)
    refined_boxes[3, 0, 0] = float("nan")
    gt_boxes = boxes_hwl_to_corners_3d(gt_centers)
    delta = compute_post_nms_delta_iou(
        original_boxes, refined_boxes, gt_boxes
    )
    assert delta.shape == (4,)
    assert not delta.requires_grad and delta.grad_fn is None
    assert abs(float(delta[0].item()) + 1.0) < 1.0e-5
    assert float(delta[1].item()) > 0.0
    assert float(delta[2].item()) == 0.0
    assert bool(torch.isnan(delta[3]))
    assert compute_post_nms_delta_iou(
        original_boxes[:0], refined_boxes[:0], gt_boxes
    ).numel() == 0
    assert compute_post_nms_delta_iou(
        original_boxes, refined_boxes, gt_boxes[:0]
    ).numel() == 0

    accumulator = DeltaIoUDiagnosticAccumulator(neutral_threshold=0.125)
    first_chunk = torch.tensor([-0.5, -0.125])
    second_chunk = torch.tensor(
        [0.0, 0.125, 0.25, 1.0, float("nan"), float("inf")]
    )
    accumulator.update(first_chunk)
    accumulator.update(second_chunk)
    values = accumulator.values()
    assert values.device.type == "cpu" and not values.requires_grad
    summary = accumulator.summary()
    finite = torch.cat((first_chunk, second_chunk))
    finite = finite[torch.isfinite(finite)].to(dtype=torch.float64)
    expected_quantiles = torch.quantile(
        finite, finite.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90])
    )
    assert summary["doma_delta_iou_count"] == 8
    assert summary["doma_delta_iou_finite_count"] == 6
    assert summary["doma_delta_iou_nonfinite_count"] == 2
    for key, expected in zip(
        ("p10", "p25", "p50", "p75", "p90"), expected_quantiles
    ):
        assert abs(summary["doma_delta_iou_%s" % key] - float(expected)) < 1.0e-12
    mean_of_batch_medians = float(
        (
            torch.quantile(first_chunk, 0.5)
            + torch.quantile(second_chunk[torch.isfinite(second_chunk)], 0.5)
        ).item()
        / 2.0
    )
    assert abs(summary["doma_delta_iou_p50"] - mean_of_batch_medians) > 1.0e-3
    assert abs(summary["doma_delta_iou_improve_ratio"] - 2.0 / 6.0) < 1.0e-12
    assert abs(summary["doma_delta_iou_worsen_ratio"] - 1.0 / 6.0) < 1.0e-12
    assert abs(summary["doma_delta_iou_neutral_ratio"] - 3.0 / 6.0) < 1.0e-12
    assert abs(summary["doma_delta_iou_abs_gt_0_01_ratio"] - 5.0 / 6.0) < 1.0e-12
    assert abs(summary["doma_delta_iou_abs_gt_0_05_ratio"] - 5.0 / 6.0) < 1.0e-12

    baseline_args = _load_args("V2/final_infer/m1m2m3m4.yaml")
    diagnostic_args = _research_args(
        "V2/final_infer/m1m2m3m4.yaml", diagnostics=True
    )
    torch.manual_seed(20260907)
    baseline_model = _Holder(baseline_args)
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260907)
    diagnostic_model = _Holder(diagnostic_args)
    diagnostic_rng = torch.random.get_rng_state().clone()
    assert torch.equal(baseline_rng, diagnostic_rng)
    assert _state_max_abs_diff(
        baseline_model.state_dict(), diagnostic_model.state_dict()
    ) == 0.0
    assert diagnostic_model.doma_flags["delta_iou_diagnostics_inference"]
    assert not hasattr(diagnostic_model, "doma_shared_qar_head")
    return {
        "post_nms_fixed_gt_match": True,
        "fixed_match_cross_gt_delta": float(delta[0].item()),
        "improve_worsen_neutral": True,
        "empty_and_nonfinite_safe": True,
        "raw_cpu_global_aggregation": True,
        "global_quantiles_not_batch_averages": True,
        "finite_count": summary["doma_delta_iou_finite_count"],
        "p50": summary["doma_delta_iou_p50"],
        "diagnostic_only_has_no_qar_head": True,
        "state_and_construction_rng_unchanged": True,
    }


def _test_qar_independent_controls():
    gate_only_stage1_args = _research_args(
        "V2/stage1/m1.yaml", qar=True
    )
    gate_only_stage1_args["doma"]["quality_aware_refinement"] = _qar_config(
        training_loss=False
    )
    gate_only_stage1 = _Holder(gate_only_stage1_args)
    assert all(
        not parameter.requires_grad
        for parameter in gate_only_stage1.doma_shared_qar_head.parameters()
    )

    gate_only_stage2_args = _research_args(
        "V2/stage2/m2.yaml", qar=True
    )
    gate_only_stage2_args["doma"]["quality_aware_refinement"] = _qar_config(
        training_loss=False
    )
    gate_only_stage2 = _Holder(gate_only_stage2_args)
    assert gate_only_stage2.doma_shared_qar_head.require_checkpoint
    _expect_raises(
        RuntimeError,
        lambda: assert_doma_qar_checkpoint_ready(gate_only_stage2),
        "checkpoint parameters were not verified",
    )
    gate_only_stage2.load_state_dict(
        copy.deepcopy(gate_only_stage2.state_dict()), strict=False
    )
    gate_only_payload, _, _, _ = _run_training(
        gate_only_stage2, "m2", 4616
    )
    assert "quality_aware_refinement_training_loss_enabled" not in gate_only_payload
    assert "qar_prediction" not in gate_only_payload["scenes"][0]
    _, gate_only_stats = compute_doma_object_loss(gate_only_payload)
    assert "doma_qar_loss" not in gate_only_stats

    loss_only_infer_args = _research_args(
        "V2/final_infer/m1m2m3m4.yaml", qar=True
    )
    loss_only_infer_args["doma"]["quality_aware_refinement"] = _qar_config(
        inference_gate=False
    )
    loss_only_infer = _Holder(loss_only_infer_args)
    assert not loss_only_infer.doma_shared_qar_head.require_checkpoint
    old_infer_state = copy.deepcopy(
        _Holder(_load_args("V2/final_infer/m1m2m3m4.yaml")).state_dict()
    )
    incompatible = loss_only_infer.load_state_dict(old_infer_state, strict=False)
    assert any(
        key.startswith("doma_shared_qar_head.")
        for key in incompatible.missing_keys
    )
    assert_doma_qar_checkpoint_ready(loss_only_infer, resume_epoch=0)
    loss_only_infer.eval()
    with torch.no_grad():
        refiner_output = loss_only_infer.doma_shared_object_refiner.network[-1]
        refiner_output.weight.zero_()
        refiner_output.bias.zero_()
        refiner_output.bias[0] = 0.2
    centers = torch.tensor(
        [[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.0]]
    )
    boxes = boxes_hwl_to_corners_3d(centers)
    scores = torch.tensor([0.8])
    output, returned_scores = refine_doma_detections(
        loss_only_infer,
        boxes,
        scores,
        _inference_context(loss_only_infer),
    )
    assert returned_scores is scores
    assert not torch.equal(output, boxes)
    return {
        "gate_only_stage1_head_frozen": True,
        "gate_only_stage2_has_no_loss_but_requires_shared_checkpoint": True,
        "loss_only_inference_has_no_gate_or_checkpoint_guard": True,
        "loss_only_inference_uses_normal_refinement": True,
    }


def _test_both_stage2():
    args = _research_args("V2/stage2/m2.yaml", opa=True, qar=True)
    args["doma"]["quality_aware_refinement"] = _qar_config(
        apply_to=["stage1_anchor", "stage2_adapt"],
        detach_features=False,
    )
    args["doma"]["loss"]["object_loss_weight"] = 0.0
    model = _Holder(args)
    with torch.no_grad():
        model.doma_shared_object_refiner.network[-1].bias[0] = 0.2
    full_state = copy.deepcopy(model.state_dict())
    incompatible = model.load_state_dict(full_state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert model.doma_shared_qar_head.checkpoint_verified
    assert_doma_qar_checkpoint_ready(model, resume_epoch=1)

    payload, _, _, _ = _run_training(model, "m2", 4404, requires_grad=True)
    loss, stats = compute_doma_object_loss(payload)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert "doma_protocol_loss" in stats
    assert "doma_qar_loss" in stats
    assert stats["doma_protocol_pair_count"] > 0
    return {
        "full_self_state_verified": True,
        "protocol_loss": stats["doma_protocol_loss"],
        "qar_loss": stats["doma_qar_loss"],
        "total_loss": float(loss.detach().item()),
        "both_stat_families_present": True,
    }


def _inference_context(model):
    generator = torch.Generator().manual_seed(4505)
    detail = torch.randn(
        1, model.doma_common_bev_channels, 32, 32, generator=generator
    )
    context_feature = torch.randn(
        1, model.doma_context_bev_channels, 16, 16, generator=generator
    )
    scene = {
        "agent_features": detail,
        "agent_support": detail.new_ones((1, 1, 32, 32)),
        "agent_modalities": ("m1",),
        "context_agent_features": context_feature,
        "context_agent_support": context_feature.new_ones((1, 1, 16, 16)),
    }
    return {"scenes": (scene,), "box_order": "hwl", "aligned_to": "ego"}


def _test_qar_inference_gate():
    args = _research_args("V2/final_infer/m1m2m3m4.yaml", qar=True)
    args["doma"]["quality_aware_refinement"] = _qar_config(
        training_loss=False
    )
    args["doma"]["quality_aware_refinement"]["inference_gate"][
        "threshold"
    ] = 0.0
    model = _Holder(args)
    full_state = copy.deepcopy(model.state_dict())
    incompatible = model.load_state_dict(full_state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert model.doma_shared_qar_head.checkpoint_verified
    assert_doma_qar_checkpoint_ready(model, resume_epoch=1)
    model.eval()
    configure_doma_trainability(model)
    with torch.no_grad():
        refiner_output = model.doma_shared_object_refiner.network[-1]
        refiner_output.weight.zero_()
        refiner_output.bias.zero_()
        refiner_output.bias[0] = 0.2
        qar_output = model.doma_shared_qar_head.network[-1]
        qar_output.weight.zero_()

    centers = torch.tensor(
        [
            [-10.0, -3.0, -1.0, 1.56, 1.6, 3.9, 0.0],
            [0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.1],
            [10.0, 3.0, -1.0, 1.56, 1.6, 3.9, -0.2],
        ]
    )
    boxes = boxes_hwl_to_corners_3d(centers)
    scores = torch.tensor([0.2, 0.9, 0.5])
    context = _inference_context(model)
    outputs = {}
    for name, bias in (
        ("positive", 2.0),
        ("zero", 0.0),
        ("negative", -2.0),
        ("nan", float("nan")),
    ):
        with torch.no_grad():
            model.doma_shared_qar_head.network[-1].bias.fill_(bias)
        output, returned_scores = refine_doma_detections(
            model, boxes, scores, context
        )
        assert returned_scores is scores
        assert output.shape == boxes.shape
        outputs[name] = output

    assert not torch.equal(outputs["positive"], boxes)
    for name in ("zero", "negative", "nan"):
        assert torch.equal(outputs[name], boxes)
    positive_centers = corners_3d_to_boxes_hwl(outputs["positive"])
    assert bool((positive_centers[:, 0] > centers[:, 0]).all())
    with torch.no_grad():
        model.doma_shared_qar_head.network[-1].bias.fill_(2.0)
        model.doma_shared_object_refiner.network[-1].bias[3] = 1000.0
    nonfinite_output, returned_scores = refine_doma_detections(
        model, boxes, scores, context
    )
    assert returned_scores is scores
    assert torch.equal(nonfinite_output, boxes)
    with torch.no_grad():
        model.doma_shared_object_refiner.network[-1].bias[3] = -1000.0
    zero_size_output, returned_scores = refine_doma_detections(
        model, boxes, scores, context
    )
    assert returned_scores is scores
    assert torch.equal(zero_size_output, boxes)
    return {
        "positive_applied_count": int(
            (positive_centers[:, 0] > centers[:, 0]).sum().item()
        ),
        "zero_fallback": True,
        "negative_fallback": True,
        "nan_fallback": True,
        "nonfinite_refinement_fallback": True,
        "nonpositive_size_fallback": True,
        "score_object_identity": True,
        "count_and_order_preserved": True,
    }


def _test_checkpoint_contract():
    old_args = _load_args("V2/stage2/m2.yaml")
    disabled_args = copy.deepcopy(old_args)
    disabled_args["doma"]["quality_aware_refinement"] = {"enabled": False}
    old_model = _Holder(old_args)
    disabled_model = _Holder(disabled_args)
    incompatible = disabled_model.load_state_dict(
        copy.deepcopy(old_model.state_dict()), strict=False
    )
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert_doma_qar_checkpoint_ready(disabled_model, resume_epoch=0)

    mode_results = {}
    for mode, relative_path in (
        ("stage2_adapt", "V2/stage2/m2.yaml"),
        ("inference", "V2/final_infer/m1m2m3m4.yaml"),
    ):
        base_args = _load_args(relative_path)
        enabled_args = _research_args(relative_path, qar=True)
        old_state = copy.deepcopy(_Holder(base_args).state_dict())

        missing_model = _Holder(enabled_args)
        _expect_raises(
            RuntimeError,
            lambda: missing_model.load_state_dict(old_state, strict=False),
            "checkpoint is missing Stage1-owned parameters",
        )

        partial_model = _Holder(enabled_args)
        partial_state = copy.deepcopy(partial_model.state_dict())
        missing_key = next(
            key for key in partial_state if key.startswith("doma_shared_qar_head.")
        )
        del partial_state[missing_key]
        _expect_raises(
            RuntimeError,
            lambda: partial_model.load_state_dict(partial_state, strict=False),
            "checkpoint is missing Stage1-owned parameters",
        )

        full_model = _Holder(enabled_args)
        full_state = copy.deepcopy(full_model.state_dict())
        incompatible = full_model.load_state_dict(full_state, strict=False)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        assert full_model.doma_shared_qar_head.checkpoint_verified
        assert_doma_qar_checkpoint_ready(full_model, resume_epoch=1)
        assert_doma_qar_checkpoint_ready(full_model, resume_epoch=0)
        mode_results[mode] = {
            "old_rejected": True,
            "partial_rejected": True,
            "full_accepted": True,
            "verified_epoch_zero_accepted": True,
        }

    malformed = _Holder(_research_args("V2/stage2/m2.yaml", qar=True))
    malformed_state = copy.deepcopy(malformed.state_dict())
    incompatible = malformed.load_state_dict(malformed_state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert malformed.doma_shared_qar_head.checkpoint_verified
    malformed_key = next(
        key for key in malformed_state if key.startswith("doma_shared_qar_head.")
    )
    malformed_state[malformed_key] = malformed_state[malformed_key].reshape(-1)[:1]
    _expect_raises(
        RuntimeError,
        lambda: malformed.load_state_dict(malformed_state, strict=False),
        "size mismatch",
    )
    assert not malformed.doma_shared_qar_head.checkpoint_verified

    unverified = _Holder(_research_args("V2/stage2/m2.yaml", qar=True))
    _expect_raises(
        RuntimeError,
        lambda: assert_doma_qar_checkpoint_ready(unverified),
        "checkpoint parameters were not verified",
    )
    _expect_raises(
        RuntimeError,
        lambda: assert_doma_qar_checkpoint_ready(unverified, resume_epoch=0),
        "no checkpoint was loaded",
    )
    head = unverified.doma_shared_qar_head
    _expect_raises(
        RuntimeError,
        lambda: head(
            torch.zeros((1, 128)),
            torch.zeros((1, 32)),
            torch.zeros((1, 8)),
        ),
        "parameters have not been loaded",
    )
    return {
        "disabled_old_state_load": True,
        "enabled_modes": mode_results,
        "no_checkpoint_helper_rejected": True,
        "stage2_forward_before_load_rejected": True,
        "malformed_load_does_not_verify": True,
    }


def _optimizer_hypes():
    return {
        "optimizer": {
            "core_method": "Adam",
            "lr": 0.001,
            "args": {"weight_decay": 1.0e-4},
            "doma_param_groups": {"enabled": True, "weight_decay": 0.0},
        }
    }


def _optimizer_ids(optimizer):
    return [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _assert_optimizer_coverage(model, optimizer):
    observed = _optimizer_ids(optimizer)
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert len(observed) == len(set(observed))
    assert set(observed) == expected


def _test_optimizer_contract():
    stage1 = _Holder(_research_args("V2/stage1/m1.yaml", qar=True))
    with contextlib.redirect_stdout(io.StringIO()):
        stage1_optimizer = train_utils.setup_optimizer(
            _optimizer_hypes(), stage1
        )
    _assert_optimizer_coverage(stage1, stage1_optimizer)
    stage1_doma_ids = {
        id(parameter) for parameter in stage1_optimizer.param_groups[1]["params"]
    }
    qar_parameters = tuple(stage1.doma_shared_qar_head.parameters())
    assert qar_parameters and all(parameter.requires_grad for parameter in qar_parameters)
    assert all(id(parameter) in stage1_doma_ids for parameter in qar_parameters)
    flattened = _optimizer_ids(stage1_optimizer)
    assert all(flattened.count(id(parameter)) == 1 for parameter in qar_parameters)

    stage2 = _Holder(_research_args("V2/stage2/m2.yaml", qar=True))
    full_state = copy.deepcopy(stage2.state_dict())
    stage2.load_state_dict(full_state, strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        stage2_optimizer = train_utils.setup_optimizer(
            _optimizer_hypes(), stage2
        )
    _assert_optimizer_coverage(stage2, stage2_optimizer)
    stage2_optimizer_ids = set(_optimizer_ids(stage2_optimizer))
    stage2_qar = tuple(stage2.doma_shared_qar_head.parameters())
    assert stage2_qar and all(not parameter.requires_grad for parameter in stage2_qar)
    assert all(id(parameter) not in stage2_optimizer_ids for parameter in stage2_qar)
    return {
        "stage1_qar_parameter_tensors": len(qar_parameters),
        "stage1_qar_in_doma_group_exactly_once": True,
        "stage2_qar_frozen_and_excluded": True,
        "trainable_coverage_exactly_once": True,
        "duplicates": 0,
    }


def _test_config_chain_smoke():
    common_qar = _qar_config(
        inference_gate=False,
        apply_to=["stage1_anchor"],
        detach_features=True,
    )
    common_opa = _opa_config(detail=True, context=False)

    def chain_args(relative_path):
        args = _load_args(relative_path)
        args["doma"]["ablation"] = True
        args["doma"]["quality_aware_refinement"] = copy.deepcopy(common_qar)
        args["doma"]["object_protocol_alignment"] = copy.deepcopy(common_opa)
        return args

    stage1_args = chain_args("V2/stage1/m1.yaml")
    stage1_fingerprint = doma_method_fingerprint(stage1_args["doma"])
    stage1_model = _Holder(stage1_args)
    assert stage1_model.doma_flags["quality_aware_refinement_training_loss"]
    assert not stage1_model.doma_flags["object_protocol_alignment"]
    with torch.no_grad():
        stage1_model.doma_shared_object_refiner.network[-1].bias[0] = 0.25
    stage1_payload, _, _, _ = _run_training(
        stage1_model, "m1", 4801, requires_grad=True
    )
    stage1_loss, stage1_stats = compute_doma_object_loss(stage1_payload)
    stage1_model.zero_grad(set_to_none=True)
    stage1_loss.backward()
    assert _module_grad_l1(stage1_model.doma_shared_qar_head) > 0.0
    assert "doma_qar_loss" in stage1_stats
    with contextlib.redirect_stdout(io.StringIO()):
        stage1_optimizer = train_utils.setup_optimizer(
            _optimizer_hypes(), stage1_model
        )
    _assert_optimizer_coverage(stage1_model, stage1_optimizer)
    stage1_optimizer.step()
    with torch.no_grad():
        stage1_qar_output = stage1_model.doma_shared_qar_head.network[-1]
        stage1_qar_output.weight.zero_()
        stage1_qar_output.bias.fill_(0.2)
    stage1_state = copy.deepcopy(stage1_model.state_dict())

    stage2_states = []
    stage2_configs = []
    stage2_reports = {}
    for offset, modality in enumerate(("m2", "m3", "m4")):
        args = chain_args("V2/stage2/%s.yaml" % modality)
        assert doma_method_fingerprint(args["doma"]) == stage1_fingerprint
        stage2_configs.append(copy.deepcopy(args["doma"]))
        model = _Holder(args)
        assert not model.doma_flags["quality_aware_refinement_training_loss"]
        assert model.doma_flags["object_protocol_alignment"]
        state = copy.deepcopy(model.state_dict())
        for key, value in stage1_state.items():
            if key.startswith(SHARED_PREFIXES):
                assert key in state
                state[key] = value.clone()
        incompatible = model.load_state_dict(state, strict=False)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        assert_doma_qar_checkpoint_ready(model, resume_epoch=1)
        assert all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith(SHARED_PREFIXES)
        )
        payload, _, _, _ = _run_training(
            model, modality, 4810 + offset, requires_grad=True
        )
        stage2_loss, stats = compute_doma_object_loss(payload)
        assert "doma_qar_loss" not in stats
        assert stats["doma_protocol_pair_count"] == 1
        model.zero_grad(set_to_none=True)
        stage2_loss.backward()
        active_adapter_grad = _module_grad_l1(
            getattr(model, "doma_object_adapter_%s" % modality)
        )
        assert active_adapter_grad > 0.0
        pre_step_state = copy.deepcopy(model.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            stage2_optimizer = train_utils.setup_optimizer(
                _optimizer_hypes(), model
            )
        _assert_optimizer_coverage(model, stage2_optimizer)
        stage2_optimizer.step()
        state = copy.deepcopy(model.state_dict())
        for key in stage1_state:
            if key.startswith(SHARED_PREFIXES):
                assert torch.equal(state[key], stage1_state[key])
        adapter_prefix = "doma_object_adapter_%s." % modality
        assert any(
            not torch.equal(state[key], pre_step_state[key])
            for key in state
            if key.startswith(adapter_prefix)
        )
        stage2_states.append(state)
        stage2_reports[modality] = {
            "qar_training_loss": False,
            "opa_detail_pair_count": stats["doma_protocol_pair_count"],
            "active_adapter_grad_l1": active_adapter_grad,
            "optimizer_step": True,
            "shared_checkpoint_verified": True,
        }

    merged = apply_doma_merge_ownership(
        OrderedDict(), stage2_states + [stage1_state]
    )
    qar_keys = [
        key for key in stage1_state if key.startswith("doma_shared_qar_head.")
    ]
    assert qar_keys and all(
        torch.equal(merged[key], stage1_state[key]) for key in qar_keys
    )
    for modality, source in zip(("m2", "m3", "m4"), stage2_states):
        prefix = "doma_object_adapter_%s." % modality
        keys = [key for key in source if key.startswith(prefix)]
        assert keys and all(torch.equal(merged[key], source[key]) for key in keys)

    final_args = _load_args("V2/final_infer/m1m2m3m4.yaml")
    final_args["doma"]["ablation"] = True
    final_args["doma"]["quality_aware_refinement"] = _qar_config(
        training_loss=False,
        inference_gate=True,
        threshold=0.0,
    )
    final_model = _Holder(final_args)
    incompatible = final_model.load_state_dict(merged, strict=False)
    assert all(key == "base_probe.weight" or key == "base_probe.bias"
               for key in incompatible.missing_keys)
    assert not incompatible.unexpected_keys
    assert_doma_qar_checkpoint_ready(final_model, resume_epoch=1)
    assert final_model.doma_flags["quality_aware_refinement_inference_gate"]
    assert not final_model.doma_flags["quality_aware_refinement_training_loss"]
    assert run_doma_training(final_model, None, {}) is None
    final_model.eval()
    centers = torch.tensor(
        [[0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.0]]
    )
    boxes = boxes_hwl_to_corners_3d(centers)
    scores = torch.tensor([0.9])
    final_boxes, final_scores = refine_doma_detections(
        final_model,
        boxes,
        scores,
        _inference_context(final_model),
    )
    assert final_scores is scores
    assert not torch.equal(final_boxes, boxes)

    consumer_only = copy.deepcopy(stage1_args["doma"])
    consumer_only["quality_aware_refinement"]["inference_gate"] = {
        "enabled": True,
        "mode": "hard",
        "threshold": 0.25,
    }
    consumer_only["delta_iou_diagnostics"] = {
        "enabled": True,
        "apply_to": ["inference"],
        "neutral_threshold": 1.0e-5,
    }
    consumer_snapshot = copy.deepcopy(consumer_only)
    assert doma_method_fingerprint(consumer_only) == stage1_fingerprint
    assert consumer_only == consumer_snapshot
    structural_change = copy.deepcopy(stage1_args["doma"])
    structural_change["quality_aware_refinement"]["hidden_dim"] += 1
    assert doma_method_fingerprint(structural_change) != stage1_fingerprint
    training_change = copy.deepcopy(stage1_args["doma"])
    training_change["quality_aware_refinement"]["training_loss"][
        "detach_features"
    ] = False
    assert doma_method_fingerprint(training_change) != stage1_fingerprint
    ordered_a = copy.deepcopy(stage1_args["doma"])
    ordered_a["quality_aware_refinement"]["training_loss"]["apply_to"] = [
        "stage1_anchor",
        "stage2_adapt",
    ]
    ordered_a["quality_aware_refinement"]["training_loss"][
        "detach_features"
    ] = False
    ordered_b = copy.deepcopy(ordered_a)
    ordered_b["quality_aware_refinement"]["training_loss"]["apply_to"] = [
        "stage2_adapt",
        "stage1_anchor",
    ]
    assert doma_method_fingerprint(ordered_a) == doma_method_fingerprint(
        ordered_b
    )
    assert doma_method_fingerprint(ordered_a) != stage1_fingerprint

    integration_configs = copy.deepcopy(
        stage2_configs + [stage1_args["doma"]]
    )
    integration_configs[0]["quality_aware_refinement"]["inference_gate"] = {
        "enabled": True,
        "mode": "hard",
        "threshold": 0.5,
    }
    integration_configs[1]["delta_iou_diagnostics"] = {
        "enabled": True,
        "apply_to": ["inference"],
        "neutral_threshold": 1.0e-6,
    }
    with tempfile.TemporaryDirectory(prefix="doma_config_chain_") as root:
        model_dirs = []
        for index, config in enumerate(integration_configs):
            model_dir = Path(root) / str(index)
            model_dir.mkdir()
            with open(model_dir / "config.yaml", "w") as stream:
                yaml.safe_dump(
                    {"model": {"args": {"doma": config}}}, stream
                )
            model_dirs.append(str(model_dir))
        _validate_config_fingerprints(model_dirs)
        incompatible_config = copy.deepcopy(integration_configs[0])
        incompatible_config["quality_aware_refinement"]["hidden_dim"] += 1
        with open(Path(model_dirs[0]) / "config.yaml", "w") as stream:
            yaml.safe_dump(
                {"model": {"args": {"doma": incompatible_config}}}, stream
            )
        _expect_raises(
            RuntimeError,
            lambda: _validate_config_fingerprints(model_dirs),
            "method configs differ",
        )
    return {
        "stage1": {
            "qar_training_loss": True,
            "qar_head_gradient": True,
            "optimizer_step": True,
        },
        "stage2": stage2_reports,
        "merge": {
            "stage1_owns_shared_qar": True,
            "stage2_owns_each_adapter": True,
            "frozen_shared_equal": True,
        },
        "final_inference": {
            "training_loss": False,
            "gate_called": True,
            "checkpoint_guard": True,
        },
        "fingerprint": {
            "runtime_modes_compatible": True,
            "consumer_only_fields_ignored": True,
            "apply_to_order_normalized": True,
            "config_yaml_integration": True,
            "normalization_input_not_mutated": True,
            "training_semantics_retained": True,
            "structural_fields_retained": True,
        },
    }


def _test_merge_prefix():
    assert "doma_shared_qar_head." in SHARED_PREFIXES
    assert "doma_shared_qar_head." in SHARED_DOMA_PREFIXES
    stage1_model = _Holder(_research_args("V2/stage1/m1.yaml", qar=True))
    stage1_state = copy.deepcopy(stage1_model.state_dict())
    stage2_states = []
    for modality in ("m2", "m3", "m4"):
        model = _Holder(
            _research_args("V2/stage2/%s.yaml" % modality, qar=True)
        )
        state = copy.deepcopy(model.state_dict())
        for key, value in stage1_state.items():
            if key.startswith(SHARED_PREFIXES):
                assert key in state
                state[key] = value.clone()
        stage2_states.append(state)
    ordered = stage2_states + [stage1_state]
    merged = apply_doma_merge_ownership(OrderedDict(), ordered)
    qar_keys = [
        key for key in stage1_state if key.startswith("doma_shared_qar_head.")
    ]
    assert qar_keys
    assert all(key in merged for key in qar_keys)
    assert all(torch.equal(merged[key], stage1_state[key]) for key in qar_keys)

    incomplete = copy.deepcopy(ordered)
    del incomplete[0][qar_keys[0]]
    _expect_raises(
        RuntimeError,
        lambda: apply_doma_merge_ownership(OrderedDict(), incomplete),
        "shared-key contract differs",
    )
    return {
        "qar_shared_prefix_registered": True,
        "qar_shared_key_count": len(qar_keys),
        "stage1_owns_merged_qar": True,
        "partial_stage2_qar_rejected": True,
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260904)
    report = {
        "config_contract": _test_config_contract(),
        "qar_apply_to": _test_qar_apply_to(),
        "disabled_parity": _test_disabled_parity(),
        "opa_stage2": _test_opa_stage2(),
        "opa_branch_specific_validity": _test_opa_branch_specific_validity(),
        "qar_stage1": _test_qar_stage1(),
        "qar_feature_detach": _test_qar_feature_detach(),
        "delta_iou_diagnostics": _test_delta_iou_diagnostics(),
        "inference_delta_iou_diagnostics": (
            _test_inference_delta_iou_diagnostics()
        ),
        "qar_independent_controls": _test_qar_independent_controls(),
        "opa_qar_stage2": _test_both_stage2(),
        "qar_inference_gate": _test_qar_inference_gate(),
        "checkpoint_contract": _test_checkpoint_contract(),
        "optimizer_contract": _test_optimizer_contract(),
        "config_chain_smoke": _test_config_chain_smoke(),
        "merge_prefix": _test_merge_prefix(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DOMA research modules acceptance: PASS")


if __name__ == "__main__":
    main()
