"""CPU acceptance checks for opt-in DOMA optimizer parameter groups.

Optional dependency stubs are construction-only and are never used for a
model forward.  Existing experiment YAML files are loaded read-only; the
optimizer opt-in block is added only to in-memory copies.
"""

import contextlib
import copy
import io
import json

import torch
import torch.nn as nn

from opencood.tools import train_utils
from opencood.tools.check_doma_model_counts import (
    _install_optional_import_stubs,
)


_install_optional_import_stubs()

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.doma_heter_pyramid_collab import DOMAHeterPyramidCollab
from opencood.models.doma_heter_pyramid_single import DOMAHeterPyramidSingle


ROOT = "opencood/hypes_yaml/opv2v/MoreModality/DOMA"
BASE_WEIGHT_DECAY = 1.0e-4
DOMA_WEIGHT_DECAY = 0.0
LR = 0.002
BETAS = (0.85, 0.95)


class _LegacyFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 2)
        self.doma_probe = nn.Linear(2, 1)
        self.frozen_probe = nn.Parameter(torch.ones(1), requires_grad=False)
        self.parameters_calls = 0

    def parameters(self, recurse=True):
        self.parameters_calls += 1
        return super().parameters(recurse=recurse)


def _optimizer_hypes(doma_config=None):
    optimizer = {
        "core_method": "Adam",
        "lr": LR,
        "args": {
            "eps": 1.0e-10,
            "weight_decay": BASE_WEIGHT_DECAY,
            "betas": BETAS,
            "amsgrad": True,
        },
    }
    if doma_config is not None:
        optimizer["doma_param_groups"] = doma_config
    return {"optimizer": optimizer}


def _optimizer_param_ids(optimizer):
    return [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _snapshot_requires_grad(model):
    return {
        id(parameter): parameter.requires_grad
        for parameter in nn.Module.parameters(model)
    }


def _assert_requires_grad_unchanged(model, before):
    after = _snapshot_requires_grad(model)
    assert after == before


def _assert_trainable_coverage(model, optimizer):
    expected = {
        id(parameter)
        for parameter in nn.Module.parameters(model)
        if parameter.requires_grad
    }
    observed = _optimizer_param_ids(optimizer)
    assert len(observed) == len(set(observed))
    assert set(observed) == expected


def _assert_name_partition(model, optimizer):
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    base_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    doma_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert all(not names_by_id[parameter_id].startswith("doma_")
               for parameter_id in base_ids)
    assert all(names_by_id[parameter_id].startswith("doma_")
               for parameter_id in doma_ids)


def _assert_group_options(optimizer, expected_lr):
    assert len(optimizer.param_groups) == 2
    base_group, doma_group = optimizer.param_groups
    assert base_group["lr"] == expected_lr
    assert doma_group["lr"] == expected_lr
    assert base_group["weight_decay"] == BASE_WEIGHT_DECAY
    assert doma_group["weight_decay"] == DOMA_WEIGHT_DECAY
    for group in optimizer.param_groups:
        assert tuple(group["betas"]) == BETAS
        assert group["eps"] == 1.0e-10
        assert group["amsgrad"] is True
    base_ids = {id(parameter) for parameter in base_group["params"]}
    doma_ids = {id(parameter) for parameter in doma_group["params"]}
    assert not base_ids & doma_ids


def _test_legacy_equivalence():
    model = _LegacyFixture()
    expected_ids = [id(parameter) for parameter in nn.Module.parameters(model)]
    requires_grad_before = _snapshot_requires_grad(model)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        missing = train_utils.setup_optimizer(_optimizer_hypes(), model)
    assert output.getvalue() == ""
    assert model.parameters_calls == 1

    false_hypes = _optimizer_hypes(
        {"enabled": False, "weight_decay": DOMA_WEIGHT_DECAY}
    )
    with contextlib.redirect_stdout(output):
        disabled = train_utils.setup_optimizer(false_hypes, model)
    assert output.getvalue() == ""
    assert model.parameters_calls == 2

    implicit_false_hypes = _optimizer_hypes(
        {"weight_decay": DOMA_WEIGHT_DECAY}
    )
    with contextlib.redirect_stdout(output):
        implicit_false = train_utils.setup_optimizer(implicit_false_hypes, model)
    assert output.getvalue() == ""
    assert model.parameters_calls == 3

    for optimizer in (missing, disabled, implicit_false):
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert [id(parameter) for parameter in group["params"]] == expected_ids
        assert group["lr"] == LR
        assert group["weight_decay"] == BASE_WEIGHT_DECAY
        assert tuple(group["betas"]) == BETAS
        assert group["eps"] == 1.0e-10
        assert group["amsgrad"] is True
    assert missing.defaults == disabled.defaults == implicit_false.defaults
    _assert_requires_grad_unchanged(model, requires_grad_before)
    return {
        "optimizer": "Adam",
        "groups": 1,
        "parameters_including_frozen": len(expected_ids),
        "missing_equals_disabled": True,
        "missing_enabled_defaults_disabled": True,
        "legacy_model_parameters_call": True,
    }


def _load_stage_hypes(relative_path):
    hypes = load_yaml("%s/%s" % (ROOT, relative_path))
    hypes = copy.deepcopy(hypes)
    hypes["optimizer"]["args"]["betas"] = BETAS
    hypes["optimizer"]["args"]["amsgrad"] = True
    hypes["optimizer"]["doma_param_groups"] = {
        "enabled": True,
        "weight_decay": DOMA_WEIGHT_DECAY,
    }
    return hypes


def _construct_silently(model_type, hypes):
    with contextlib.redirect_stdout(io.StringIO()):
        return model_type(hypes["model"]["args"])


def _assert_prefix_membership(model, group_ids, prefixes):
    named = tuple(model.named_parameters())
    for prefix in prefixes:
        matching = [
            parameter
            for name, parameter in named
            if name.startswith(prefix)
        ]
        assert matching, "no parameters found for %s" % prefix
        assert all(parameter.requires_grad for parameter in matching)
        assert all(id(parameter) in group_ids for parameter in matching)


def _test_stage1():
    hypes = _load_stage_hypes("V2/stage1/m1.yaml")
    model = _construct_silently(DOMAHeterPyramidCollab, hypes)
    requires_grad_before = _snapshot_requires_grad(model)
    optimizer = train_utils.setup_optimizer(hypes, model)
    _assert_group_options(optimizer, hypes["optimizer"]["lr"])
    _assert_trainable_coverage(model, optimizer)
    _assert_name_partition(model, optimizer)
    _assert_requires_grad_unchanged(model, requires_grad_before)

    base_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    doma_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    required_doma = (
        "doma_shared_object_encoder.",
        "doma_shared_geometry_encoder.",
        "doma_shared_object_refiner.",
        "doma_shared_context_encoder.",
        "doma_shared_multigranularity_fusion.",
    )
    _assert_prefix_membership(model, doma_ids, required_doma)
    assert all(
        not name.startswith("doma_")
        for name, parameter in model.named_parameters()
        if id(parameter) in base_ids
    )

    default_hypes = copy.deepcopy(hypes)
    default_hypes["optimizer"]["doma_param_groups"] = {"enabled": True}
    default_optimizer = train_utils.setup_optimizer(default_hypes, model)
    assert default_optimizer.param_groups[1]["weight_decay"] == 0.0
    return {
        "groups": 2,
        "base_numel": sum(parameter.numel() for parameter in optimizer.param_groups[0]["params"]),
        "doma_numel": sum(parameter.numel() for parameter in optimizer.param_groups[1]["params"]),
        "required_doma_modules": list(required_doma),
        "default_doma_weight_decay": 0.0,
    }


def _test_stage2_m3():
    hypes = _load_stage_hypes("V2/stage2/m3.yaml")
    model = _construct_silently(DOMAHeterPyramidSingle, hypes)
    # The construction-only spconv stub has no parameters. Add a diagnostic
    # probe only when needed so encoder_m3's name classification is exercised.
    if not any(parameter.requires_grad for parameter in model.encoder_m3.parameters()):
        model.encoder_m3.register_parameter(
            "optimizer_group_probe", nn.Parameter(torch.ones(1)))

    requires_grad_before = _snapshot_requires_grad(model)
    optimizer = train_utils.setup_optimizer(hypes, model)
    _assert_group_options(optimizer, hypes["optimizer"]["lr"])
    _assert_trainable_coverage(model, optimizer)
    _assert_name_partition(model, optimizer)
    _assert_requires_grad_unchanged(model, requires_grad_before)

    base_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    doma_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    _assert_prefix_membership(
        model,
        base_ids,
        ("encoder_m3.", "backbone_m3.", "aligner_m3."),
    )
    _assert_prefix_membership(
        model,
        doma_ids,
        ("doma_object_adapter_m3.", "doma_context_adapter_m3."),
    )

    shared_doma = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("doma_shared_")
    ]
    assert shared_doma
    assert all(not parameter.requires_grad for _, parameter in shared_doma)
    optimizer_ids = set(_optimizer_param_ids(optimizer))
    assert all(id(parameter) not in optimizer_ids for _, parameter in shared_doma)

    scheduler_hypes = {
        "lr_scheduler": {
            "core_method": "multistep",
            "step_size": [1],
            "gamma": 0.1,
        }
    }
    weight_decays_before = [
        group["weight_decay"] for group in optimizer.param_groups
    ]
    scheduler = train_utils.setup_lr_schedular(scheduler_hypes, optimizer)
    optimizer.step()
    scheduler.step()
    expected_lr = hypes["optimizer"]["lr"] * 0.1
    assert scheduler.get_last_lr() == [expected_lr, expected_lr]
    assert [group["weight_decay"] for group in optimizer.param_groups] \
        == weight_decays_before
    return {
        "groups": 2,
        "base_modules": ["encoder_m3", "backbone_m3", "aligner_m3"],
        "doma_modules": [
            "doma_object_adapter_m3",
            "doma_context_adapter_m3",
        ],
        "shared_doma_frozen_and_excluded": True,
        "scheduler_lrs_after_milestone": scheduler.get_last_lr(),
        "scheduler_weight_decays_unchanged": True,
    }


def _test_no_doma_fail_closed():
    model = nn.Linear(2, 1)
    try:
        train_utils.setup_optimizer(
            _optimizer_hypes({"enabled": True, "weight_decay": 0.0}),
            model,
        )
    except RuntimeError as error:
        assert "no trainable doma_ parameters" in str(error)
        return True
    raise AssertionError("enabled DOMA optimizer groups accepted a model without DOMA")


def main():
    torch.manual_seed(20260902)
    report = {
        "test_1_and_2_legacy": _test_legacy_equivalence(),
        "test_3_stage1": _test_stage1(),
        "test_4_stage2_m3": _test_stage2_m3(),
        "test_5_uniqueness": {
            "trainable_parameter_coverage": "exactly_once",
            "requires_grad_unchanged": True,
        },
        "enabled_without_doma_fails_closed": _test_no_doma_fail_closed(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DOMA optimizer parameter-group acceptance: PASS")


if __name__ == "__main__":
    main()
