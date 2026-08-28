"""In-memory acceptance checks for DOMA Stage2 ownership and merge safety."""

import copy
import json
from collections import OrderedDict

import torch

from opencood.models.sub_modules.doma_object import (
    configure_doma_trainability,
    install_doma_modules,
)
from opencood.tools.check_doma_static import DOMA_YAML_ROOT, _DOMAHolder, _load_yaml
from opencood.tools.doma_tools import SHARED_PREFIXES, apply_doma_merge_ownership


EXPECTED_STAGE2 = {
    "V1": (146696, 30, 8448),
    "V2": (357257, 53, 41728),
    "V3": (367754, 57, 41728),
}
EXPECTED_MERGED = {
    "V1": (163592, 42, 0),
    "V2": (440713, 77, 0),
    "V3": (451210, 81, 0),
}


def _build(relative_path):
    hypes = _load_yaml(DOMA_YAML_ROOT / relative_path)
    holder = _DOMAHolder(hypes["model"]["args"])
    install_doma_modules(holder, hypes["model"]["args"])
    holder._doma_log_printed = True
    configure_doma_trainability(holder)
    return holder


def _counts(holder):
    return (
        sum(parameter.numel() for parameter in holder.parameters()),
        len(holder.state_dict()),
        sum(
            parameter.numel()
            for parameter in holder.parameters()
            if parameter.requires_grad
        ),
    )


def _stage2_state_with_loaded_shared(stage2_holder, stage1_state):
    state = copy.deepcopy(stage2_holder.state_dict())
    for key, value in stage1_state.items():
        if key.startswith(SHARED_PREFIXES):
            if key not in state:
                raise AssertionError("Stage2 is missing Stage1 shared key %s" % key)
            state[key] = value.clone()
    return state


def _expect_runtime_error(callable_value, message):
    try:
        callable_value()
    except RuntimeError:
        return
    raise AssertionError(message)


def main():
    report = {}
    for version in ("V1", "V2", "V3"):
        stage1 = _build(version + "/stage1/m1.yaml")
        stage1_state = copy.deepcopy(stage1.state_dict())
        stage2_holders = [
            _build(version + "/stage2/%s.yaml" % modality)
            for modality in ("m2", "m3", "m4")
        ]
        for holder in stage2_holders:
            if _counts(holder) != EXPECTED_STAGE2[version]:
                raise AssertionError("%s Stage2 count mismatch" % version)
        stage2_states = [
            _stage2_state_with_loaded_shared(holder, stage1_state)
            for holder in stage2_holders
        ]
        ordered = stage2_states + [stage1_state]
        merged = apply_doma_merge_ownership(OrderedDict(), ordered)
        final_holder = _build(version + "/final_infer/m1m2m3m4.yaml")
        if _counts(final_holder) != EXPECTED_MERGED[version]:
            raise AssertionError("%s merged count mismatch" % version)
        if set(merged) != set(final_holder.state_dict()):
            raise AssertionError("%s merged state-key set mismatch" % version)

        perturbed = copy.deepcopy(ordered)
        shared_key = next(
            key for key in stage1_state if key.startswith(SHARED_PREFIXES)
        )
        perturbed[0][shared_key] = perturbed[0][shared_key] + 1
        _expect_runtime_error(
            lambda: apply_doma_merge_ownership(OrderedDict(), perturbed),
            "%s accepted a changed frozen Stage1 tensor" % version,
        )

        incomplete = copy.deepcopy(ordered)
        adapter_prefix = "doma_object_adapter_m2."
        adapter_key = next(key for key in incomplete[0] if key.startswith(adapter_prefix))
        del incomplete[0][adapter_key]
        _expect_runtime_error(
            lambda: apply_doma_merge_ownership(OrderedDict(), incomplete),
            "%s accepted an incomplete adapter" % version,
        )
        report[version] = {
            "stage2": EXPECTED_STAGE2[version],
            "merged": EXPECTED_MERGED[version],
            "merged_key_count": len(merged),
            "rejects_changed_shared_tensor": True,
            "rejects_incomplete_adapter": True,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DOMA merge ownership acceptance: PASS")


if __name__ == "__main__":
    main()
