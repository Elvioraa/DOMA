"""DOMA checkpoint merge with explicit Stage1/Stage2 ownership."""

import copy
import json
import os
import sys
from collections import OrderedDict

import torch
import yaml

from opencood.models.sub_modules.doma_config import VALID_TRAINING_MODES
from opencood.tools.heal_tools import get_model_path_from_dir, merge_dict


SHARED_PREFIXES = (
    "doma_shared_object_encoder.",
    "doma_shared_geometry_encoder.",
    "doma_shared_object_refiner.",
    "doma_shared_context_encoder.",
    "doma_shared_multigranularity_fusion.",
    "doma_shared_quality_head.",
    "doma_shared_qar_head.",
)
ADAPTER_SUFFIXES = (
    "delta.0.weight",
    "delta.0.bias",
    "delta.1.weight",
    "delta.1.bias",
    "delta.3.weight",
    "delta.3.bias",
)


def apply_doma_merge_ownership(merged_dict, ordered_stage_dicts):
    """Overlay DOMA keys for checkpoints ordered as m2, m3, m4, Stage1 m1."""
    if not any(
        key.startswith("doma_")
        for state_dict in ordered_stage_dicts
        for key in state_dict
    ):
        return merged_dict
    if len(ordered_stage_dicts) != 4:
        raise RuntimeError("DOMA merge_final requires m2, m3, m4, m1 order")

    stage2_by_modality = {
        "m2": ordered_stage_dicts[0],
        "m3": ordered_stage_dicts[1],
        "m4": ordered_stage_dicts[2],
    }
    stage1 = ordered_stage_dicts[3]
    result = OrderedDict(merged_dict)
    stage1_shared = {key for key in stage1 if key.startswith(SHARED_PREFIXES)}
    if not stage1_shared:
        raise RuntimeError("Stage1 m1 checkpoint has no DOMA shared parameters")

    for modality, source in stage2_by_modality.items():
        source_shared = {key for key in source if key.startswith(SHARED_PREFIXES)}
        if source_shared != stage1_shared:
            missing = sorted(stage1_shared - source_shared)
            extra = sorted(source_shared - stage1_shared)
            raise RuntimeError(
                "Stage2 %s DOMA shared-key contract differs from Stage1; missing=%s extra=%s"
                % (modality, missing, extra)
            )
        unequal = [
            key
            for key in stage1_shared
            if not torch.equal(source[key], stage1[key])
        ]
        if unequal:
            raise RuntimeError(
                "Stage2 %s changed frozen Stage1-owned DOMA parameters: %s"
                % (modality, ", ".join(sorted(unequal)))
            )

    for key in [key for key in result if key.startswith("doma_")]:
        del result[key]
    for key in sorted(stage1_shared):
        result[key] = stage1[key]

    require_context = any(
        key.startswith(
            (
                "doma_shared_context_encoder.",
                "doma_shared_multigranularity_fusion.",
            )
        )
        for key in stage1_shared
    )
    for modality, source in stage2_by_modality.items():
        _copy_owned_prefix(
            result,
            source,
            "doma_object_adapter_%s." % modality,
            "stage2/%s" % modality,
        )
        context_prefix = "doma_context_adapter_%s." % modality
        if require_context:
            _copy_owned_prefix(
                result, source, context_prefix, "stage2/%s" % modality
            )
        elif any(key.startswith(context_prefix) for key in source):
            raise RuntimeError(
                "Stage2 %s unexpectedly contains a higher-version Context adapter"
                % modality
            )
    return result


def merge_and_save_final(aligned_model_dir_list, output_model_dir):
    """Run Official HEAL merge order, then enforce DOMA ownership."""
    if len(aligned_model_dir_list) != 4:
        raise ValueError("expected model directories in m2, m3, m4, m1 order")
    _validate_config_fingerprints(aligned_model_dir_list)
    final_dict = OrderedDict()
    ordered_stage_dicts = []
    for model_dir in aligned_model_dir_list:
        checkpoint_path = get_model_path_from_dir(model_dir)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        ordered_stage_dicts.append(state_dict)
        final_dict = merge_dict(final_dict, state_dict)
    final_dict = apply_doma_merge_ownership(final_dict, ordered_stage_dicts)
    os.makedirs(output_model_dir, exist_ok=True)
    output_path = os.path.join(output_model_dir, "net_epoch1.pth")
    torch.save(final_dict, output_path)
    print("DOMA merged checkpoint saved to %s" % output_path)
    return output_path


def _copy_owned_prefix(destination, source, prefix, owner):
    keys = {key for key in source if key.startswith(prefix)}
    expected = {prefix + suffix for suffix in ADAPTER_SUFFIXES}
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(
            "%s adapter contract differs for %s; missing=%s extra=%s"
            % (owner, prefix, missing, extra)
        )
    for key in sorted(keys):
        destination[key] = source[key]


def _normalize_doma_merge_config(doma_config):
    """Return the method/training subset relevant to checkpoint merging."""
    if not isinstance(doma_config, dict):
        raise TypeError("DOMA merge config must be a mapping")

    normalized = copy.deepcopy(doma_config)
    normalized.pop("mode", None)
    normalized.pop("active_modality", None)
    normalized.pop("delta_iou_diagnostics", None)

    qar_config = normalized.get("quality_aware_refinement")
    if isinstance(qar_config, dict):
        # Deployment policy does not change checkpoint structure or training.
        qar_config.pop("inference_gate", None)
        training_loss = qar_config.get("training_loss")
        if isinstance(training_loss, dict) and "apply_to" in training_loss:
            # apply_to has set semantics; preserve the canonical mode order so
            # equivalent YAML lists have the same deterministic fingerprint.
            training_loss["apply_to"] = sorted(
                training_loss["apply_to"],
                key=VALID_TRAINING_MODES.index,
            )
    return normalized


def doma_method_fingerprint(doma_config):
    """Serialize a normalized DOMA merge contract deterministically."""
    return json.dumps(
        _normalize_doma_merge_config(doma_config),
        sort_keys=True,
        separators=(",", ":"),
    )


# Backward-compatible internal spelling for callers from the earlier preview.
_doma_merge_fingerprint = doma_method_fingerprint


def _validate_config_fingerprints(model_dirs):
    fingerprints = []
    for model_dir in model_dirs:
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError("DOMA merge requires %s" % config_path)
        with open(config_path, "r") as stream:
            hypes = yaml.safe_load(stream)
        fingerprints.append(
            doma_method_fingerprint(hypes["model"]["args"]["doma"])
        )
    if len(set(fingerprints)) != 1:
        raise RuntimeError(
            "DOMA method configs differ across m2/m3/m4/m1 checkpoints"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "merge_final":
        raise SystemExit(
            "usage: python -m opencood.tools.doma_tools merge_final "
            "<m2_dir> <m3_dir> <m4_dir> <m1_dir> <output_dir>"
        )
    merge_and_save_final(sys.argv[2:-1], sys.argv[-1])
