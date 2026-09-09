"""Optional validation-split detection AP checkpoint selection helpers."""

import glob
import math
import os
import random
import re
from contextlib import contextmanager

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from opencood.tools import seed_utils, train_utils


SUPPORTED_METRICS = {
    "ap30": 0.3,
    "ap50": 0.5,
    "ap70": 0.7,
}
SUPPORTED_FUSION_METHODS = {
    "early", "intermediate", "late", "no", "no_w_uncertainty", "single",
}
HISTORY_FILENAME = "detection_validation_history.yaml"
SELECTION_METHOD = "validation_detection_ap"
_BESTDET_PATTERN = re.compile(r"net_epoch_bestdet_at([0-9]+)\.pth")


def _disabled_checkpoint_selection_config():
    return {
        "enabled": False,
        "metric": "ap70",
        "eval_freq": 2,
        "save_bestdet": True,
        "keep_only_best": True,
    }


def get_checkpoint_selection_config(hypes):
    """Normalize the optional config; disabled means completely inert."""
    raw_config = hypes.get("checkpoint_selection")
    if raw_config is None:
        return _disabled_checkpoint_selection_config()
    if not isinstance(raw_config, dict):
        raise TypeError("checkpoint_selection must be a mapping")

    enabled = raw_config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("checkpoint_selection.enabled must be bool")
    if not enabled:
        return _disabled_checkpoint_selection_config()

    config = {
        "enabled": True,
        "metric": raw_config.get("metric", "ap70"),
        "eval_freq": raw_config.get("eval_freq", 2),
        "save_bestdet": raw_config.get("save_bestdet", True),
        "keep_only_best": raw_config.get("keep_only_best", True),
    }
    for key in ("save_bestdet", "keep_only_best"):
        if type(config[key]) is not bool:
            raise TypeError("checkpoint_selection.{} must be bool".format(key))
    if config["metric"] not in SUPPORTED_METRICS:
        raise ValueError(
            "Unsupported checkpoint selection metric: {}. Expected one of: {}"
            .format(config["metric"], ", ".join(SUPPORTED_METRICS)))
    if type(config["eval_freq"]) is not int or config["eval_freq"] <= 0:
        raise ValueError(
            "checkpoint_selection.eval_freq must be a positive integer")
    return config


def validate_fusion_method(fusion_method):
    if fusion_method not in SUPPORTED_FUSION_METHODS:
        raise ValueError(
            "Unsupported detection validation fusion method: {}".format(
                fusion_method))


def build_detection_validation_loader(dataset, seed, num_workers=4):
    """Build a test-collate loader over the existing validation dataset."""
    if len(dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty; cannot evaluate detection AP")
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        **seed_utils.dataloader_seed_kwargs(seed)
    )


def should_evaluate_detection(epoch, config):
    """Use completed, one-based epochs for evaluation frequency semantics."""
    return (config["enabled"] and
            (epoch + 1) % config["eval_freq"] == 0)


@contextmanager
def preserve_global_rng_state():
    """Make optional validation invisible to the training RNG trajectory."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _run_inference(batch_data, model, dataset, fusion_method):
    # Keep inference-only geometry dependencies off the default training path.
    from opencood.tools import inference_utils

    if fusion_method == "late":
        return inference_utils.inference_late_fusion(batch_data, model, dataset)
    if fusion_method in ("early", "intermediate"):
        return inference_utils.inference_intermediate_fusion(
            batch_data, model, dataset)
    if fusion_method == "no":
        return inference_utils.inference_no_fusion(batch_data, model, dataset)
    if fusion_method == "no_w_uncertainty":
        return inference_utils.inference_no_fusion_w_uncertainty(
            batch_data, model, dataset)
    if fusion_method == "single":
        return inference_utils.inference_no_fusion(
            batch_data, model, dataset, single_gt=True)
    validate_fusion_method(fusion_method)
    raise AssertionError("unreachable")


def evaluate_detection_ap(model, data_loader, dataset, device,
                          fusion_method="intermediate"):
    """Evaluate validation AP without advancing any global experiment RNG."""
    from opencood.utils import eval_utils

    validate_fusion_method(fusion_method)
    result_stat = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in SUPPORTED_METRICS.values()
    }
    module_training_states = [
        (module, module.training) for module in model.modules()
    ]
    processed_batches = 0

    with preserve_global_rng_state():
        model.eval()
        try:
            with torch.no_grad():
                for batch_data in data_loader:
                    if batch_data is None:
                        continue
                    processed_batches += 1
                    batch_data = train_utils.to_device(batch_data, device)
                    infer_result = _run_inference(
                        batch_data, model, dataset, fusion_method)
                    for threshold in SUPPORTED_METRICS.values():
                        eval_utils.caluclate_tp_fp(
                            infer_result["pred_box_tensor"],
                            infer_result["pred_score"],
                            infer_result["gt_box_tensor"],
                            result_stat,
                            threshold,
                        )
        finally:
            # Restore exact nested train/eval flags without touching requires_grad.
            for module, training in module_training_states:
                module.training = training

        if processed_batches == 0:
            raise RuntimeError(
                "Validation detection loader produced no usable batches")
        if result_stat[0.3]["gt"] <= 0:
            raise RuntimeError(
                "Validation detection evaluation found no ground-truth boxes")

        metrics = {}
        for metric, threshold in SUPPORTED_METRICS.items():
            value, _, _ = eval_utils.calculate_ap(result_stat, threshold)
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(
                    "Validation detection evaluation returned non-finite {}: {}"
                    .format(metric, value))
            metrics[metric] = value
    return metrics


def _bestdet_checkpoints(save_dir):
    checkpoints = {}
    for checkpoint_path in glob.glob(
            os.path.join(save_dir, "net_epoch_bestdet_at*.pth")):
        match = _BESTDET_PATTERN.fullmatch(os.path.basename(checkpoint_path))
        if match is not None:
            checkpoints[int(match.group(1))] = checkpoint_path
    return checkpoints


def _atomic_yaml_dump(data, target_path):
    tmp_path = target_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, default_flow_style=False,
                           sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_bestdet_checkpoint(model, save_dir, epoch):
    """Atomically publish a bestdet checkpoint without deleting older ones."""
    checkpoint_path = os.path.join(
        save_dir, "net_epoch_bestdet_at{}.pth".format(epoch))
    tmp_path = checkpoint_path + ".tmp"
    try:
        with open(tmp_path, "wb") as stream:
            torch.save(model.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return checkpoint_path


def _remove_stale_bestdet_checkpoints(save_dir, selected_epoch):
    for epoch, checkpoint_path in _bestdet_checkpoints(save_dir).items():
        if epoch != selected_epoch:
            os.remove(checkpoint_path)


def _history_error(message):
    return RuntimeError("Invalid detection validation history: " + message)


def _validate_history_entries(entries, metric):
    if not isinstance(entries, list):
        raise _history_error("history must be a list")
    best_value = -float("inf")
    best_epoch = None
    improvement_epochs = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise _history_error("history[{}] must be a mapping".format(index))
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise _history_error(
                "history[{}].run_id must be a non-empty string".format(index))
        epoch = item.get("epoch")
        if type(epoch) is not int or epoch <= 0:
            raise _history_error(
                "history[{}].epoch must be a positive integer".format(index))
        for metric_name in SUPPORTED_METRICS:
            value = item.get(metric_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _history_error(
                    "history[{}].{} must be numeric".format(index, metric_name))
            if not math.isfinite(float(value)):
                raise _history_error(
                    "history[{}].{} must be finite".format(index, metric_name))
        current_value = float(item[metric])
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch
            improvement_epochs.add(epoch)
    return best_value, best_epoch, improvement_epochs


def restore_detection_selection_state(save_dir, config):
    """Strictly restore the authoritative AP selection state."""
    if not config["enabled"]:
        return -float("inf"), None, [], None

    history_path = os.path.join(save_dir, HISTORY_FILENAME)
    checkpoints = _bestdet_checkpoints(save_dir)
    if not os.path.exists(history_path):
        if checkpoints:
            raise RuntimeError(
                "Existing bestdet checkpoint found but detection validation "
                "history is missing. Cannot safely recover the previous best "
                "metric.")
        return -float("inf"), None, [], None

    try:
        with open(history_path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise _history_error("cannot be read: {}".format(exc)) from exc
    if not isinstance(data, dict):
        raise _history_error("top level must be a mapping")
    if data.get("enabled") is not True:
        raise _history_error("enabled must be true")
    if data.get("selection_method") != SELECTION_METHOD:
        raise _history_error(
            "selection_method must be {}".format(SELECTION_METHOD))
    recorded_metric = data.get("metric")
    if recorded_metric != config["metric"]:
        raise RuntimeError(
            "Detection validation history metric does not match current YAML: "
            "{} != {}. Restore the original metric or use a new experiment "
            "directory.".format(recorded_metric, config["metric"]))

    history = data.get("history")
    calculated_best, calculated_epoch, improvement_epochs = \
        _validate_history_entries(history, config["metric"])
    best_value = data.get("best_value")
    best_epoch = data.get("best_epoch")
    selected_epoch = data.get("selected_epoch")
    selected_checkpoint = data.get("selected_checkpoint")

    if not history:
        if best_value is not None or best_epoch is not None:
            raise _history_error("empty history must not declare a best result")
        if selected_epoch is not None or selected_checkpoint is not None:
            raise _history_error("empty history must not declare a selection")
        if checkpoints:
            raise _history_error("checkpoints exist for an empty history")
        return -float("inf"), None, [], None

    if isinstance(best_value, bool) or not isinstance(best_value, (int, float)):
        raise _history_error("best_value must be numeric")
    best_value = float(best_value)
    if not math.isfinite(best_value) or best_value != calculated_best:
        raise _history_error("best_value does not match the history")
    if best_epoch != calculated_epoch or selected_epoch != best_epoch:
        raise _history_error(
            "best_epoch/selected_epoch does not match the history")

    expected_name = "net_epoch_bestdet_at{}.pth".format(best_epoch)
    if selected_checkpoint not in (None, expected_name):
        raise _history_error(
            "selected_checkpoint does not match selected_epoch")
    unexpected_epochs = set(checkpoints) - improvement_epochs
    if unexpected_epochs:
        raise _history_error(
            "checkpoint epoch(s) are not committed in history: {}".format(
                ", ".join(str(epoch) for epoch in sorted(unexpected_epochs))))

    if config["save_bestdet"]:
        if selected_checkpoint != expected_name or best_epoch not in checkpoints:
            raise RuntimeError(
                "Detection validation history selects {}, but the matching "
                "bestdet checkpoint is missing.".format(expected_name))
        if config["keep_only_best"]:
            _remove_stale_bestdet_checkpoints(save_dir, best_epoch)
        elif set(checkpoints) != improvement_epochs:
            raise _history_error(
                "keep_only_best=false requires every recorded bestdet "
                "checkpoint to be present")
    elif selected_checkpoint is not None:
        raise _history_error(
            "save_bestdet=false history must not select a checkpoint")

    return best_value, best_epoch, history, selected_checkpoint


def save_detection_history(save_dir, config, best_epoch, best_value, history,
                           selected_checkpoint):
    """Atomically persist AP history and authoritative selection metadata."""
    history_path = os.path.join(save_dir, HISTORY_FILENAME)
    data = {
        "enabled": True,
        "selection_method": SELECTION_METHOD,
        "metric": config["metric"],
        "save_bestdet": config["save_bestdet"],
        "keep_only_best": config["keep_only_best"],
        "best_epoch": best_epoch,
        "selected_epoch": best_epoch,
        "selected_checkpoint": selected_checkpoint,
        "best_value": None if best_epoch is None else float(best_value),
        "history": history,
    }
    _atomic_yaml_dump(data, history_path)
    return history_path


def commit_detection_evaluation(model, save_dir, config, run_id, epoch,
                                metrics, best_value, best_epoch, history):
    """Commit checkpoint then history, and only then remove stale checkpoints."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    normalized_metrics = {}
    for metric in SUPPORTED_METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(metric))
        normalized_metrics[metric] = value

    current_value = normalized_metrics[config["metric"]]
    is_new_best = current_value > best_value
    next_best_value = current_value if is_new_best else best_value
    next_best_epoch = epoch if is_new_best else best_epoch
    if config["save_bestdet"] and next_best_epoch is not None:
        selected_checkpoint = \
            "net_epoch_bestdet_at{}.pth".format(next_best_epoch)
    else:
        selected_checkpoint = None

    saved_checkpoint_path = None
    if is_new_best and config["save_bestdet"]:
        saved_checkpoint_path = save_bestdet_checkpoint(model, save_dir, epoch)

    next_history = list(history)
    history_item = {"run_id": run_id, "epoch": epoch}
    history_item.update(normalized_metrics)
    next_history.append(history_item)
    save_detection_history(
        save_dir, config, next_best_epoch, next_best_value, next_history,
        selected_checkpoint)

    if (is_new_best and config["save_bestdet"] and
            config["keep_only_best"]):
        _remove_stale_bestdet_checkpoints(save_dir, next_best_epoch)
    return (next_best_value, next_best_epoch, next_history,
            saved_checkpoint_path, is_new_best)


def resolve_selected_checkpoint(save_dir, config):
    """Resolve bestdet only for an explicitly enabled experiment."""
    if not config["enabled"]:
        return None
    _, best_epoch, _, selected_checkpoint = \
        restore_detection_selection_state(save_dir, config)
    if best_epoch is None:
        raise RuntimeError(
            "checkpoint_selection is enabled, but no validation detection "
            "result has been committed")
    if selected_checkpoint is None:
        raise RuntimeError(
            "checkpoint_selection is enabled with save_bestdet=false; no "
            "bestdet checkpoint is available for inference or merging")
    return os.path.join(save_dir, selected_checkpoint)


def resolve_selected_checkpoint_from_model_dir(model_dir):
    """Read saved config and resolve bestdet, or return None for old/default runs."""
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as stream:
        hypes = yaml.safe_load(stream) or {}
    if not isinstance(hypes, dict):
        raise RuntimeError(
            "Saved experiment config must be a mapping: {}".format(config_path))
    config = get_checkpoint_selection_config(hypes)
    return resolve_selected_checkpoint(model_dir, config)
