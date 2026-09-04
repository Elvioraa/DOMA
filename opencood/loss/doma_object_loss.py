"""Object residual and opt-in auxiliary losses for DOMA HEAL."""

import torch
import torch.nn.functional as F


def compute_doma_object_loss(payload):
    """Compute dimension-averaged SmoothL1 DOMA training loss and stats.

    Stage1 supervises every valid agent residual and the uniform consensus.
    Stage2 has one active modality per sample and therefore uses only the
    individual adaptation term; duplicating it as a one-agent consensus would
    merely double the configured learning signal.
    """
    if not isinstance(payload, dict) or not payload.get("enabled", False):
        raise ValueError("payload must be an enabled DOMA object result")
    scenes = payload.get("scenes")
    if not isinstance(scenes, tuple):
        raise TypeError("DOMA object scenes must be a tuple")
    config = payload["loss_config"]
    mode = payload["mode"]

    individual_terms = []
    consensus_terms = []
    quality_terms = []
    quality_predictions = []
    quality_targets = []
    reference = None
    for scene in scenes:
        predictions = scene["individual_residuals"]
        targets = scene["individual_targets"].to(dtype=predictions.dtype)
        reference = predictions if reference is None else reference
        if predictions.numel():
            individual_terms.append(_residual_smooth_l1_per_object(predictions, targets))
        if (
            mode == "stage1_anchor"
            and bool(payload.get("consensus_enabled", False))
            and bool(scene["any_valid"].any())
        ):
            mask = scene["any_valid"]
            consensus_terms.append(
                _residual_smooth_l1_per_object(
                    scene["fused_residuals"][mask],
                    scene["target_residuals"][mask].to(
                        dtype=scene["fused_residuals"].dtype
                    ),
                )
            )
        if "individual_quality" in scene:
            predictions_q = scene["individual_quality"]
            targets_q = scene["quality_targets"].to(dtype=predictions_q.dtype)
            if predictions_q.shape != targets_q.shape:
                raise ValueError("quality prediction and target shapes must match")
            if predictions_q.numel():
                quality_terms.append(
                    F.smooth_l1_loss(predictions_q, targets_q, reduction="none")
                )
                quality_predictions.append(predictions_q.detach())
                quality_targets.append(targets_q.detach())

    if reference is None:
        raise ValueError("DOMA object payload contains no scenes")
    zero = reference.sum() * 0.0
    individual_loss = torch.cat(individual_terms).mean() if individual_terms else zero
    consensus_loss = torch.cat(consensus_terms).mean() if consensus_terms else zero
    quality_loss = torch.cat(quality_terms).mean() if quality_terms else zero
    unweighted = (
        float(config["individual_loss_weight"]) * individual_loss
        + float(config["consensus_loss_weight"]) * consensus_loss
    )
    quality_enabled = bool(payload.get("quality_enabled", False))
    if quality_enabled:
        unweighted = unweighted + float(config["quality_loss_weight"]) * quality_loss
    object_loss = float(config["object_loss_weight"]) * unweighted
    stats = {
        "doma_enabled": True,
        "doma_object_loss": float(object_loss.detach().item()),
        "doma_individual_loss": float(individual_loss.detach().item()),
        "doma_consensus_loss": float(consensus_loss.detach().item()),
        "doma_valid_object_ratio": payload["stats"]["valid_object_ratio"],
        "doma_mean_roi_coverage": payload["stats"]["mean_roi_coverage"],
        "doma_object_roi_count": payload["stats"]["object_roi_count"],
        "doma_valid_agent_object_pairs": payload["stats"][
            "valid_agent_object_pairs"
        ],
    }
    if quality_enabled:
        stats.update(
            {
                "doma_quality_loss": float(quality_loss.detach().item()),
                "doma_mean_pred_quality": float(
                    torch.cat(quality_predictions).mean().item()
                ) if quality_predictions else 0.0,
                "doma_mean_quality_target": float(
                    torch.cat(quality_targets).mean().item()
                ) if quality_targets else 0.0,
            }
        )

    protocol_enabled = _optional_bool_flag(
        payload, "object_protocol_alignment_enabled"
    )
    qar_loss_enabled = _optional_bool_flag(
        payload, "quality_aware_refinement_training_loss_enabled"
    )
    diagnostics_enabled = _optional_bool_flag(
        payload, "delta_iou_diagnostics_enabled"
    )
    if not (protocol_enabled or qar_loss_enabled or diagnostics_enabled):
        return object_loss, stats

    auxiliary_loss = zero
    if protocol_enabled:
        protocol_config = _required_mapping(
            payload,
            "object_protocol_alignment_config",
        )
        protocol_weight = _nonnegative_real(
            protocol_config.get("weight"),
            "object_protocol_alignment_config.weight",
        )
        protocol_loss_type = protocol_config.get("loss_type")
        if protocol_loss_type not in ("cosine", "smooth_l1", "mse"):
            raise ValueError(
                "object_protocol_alignment_config.loss_type must be one of "
                "('cosine', 'smooth_l1', 'mse')"
            )
        protocol_loss, protocol_pair_count = _compute_protocol_loss(
            scenes,
            protocol_loss_type,
            zero,
        )
        weighted_protocol_loss = protocol_weight * protocol_loss
        auxiliary_loss = auxiliary_loss + weighted_protocol_loss
        stats.update(
            {
                "doma_protocol_loss": float(protocol_loss.detach().item()),
                "doma_protocol_weighted_loss": float(
                    weighted_protocol_loss.detach().item()
                ),
                "doma_protocol_pair_count": protocol_pair_count,
            }
        )

    if qar_loss_enabled:
        qar_config = _required_mapping(
            payload,
            "quality_aware_refinement_training_loss_config",
        )
        qar_weight = _nonnegative_real(
            qar_config.get("weight"),
            "quality_aware_refinement_training_loss_config.weight",
        )
        qar_loss, qar_stats = _compute_qar_loss(scenes, zero)
        weighted_qar_loss = qar_weight * qar_loss
        auxiliary_loss = auxiliary_loss + weighted_qar_loss
        stats.update(
            {
                "doma_qar_loss": float(qar_loss.detach().item()),
                "doma_qar_weighted_loss": float(
                    weighted_qar_loss.detach().item()
                ),
                "doma_qar_pred_mean": qar_stats["pred_mean"],
                "doma_qar_target_mean": qar_stats["target_mean"],
            }
        )

    if diagnostics_enabled:
        stats.update(_compute_delta_iou_diagnostics(scenes))

    if not (protocol_enabled or qar_loss_enabled):
        return object_loss, stats

    total_loss = object_loss + auxiliary_loss
    stats["doma_auxiliary_loss"] = float(auxiliary_loss.detach().item())
    stats["doma_total_loss"] = float(total_loss.detach().item())
    return total_loss, stats


def _residual_smooth_l1_per_object(predictions, targets):
    # Dimension mean first; the caller then averages all valid objects globally.
    return F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=-1)


def _compute_protocol_loss(scenes, loss_type, zero):
    """Average feature-reduced losses over every valid pair and branch."""
    pair_terms = []
    pair_count = 0
    for scene in scenes:
        if "protocol_pairs" not in scene:
            raise KeyError(
                "OPA-enabled DOMA scene is missing protocol_pairs"
            )
        protocol_pairs = scene["protocol_pairs"]
        if not isinstance(protocol_pairs, dict):
            raise TypeError("scene.protocol_pairs must be a mapping")
        for branch, pair in protocol_pairs.items():
            if not isinstance(pair, dict):
                raise TypeError(
                    "scene.protocol_pairs[%r] must be a mapping" % branch
                )
            prediction = pair.get("prediction")
            target = pair.get("target")
            _validate_protocol_pair(prediction, target, branch)
            if prediction.shape[0] == 0:
                continue
            target = target.to(dtype=prediction.dtype)
            if loss_type == "cosine":
                terms = 1.0 - F.cosine_similarity(
                    prediction,
                    target,
                    dim=-1,
                )
            elif loss_type == "smooth_l1":
                terms = F.smooth_l1_loss(
                    prediction,
                    target,
                    reduction="none",
                ).mean(dim=-1)
            else:
                terms = F.mse_loss(
                    prediction,
                    target,
                    reduction="none",
                ).mean(dim=-1)
            pair_terms.append(terms)
            pair_count += int(terms.numel())
    loss = torch.cat(pair_terms).mean() if pair_terms else zero
    return loss, pair_count


def _compute_qar_loss(scenes, zero):
    """Compute proposal-wise QAR Smooth-L1 and prediction diagnostics."""
    loss_terms = []
    predictions = []
    targets = []
    for scene in scenes:
        for key in ("qar_predictions", "qar_targets"):
            if key not in scene:
                raise KeyError("QAR-enabled DOMA scene is missing %s" % key)
        prediction = scene["qar_predictions"]
        target = scene["qar_targets"]
        _validate_qar_tensors(prediction, target)
        if prediction.numel() == 0:
            continue
        target = target.to(dtype=prediction.dtype).detach()
        loss_terms.append(
            F.smooth_l1_loss(prediction, target, reduction="none")
        )
        predictions.append(prediction.detach())
        targets.append(target.detach())

    loss = torch.cat(loss_terms).mean() if loss_terms else zero
    if not predictions:
        return loss, {
            "pred_mean": 0.0,
            "target_mean": 0.0,
        }

    prediction_values = torch.cat(predictions)
    target_values = torch.cat(targets)
    return loss, {
        "pred_mean": float(prediction_values.mean().item()),
        "target_mean": float(target_values.mean().item()),
    }


def _compute_delta_iou_diagnostics(scenes):
    """Summarize a detached batch distribution without changing its loss."""
    values = []
    for scene in scenes:
        if "delta_iou" not in scene:
            raise KeyError(
                "Delta-IoU diagnostics-enabled scene is missing delta_iou"
            )
        value = scene["delta_iou"]
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError("scene.delta_iou must be a one-dimensional tensor")
        values.append(value.detach().to(dtype=torch.float32))

    combined = torch.cat(values) if values else torch.empty((0,))
    count = int(combined.numel())
    finite = combined[torch.isfinite(combined)]
    finite_count = int(finite.numel())
    stats = {
        "doma_delta_iou_count": count,
        "doma_delta_iou_finite_count": finite_count,
        "doma_delta_iou_nonfinite_count": count - finite_count,
        "doma_delta_iou_mean": 0.0,
        "doma_delta_iou_std": 0.0,
        "doma_delta_iou_min": 0.0,
        "doma_delta_iou_p10": 0.0,
        "doma_delta_iou_p25": 0.0,
        "doma_delta_iou_p50": 0.0,
        "doma_delta_iou_p75": 0.0,
        "doma_delta_iou_p90": 0.0,
        "doma_delta_iou_max": 0.0,
        "doma_delta_iou_improve_ratio": 0.0,
        "doma_delta_iou_zero_ratio": 0.0,
        "doma_delta_iou_worsen_ratio": 0.0,
    }
    if finite_count == 0:
        return stats

    quantiles = torch.quantile(
        finite,
        finite.new_tensor((0.10, 0.25, 0.50, 0.75, 0.90)),
    )
    stats.update(
        {
            "doma_delta_iou_mean": float(finite.mean().item()),
            "doma_delta_iou_std": float(finite.std(unbiased=False).item()),
            "doma_delta_iou_min": float(finite.min().item()),
            "doma_delta_iou_p10": float(quantiles[0].item()),
            "doma_delta_iou_p25": float(quantiles[1].item()),
            "doma_delta_iou_p50": float(quantiles[2].item()),
            "doma_delta_iou_p75": float(quantiles[3].item()),
            "doma_delta_iou_p90": float(quantiles[4].item()),
            "doma_delta_iou_max": float(finite.max().item()),
            "doma_delta_iou_improve_ratio": float(
                (finite > 0).to(dtype=torch.float32).mean().item()
            ),
            "doma_delta_iou_zero_ratio": float(
                (finite == 0).to(dtype=torch.float32).mean().item()
            ),
            "doma_delta_iou_worsen_ratio": float(
                (finite < 0).to(dtype=torch.float32).mean().item()
            ),
        }
    )
    return stats


def _validate_protocol_pair(prediction, target, branch):
    if not torch.is_tensor(prediction) or prediction.ndim != 2:
        raise ValueError(
            "protocol prediction for %r must be a tensor [N,D]" % branch
        )
    if not torch.is_tensor(target) or target.ndim != 2:
        raise ValueError(
            "protocol target for %r must be a tensor [N,D]" % branch
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "protocol prediction and target for %r must have equal shape"
            % branch
        )
    if prediction.device != target.device:
        raise ValueError(
            "protocol prediction and target for %r must share a device"
            % branch
        )
    if target.requires_grad:
        raise ValueError(
            "protocol target for %r must be detached" % branch
        )


def _validate_qar_tensors(prediction, target):
    for name, value in (
        ("qar_predictions", prediction),
        ("qar_targets", target),
    ):
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError("scene.%s must be a one-dimensional tensor" % name)
    if prediction.shape != target.shape:
        raise ValueError("scene qar_predictions and qar_targets must match")
    if prediction.device != target.device:
        raise ValueError("scene QAR tensors must share a device")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("scene QAR loss tensors must be finite")


def _optional_bool_flag(payload, key):
    value = payload.get(key, False)
    if type(value) is not bool:
        raise TypeError("payload.%s must be bool" % key)
    return value


def _required_mapping(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError("payload.%s must be a mapping" % key)
    return value


def _nonnegative_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    value = float(value)
    if value < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return value
