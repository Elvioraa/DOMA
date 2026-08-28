"""Object residual and optional quality losses for DOMA HEAL."""

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
    return object_loss, stats


def _residual_smooth_l1_per_object(predictions, targets):
    # Dimension mean first; the caller then averages all valid objects globally.
    return F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=-1)
