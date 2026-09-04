"""Strict, explicit configuration validation for DOMA.

The ``version`` field is an experiment label and a consistency guard only.
Runtime behavior is controlled by the individual YAML switches.  Canonical
profiles are checked exactly unless ``ablation: true`` is set.
"""

import math
import re


VALID_VERSIONS = ("v1", "v2", "v3")
VALID_MODES = ("stage1_anchor", "stage2_adapt", "inference")
VALID_TRAINING_MODES = ("stage1_anchor", "stage2_adapt")
VALID_CONSENSUS_MODES = ("uniform_geometry_mean", "quality_weighted")
VALID_MULTIGRANULAR_FUSIONS = ("concat_projection",)
VALID_YAW_MODES = ("sin_cos", "sin_cos_centered")
VALID_PROTOCOL_LOSS_TYPES = ("cosine", "smooth_l1", "mse")
TOP_LEVEL_KEYS = (
    "enabled",
    "version",
    "ablation",
    "mode",
    "active_modality",
    "object_space",
    "object_adapter",
    "object_encoder",
    "geometry",
    "refiner",
    "multi_granularity",
    "quality",
    "object_protocol_alignment",
    "quality_aware_refinement",
    "delta_iou_diagnostics",
    "consensus",
    "training_proposals",
    "loss",
)


def validate_doma_config(config):
    """Validate and return the supplied DOMA mapping without mutating it."""
    if not isinstance(config, dict):
        raise TypeError("model.args.doma must be a mapping")
    _reject_unknown(config, TOP_LEVEL_KEYS, "doma")
    enabled = _optional_bool(config, "enabled", False, "doma.enabled")
    version = config.get("version")
    if enabled and version not in VALID_VERSIONS:
        raise ValueError("doma.version must be one of %s" % (VALID_VERSIONS,))
    if not enabled:
        if set(config) != {"enabled"}:
            raise ValueError("disabled DOMA config must contain only enabled=false")
        return config

    mode = config.get("mode")
    if mode not in VALID_MODES:
        raise ValueError("doma.mode must be one of %s" % (VALID_MODES,))
    active_modality = config.get("active_modality")
    if mode == "stage2_adapt":
        if (
            not isinstance(active_modality, str)
            or re.fullmatch(r"m[1-9][0-9]*", active_modality) is None
        ):
            raise ValueError("doma.active_modality is required for stage2_adapt")
    elif active_modality is not None:
        raise ValueError("doma.active_modality is only valid for stage2_adapt")
    ablation = _optional_bool(config, "ablation", False, "doma.ablation")

    object_space = _required_mapping(config, "object_space", "doma.object_space")
    _reject_unknown(object_space, ("enabled", "roi"), "doma.object_space")
    object_enabled = _required_bool(
        object_space, "enabled", "doma.object_space.enabled"
    )
    roi = _required_mapping(object_space, "roi", "doma.object_space.roi")
    _reject_unknown(
        roi,
        (
            "output_size",
            "max_train_proposals",
            "max_infer_proposals",
            "chunk_size",
            "min_coverage",
        ),
        "doma.object_space.roi",
    )
    for key in ("output_size", "max_train_proposals", "max_infer_proposals", "chunk_size"):
        _positive_int(roi.get(key), "doma.object_space.roi.%s" % key)
    _unit_interval(roi.get("min_coverage"), "doma.object_space.roi.min_coverage")

    adapter = _required_mapping(config, "object_adapter", "doma.object_adapter")
    _reject_unknown(
        adapter, ("enabled", "type", "zero_init"), "doma.object_adapter"
    )
    adapter_enabled = _required_bool(
        adapter, "enabled", "doma.object_adapter.enabled"
    )
    if adapter.get("type") != "residual_1x1":
        raise ValueError("doma.object_adapter.type must be residual_1x1")
    if adapter.get("zero_init") is not True:
        raise ValueError("doma.object_adapter.zero_init must be true")

    encoder = _required_mapping(config, "object_encoder", "doma.object_encoder")
    _reject_unknown(
        encoder,
        ("enabled", "embedding_dim", "hidden_channels", "pooled_size"),
        "doma.object_encoder",
    )
    encoder_enabled = _required_bool(
        encoder, "enabled", "doma.object_encoder.enabled"
    )
    for key in ("embedding_dim", "hidden_channels", "pooled_size"):
        _positive_int(encoder.get(key), "doma.object_encoder.%s" % key)

    geometry = _required_mapping(config, "geometry", "doma.geometry")
    _reject_unknown(geometry, ("enabled", "hidden_dim"), "doma.geometry")
    geometry_enabled = _required_bool(
        geometry, "enabled", "doma.geometry.enabled"
    )
    _positive_int(geometry.get("hidden_dim"), "doma.geometry.hidden_dim")

    refiner = _required_mapping(config, "refiner", "doma.refiner")
    _reject_unknown(
        refiner,
        ("enabled", "hidden_dim", "yaw_mode", "zero_init_output"),
        "doma.refiner",
    )
    refiner_enabled = _required_bool(refiner, "enabled", "doma.refiner.enabled")
    _positive_int(refiner.get("hidden_dim"), "doma.refiner.hidden_dim")
    if refiner.get("yaw_mode") not in VALID_YAW_MODES:
        raise ValueError("doma.refiner.yaw_mode must be one of %s" % (VALID_YAW_MODES,))
    if refiner.get("zero_init_output") is not True:
        raise ValueError("doma.refiner.zero_init_output must be true")

    multi = _required_mapping(
        config, "multi_granularity", "doma.multi_granularity"
    )
    multi_enabled = _required_bool(
        multi, "enabled", "doma.multi_granularity.enabled"
    )
    detail_enabled = context_enabled = fusion_enabled = False
    if multi_enabled:
        _reject_unknown(
            multi,
            ("enabled", "detail", "context", "fusion"),
            "doma.multi_granularity",
        )
        detail = _required_mapping(
            multi, "detail", "doma.multi_granularity.detail"
        )
        _reject_unknown(detail, ("enabled", "roi_size"), "doma.multi_granularity.detail")
        context = _required_mapping(
            multi, "context", "doma.multi_granularity.context"
        )
        _reject_unknown(context, ("enabled", "roi_size"), "doma.multi_granularity.context")
        fusion = _required_mapping(
            multi, "fusion", "doma.multi_granularity.fusion"
        )
        _reject_unknown(fusion, ("enabled", "type"), "doma.multi_granularity.fusion")
        detail_enabled = _required_bool(
            detail, "enabled", "doma.multi_granularity.detail.enabled"
        )
        context_enabled = _required_bool(
            context, "enabled", "doma.multi_granularity.context.enabled"
        )
        fusion_enabled = _required_bool(
            fusion, "enabled", "doma.multi_granularity.fusion.enabled"
        )
        _positive_int(detail.get("roi_size"), "doma.multi_granularity.detail.roi_size")
        _positive_int(context.get("roi_size"), "doma.multi_granularity.context.roi_size")
        if fusion.get("type") not in VALID_MULTIGRANULAR_FUSIONS:
            raise ValueError(
                "doma.multi_granularity.fusion.type must be one of %s"
                % (VALID_MULTIGRANULAR_FUSIONS,)
            )
        if context_enabled != fusion_enabled:
            raise ValueError(
                "context and multi-granularity fusion must be enabled or disabled together"
            )
    else:
        _reject_unknown(multi, ("enabled",), "doma.multi_granularity")

    quality = _required_mapping(config, "quality", "doma.quality")
    quality_enabled = _required_bool(quality, "enabled", "doma.quality.enabled")
    if quality_enabled:
        _reject_unknown(
            quality,
            (
                "enabled",
                "target",
                "hidden_dim",
                "use_roi_coverage",
                "use_agent_distance",
                "detach_target",
                "detach_weight_for_consensus",
            ),
            "doma.quality",
        )
        if quality.get("target") != "refined_iou":
            raise ValueError("doma.quality.target must be refined_iou")
        _positive_int(quality.get("hidden_dim"), "doma.quality.hidden_dim")
        for key in (
            "use_roi_coverage",
            "use_agent_distance",
            "detach_target",
            "detach_weight_for_consensus",
        ):
            _required_bool(quality, key, "doma.quality.%s" % key)
        if quality.get("detach_target") is not True:
            raise ValueError("doma.quality.detach_target must be true")
    else:
        _reject_unknown(quality, ("enabled",), "doma.quality")

    protocol_detail_enabled = False
    protocol_context_enabled = False
    protocol, protocol_enabled = _optional_feature_mapping(
        config,
        "object_protocol_alignment",
        "doma.object_protocol_alignment",
    )
    if protocol_enabled:
        _reject_unknown(
            protocol,
            (
                "enabled",
                "method",
                "apply_to",
                "hook",
                "target_type",
                "loss_type",
                "weight",
                "branches",
            ),
            "doma.object_protocol_alignment",
        )
        if protocol.get("method") != "consistency_proxy":
            raise ValueError(
                "doma.object_protocol_alignment.method must be "
                "consistency_proxy; current OPA is not m1 distillation"
            )
        if protocol.get("apply_to") != "stage2_adapt":
            raise ValueError(
                "doma.object_protocol_alignment.apply_to must be stage2_adapt"
            )
        if protocol.get("hook") != "shared_encoder_outputs":
            raise ValueError(
                "doma.object_protocol_alignment.hook must be shared_encoder_outputs"
            )
        if protocol.get("target_type") != "gt_proposal_consistency":
            raise ValueError(
                "doma.object_protocol_alignment is a consistency proxy and "
                "requires target_type=gt_proposal_consistency"
            )
        if protocol.get("loss_type") not in VALID_PROTOCOL_LOSS_TYPES:
            raise ValueError(
                "doma.object_protocol_alignment.loss_type must be one of %s"
                % (VALID_PROTOCOL_LOSS_TYPES,)
            )
        _positive_real(
            protocol.get("weight"),
            "doma.object_protocol_alignment.weight",
        )
        branches = _required_mapping(
            protocol,
            "branches",
            "doma.object_protocol_alignment.branches",
        )
        _reject_unknown(
            branches,
            ("detail", "context"),
            "doma.object_protocol_alignment.branches",
        )
        protocol_detail_enabled = _optional_bool(
            branches,
            "detail",
            False,
            "doma.object_protocol_alignment.branches.detail",
        )
        protocol_context_enabled = _optional_bool(
            branches,
            "context",
            False,
            "doma.object_protocol_alignment.branches.context",
        )
        if not (protocol_detail_enabled or protocol_context_enabled):
            raise ValueError(
                "doma.object_protocol_alignment requires at least one enabled "
                "consistency-proxy branch"
            )

    qar_training_loss_enabled = False
    qar_training_apply_to = ()
    qar_detach_features = False
    qar_inference_gate_enabled = False
    refinement, refinement_enabled = _optional_feature_mapping(
        config,
        "quality_aware_refinement",
        "doma.quality_aware_refinement",
    )
    if refinement_enabled:
        _reject_unknown(
            refinement,
            (
                "enabled",
                "target_type",
                "hidden_dim",
                "detach_residual",
                "zero_init_output",
                "training_loss",
                "inference_gate",
            ),
            "doma.quality_aware_refinement",
        )
        if refinement.get("target_type") != "delta_iou":
            raise ValueError(
                "doma.quality_aware_refinement.target_type must be delta_iou"
            )

        training_loss = _required_mapping(
            refinement,
            "training_loss",
            "doma.quality_aware_refinement.training_loss",
        )
        qar_training_loss_enabled = _required_bool(
            training_loss,
            "enabled",
            "doma.quality_aware_refinement.training_loss.enabled",
        )
        if qar_training_loss_enabled:
            _reject_unknown(
                training_loss,
                (
                    "enabled",
                    "apply_to",
                    "loss_type",
                    "weight",
                    "detach_target",
                    "detach_features",
                ),
                "doma.quality_aware_refinement.training_loss",
            )
            qar_training_apply_to = _required_mode_list(
                training_loss,
                "apply_to",
                "doma.quality_aware_refinement.training_loss.apply_to",
                VALID_TRAINING_MODES,
            )
            if training_loss.get("loss_type") != "smooth_l1":
                raise ValueError(
                    "doma.quality_aware_refinement.training_loss.loss_type "
                    "must be smooth_l1"
                )
            _positive_real(
                training_loss.get("weight"),
                "doma.quality_aware_refinement.training_loss.weight",
            )
            if _required_bool(
                training_loss,
                "detach_target",
                "doma.quality_aware_refinement.training_loss.detach_target",
            ) is not True:
                raise ValueError(
                    "doma.quality_aware_refinement.training_loss.detach_target "
                    "must be true"
                )
            qar_detach_features = _required_bool(
                training_loss,
                "detach_features",
                "doma.quality_aware_refinement.training_loss.detach_features",
            )
            if (
                "stage2_adapt" in qar_training_apply_to
                and "stage1_anchor" not in qar_training_apply_to
            ):
                raise ValueError(
                    "doma.quality_aware_refinement.training_loss.apply_to "
                    "cannot be stage2-only because the QAR head is Stage1-owned"
                )
            if (
                "stage2_adapt" in qar_training_apply_to
                and qar_detach_features
            ):
                raise ValueError(
                    "doma.quality_aware_refinement.training_loss.detach_features "
                    "must be false when apply_to includes stage2_adapt"
                )
        else:
            _reject_unknown(
                training_loss,
                ("enabled",),
                "doma.quality_aware_refinement.training_loss",
            )

        inference_gate = _required_mapping(
            refinement,
            "inference_gate",
            "doma.quality_aware_refinement.inference_gate",
        )
        qar_inference_gate_enabled = _required_bool(
            inference_gate,
            "enabled",
            "doma.quality_aware_refinement.inference_gate.enabled",
        )
        if qar_inference_gate_enabled:
            _reject_unknown(
                inference_gate,
                ("enabled", "mode", "threshold"),
                "doma.quality_aware_refinement.inference_gate",
            )
            if inference_gate.get("mode") != "hard":
                raise ValueError(
                    "doma.quality_aware_refinement.inference_gate.mode must be hard"
                )
            _closed_interval(
                inference_gate.get("threshold"),
                -1.0,
                1.0,
                "doma.quality_aware_refinement.inference_gate.threshold",
            )
        else:
            _reject_unknown(
                inference_gate,
                ("enabled",),
                "doma.quality_aware_refinement.inference_gate",
            )

        if not (qar_training_loss_enabled or qar_inference_gate_enabled):
            raise ValueError(
                "doma.quality_aware_refinement requires training_loss or "
                "inference_gate to be enabled"
            )

        _positive_int(
            refinement.get("hidden_dim"),
            "doma.quality_aware_refinement.hidden_dim",
        )
        for key in ("detach_residual", "zero_init_output"):
            value = _required_bool(
                refinement,
                key,
                "doma.quality_aware_refinement.%s" % key,
            )
            if value is not True:
                raise ValueError(
                    "doma.quality_aware_refinement.%s must be true" % key
                )

    delta_iou_diagnostics_apply_to = ()
    delta_iou_diagnostics, delta_iou_diagnostics_enabled = (
        _optional_feature_mapping(
            config,
            "delta_iou_diagnostics",
            "doma.delta_iou_diagnostics",
        )
    )
    if delta_iou_diagnostics_enabled:
        _reject_unknown(
            delta_iou_diagnostics,
            ("enabled", "apply_to", "neutral_threshold"),
            "doma.delta_iou_diagnostics",
        )
        delta_iou_diagnostics_apply_to = _required_mode_list(
            delta_iou_diagnostics,
            "apply_to",
            "doma.delta_iou_diagnostics.apply_to",
            VALID_MODES,
        )
        _closed_interval(
            delta_iou_diagnostics.get("neutral_threshold"),
            0.0,
            1.0,
            "doma.delta_iou_diagnostics.neutral_threshold",
        )

    consensus = _required_mapping(config, "consensus", "doma.consensus")
    _reject_unknown(
        consensus,
        (
            "enabled",
            "mode",
            "fallback_to_original",
            "min_quality_sum",
            "low_quality_fallback",
        ),
        "doma.consensus",
    )
    consensus_enabled = _required_bool(
        consensus, "enabled", "doma.consensus.enabled"
    )
    consensus_mode = consensus.get("mode")
    if consensus_mode not in VALID_CONSENSUS_MODES:
        raise ValueError("doma.consensus.mode must be one of %s" % (VALID_CONSENSUS_MODES,))
    fallback_to_original = _required_bool(
        consensus, "fallback_to_original", "doma.consensus.fallback_to_original"
    )
    if fallback_to_original is not True:
        raise ValueError("DOMA V1-V3 requires fallback_to_original=true")
    if consensus_mode == "quality_weighted":
        if not quality_enabled:
            raise ValueError("quality_weighted consensus requires doma.quality.enabled=true")
        _positive_real(consensus.get("min_quality_sum"), "doma.consensus.min_quality_sum")
        if consensus.get("low_quality_fallback") != "uniform":
            raise ValueError("doma.consensus.low_quality_fallback must be uniform")
    elif "min_quality_sum" in consensus or "low_quality_fallback" in consensus:
        raise ValueError(
            "uniform consensus cannot define quality fallback parameters"
        )

    proposals = _required_mapping(
        config, "training_proposals", "doma.training_proposals"
    )
    _reject_unknown(
        proposals,
        (
            "source",
            "include_gt",
            "jitters_per_gt",
            "center_xy_std_rel",
            "center_z_std_rel",
            "log_size_std",
            "yaw_std_deg",
            "max_proposals",
        ),
        "doma.training_proposals",
    )
    if proposals.get("source") != "gt_jitter":
        raise ValueError("DOMA V1-V3 supports only training_proposals.source=gt_jitter")
    include_gt = _required_bool(
        proposals, "include_gt", "doma.training_proposals.include_gt"
    )
    jitters_per_gt = _nonnegative_int(
        proposals.get("jitters_per_gt"), "doma.training_proposals.jitters_per_gt"
    )
    jitter_std_keys = (
        "center_xy_std_rel",
        "center_z_std_rel",
        "log_size_std",
        "yaw_std_deg",
    )
    for key in jitter_std_keys:
        _nonnegative_real(proposals.get(key), "doma.training_proposals.%s" % key)
    _positive_int(proposals.get("max_proposals"), "doma.training_proposals.max_proposals")
    if proposals["max_proposals"] != roi["max_train_proposals"]:
        raise ValueError("training proposal and ROI maxima must match")

    loss = _required_mapping(config, "loss", "doma.loss")
    _reject_unknown(
        loss,
        (
            "object_loss_weight",
            "individual_loss_weight",
            "consensus_loss_weight",
            "quality_loss_weight",
        ),
        "doma.loss",
    )
    for key in (
        "object_loss_weight",
        "individual_loss_weight",
        "consensus_loss_weight",
        "quality_loss_weight",
    ):
        _nonnegative_real(loss.get(key), "doma.loss.%s" % key)
    if not object_enabled:
        active = any(
            (
                adapter_enabled,
                encoder_enabled,
                geometry_enabled,
                refiner_enabled,
                multi_enabled,
                quality_enabled,
                consensus_enabled,
            )
        )
        if active:
            raise ValueError("doma.object_space.enabled=false requires all DOMA modules off")
    if refiner_enabled and not encoder_enabled:
        raise ValueError("doma.refiner requires doma.object_encoder")
    if quality_enabled and not (encoder_enabled and refiner_enabled):
        raise ValueError("doma.quality requires object_encoder and refiner")
    if protocol_enabled and not (adapter_enabled and encoder_enabled):
        raise ValueError(
            "doma.object_protocol_alignment requires object_adapter and object_encoder"
        )
    if protocol_context_enabled and not context_enabled:
        raise ValueError(
            "doma.object_protocol_alignment.branches.context requires the "
            "DOMA Context path"
        )
    if protocol_enabled and (not include_gt or jitters_per_gt < 1):
        raise ValueError(
            "doma.object_protocol_alignment requires include_gt=true and "
            "jitters_per_gt>=1"
        )
    if protocol_enabled and active_modality == "m1":
        raise ValueError(
            "doma.object_protocol_alignment requires a non-anchor Stage2 "
            "active_modality with an adapter"
        )
    if protocol_enabled and not any(proposals[key] > 0 for key in jitter_std_keys):
        raise ValueError(
            "doma.object_protocol_alignment requires at least one non-zero "
            "training proposal jitter standard deviation"
        )
    if refinement_enabled and not (encoder_enabled and refiner_enabled):
        raise ValueError(
            "doma.quality_aware_refinement requires object_encoder and refiner"
        )
    if delta_iou_diagnostics_enabled and not (
        object_enabled and encoder_enabled and refiner_enabled
    ):
        raise ValueError(
            "doma.delta_iou_diagnostics requires object space, object_encoder, "
            "and refiner"
        )
    if consensus_enabled and not refiner_enabled:
        raise ValueError("doma.consensus requires doma.refiner")
    if multi_enabled and not (object_enabled and encoder_enabled and detail_enabled):
        raise ValueError("multi-granularity DOMA requires object space, detail, and object encoder")
    if (protocol_enabled or refinement_enabled) and not ablation:
        raise ValueError(
            "experimental OPA/QAR modules require doma.ablation=true"
        )

    if ablation:
        _validate_version_ceiling(version, multi_enabled, quality_enabled)
    else:
        _validate_canonical_profile(
            version,
            object_enabled,
            adapter_enabled,
            encoder_enabled,
            geometry_enabled,
            refiner_enabled,
            consensus_enabled,
            consensus_mode,
            multi_enabled,
            detail_enabled,
            context_enabled,
            fusion_enabled,
            quality_enabled,
            refiner["yaw_mode"],
        )
    return config


def doma_feature_flags(config):
    """Resolve explicit runtime switches from an already validated config."""
    if config is None or config.get("enabled") is not True:
        return {
            "enabled": False,
            "object_space": False,
            "object_adapter": False,
            "object_encoder": False,
            "geometry": False,
            "refiner": False,
            "consensus": False,
            "multi_granularity": False,
            "context": False,
            "quality": False,
            "object_protocol_alignment": False,
            "quality_aware_refinement": False,
            "quality_aware_refinement_training_loss": False,
            "quality_aware_refinement_inference_gate": False,
            "delta_iou_diagnostics": False,
            "delta_iou_diagnostics_training": False,
            "delta_iou_diagnostics_inference": False,
        }
    multi = config["multi_granularity"]
    protocol = config.get("object_protocol_alignment", {})
    refinement = config.get("quality_aware_refinement", {})
    refinement_enabled = refinement.get("enabled") is True
    qar_training_configured = bool(
        refinement_enabled
        and refinement.get("training_loss", {}).get("enabled") is True
    )
    qar_gate_configured = bool(
        refinement_enabled
        and refinement.get("inference_gate", {}).get("enabled") is True
    )
    diagnostics = config.get("delta_iou_diagnostics", {})
    mode = config["mode"]
    training_mode = mode in VALID_TRAINING_MODES
    qar_training_apply_to = refinement.get("training_loss", {}).get(
        "apply_to", ()
    )
    diagnostics_apply_to = diagnostics.get("apply_to", ())
    diagnostics_enabled = bool(
        diagnostics.get("enabled") is True and mode in diagnostics_apply_to
    )
    return {
        "enabled": True,
        "object_space": bool(config["object_space"]["enabled"]),
        "object_adapter": bool(config["object_adapter"]["enabled"]),
        "object_encoder": bool(config["object_encoder"]["enabled"]),
        "geometry": bool(config["geometry"]["enabled"]),
        "refiner": bool(config["refiner"]["enabled"]),
        "consensus": bool(config["consensus"]["enabled"]),
        "multi_granularity": bool(multi["enabled"]),
        "context": bool(multi.get("enabled") and multi.get("context", {}).get("enabled")),
        "quality": bool(config["quality"]["enabled"]),
        "object_protocol_alignment": bool(
            protocol.get("enabled") is True
            and config["mode"] == "stage2_adapt"
        ),
        "quality_aware_refinement": bool(
            qar_training_configured or qar_gate_configured
        ),
        "quality_aware_refinement_training_loss": bool(
            qar_training_configured and mode in qar_training_apply_to
        ),
        "quality_aware_refinement_inference_gate": bool(
            qar_gate_configured and config["mode"] == "inference"
        ),
        "delta_iou_diagnostics": bool(
            diagnostics_enabled
        ),
        "delta_iou_diagnostics_training": bool(
            diagnostics_enabled and training_mode
        ),
        "delta_iou_diagnostics_inference": bool(
            diagnostics_enabled and mode == "inference"
        ),
    }


def _validate_canonical_profile(
    version,
    object_enabled,
    adapter_enabled,
    encoder_enabled,
    geometry_enabled,
    refiner_enabled,
    consensus_enabled,
    consensus_mode,
    multi_enabled,
    detail_enabled,
    context_enabled,
    fusion_enabled,
    quality_enabled,
    yaw_mode,
):
    common = all(
        (
            object_enabled,
            adapter_enabled,
            encoder_enabled,
            geometry_enabled,
            refiner_enabled,
            consensus_enabled,
        )
    )
    expected = {
        "v1": (False, False, "uniform_geometry_mean", "sin_cos"),
        "v2": (True, False, "uniform_geometry_mean", "sin_cos_centered"),
        "v3": (True, True, "quality_weighted", "sin_cos_centered"),
    }[version]
    expected_multi, expected_quality, expected_consensus, expected_yaw = expected
    multi_complete = multi_enabled and detail_enabled and context_enabled and fusion_enabled
    valid = (
        common
        and multi_enabled == expected_multi
        and (not multi_enabled or multi_complete)
        and quality_enabled == expected_quality
        and consensus_mode == expected_consensus
        and yaw_mode == expected_yaw
    )
    if not valid:
        raise ValueError(
            "non-canonical DOMA %s combination; set doma.ablation=true for an explicit ablation"
            % version
        )


def _validate_version_ceiling(version, multi_enabled, quality_enabled):
    if version == "v1" and (multi_enabled or quality_enabled):
        raise ValueError("DOMA V1 ablations cannot enable V2/V3 modules")
    if version == "v2" and quality_enabled:
        raise ValueError("DOMA V2 ablations cannot enable the V3 quality module")


def _reject_unknown(config, allowed, name):
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError("%s has unknown keys: %s" % (name, ", ".join(unknown)))


def _required_mapping(config, key, name):
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError("%s must be a mapping" % name)
    return value


def _optional_feature_mapping(config, key, name):
    """Return an optional feature block and its explicit enable state."""
    if key not in config:
        return {}, False
    value = config[key]
    if not isinstance(value, dict):
        raise TypeError("%s must be a mapping" % name)
    enabled = _optional_bool(value, "enabled", False, "%s.enabled" % name)
    if not enabled:
        _reject_unknown(value, ("enabled",), name)
    return value, enabled


def _required_bool(config, key, name):
    if key not in config or type(config[key]) is not bool:
        raise TypeError("%s must be bool" % name)
    return config[key]


def _required_mode_list(config, key, name, allowed):
    value = config.get(key)
    if not isinstance(value, list):
        raise TypeError("%s must be a list" % name)
    if not value:
        raise ValueError("%s must not be empty" % name)
    invalid = [item for item in value if item not in allowed]
    if invalid:
        raise ValueError(
            "%s entries must be one of %s; got %r"
            % (name, tuple(allowed), invalid)
        )
    if len(value) != len(set(value)):
        raise ValueError("%s must not contain duplicate modes" % name)
    return tuple(value)


def _optional_bool(config, key, default, name):
    value = config.get(key, default)
    if type(value) is not bool:
        raise TypeError("%s must be bool" % name)
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _positive_real(value, name):
    value = _finite_real(value, name)
    if value <= 0.0:
        raise ValueError("%s must be positive" % name)
    return value


def _nonnegative_real(value, name):
    value = _finite_real(value, name)
    if value < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return value


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _closed_interval(value, lower, upper, name):
    value = _finite_real(value, name)
    if value < lower or value > upper:
        raise ValueError("%s must be in [%s,%s]" % (name, lower, upper))
    return value


def _unit_interval(value, name, allow_zero=False):
    value = _finite_real(value, name)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        bracket = "[0,1]" if allow_zero else "(0,1]"
        raise ValueError("%s must be in %s" % (name, bracket))
    return value
