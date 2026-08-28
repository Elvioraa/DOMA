"""DOMA object-centric alignment modules and runtime plumbing.

Only explicitly enabled modules are instantiated.  V1 contains the detail
object path, V2 adds context-scale representation, and V3 adds quality-aware
consensus.  No Official HEAL source imports this module.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.doma_box_coder import (
    aligned_rotated_bev_iou_hwl,
    boxes_hwl_to_corners_3d,
    corners_3d_to_boxes_hwl,
    decode_box_residual,
    encode_box_residual,
)
from opencood.models.sub_modules.doma_config import (
    doma_feature_flags,
    validate_doma_config,
)
from opencood.models.sub_modules.doma_object_roi import (
    ChunkedRotatedBEVROISampler,
    DOMABEVGeometry,
)
from opencood.models.sub_modules.doma_proposal_sampler import (
    DOMATrainingProposalSampler,
)


SHARED_DOMA_PREFIXES = (
    "doma_shared_object_encoder.",
    "doma_shared_geometry_encoder.",
    "doma_shared_object_refiner.",
    "doma_shared_context_encoder.",
    "doma_shared_multigranularity_fusion.",
    "doma_shared_quality_head.",
)
CORE_SHARED_DOMA_PREFIXES = SHARED_DOMA_PREFIXES[:3]
MULTISCALE_SHARED_DOMA_PREFIXES = SHARED_DOMA_PREFIXES[3:5]
QUALITY_SHARED_DOMA_PREFIXES = SHARED_DOMA_PREFIXES[5:]


class ResidualObjectAdapter(nn.Module):
    """Lightweight modality adapter initialized as an exact identity."""

    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.delta = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, roi_features):
        """Return identity plus a learned residual for ``[M,C,Rh,Rw]``."""
        return roi_features + self.delta(roi_features)


class SharedObjectEncoder(nn.Module):
    """Map adapted Common-BEV ROIs into one shared object representation."""

    def __init__(self, in_channels, hidden_channels, pooled_size, embedding_dim):
        super().__init__()
        groups = _group_count(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_size * pooled_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, roi_features):
        """Encode ``[M,C,Rh,Rw]`` as ``[M,embedding_dim]``."""
        return self.projection(self.features(roi_features))


class SharedGeometryEncoder(nn.Module):
    """Encode normalized proposal geometry from 8D to a shared 32D code."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, raw_geometry):
        """Encode a ``[M,8]`` normalized geometry tensor."""
        return self.network(raw_geometry)


class SharedGeometryRefiner(nn.Module):
    """Decode object and proposal embeddings into periodic 8D residuals."""

    def __init__(self, embedding_dim, geometry_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, object_embedding, geometry_embedding):
        """Predict residuals for matching object and geometry embeddings."""
        if object_embedding.shape[0] != geometry_embedding.shape[0]:
            raise ValueError("object and geometry embedding counts must match")
        return self.network(torch.cat((object_embedding, geometry_embedding), dim=-1))


class SharedMultiScaleFusion(nn.Module):
    """Fuse detail/context embeddings while starting as exact detail-only."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, detail_embedding, context_embedding):
        """Return ``[M,D]`` residual-safe concat-projection fusion."""
        if detail_embedding.shape != context_embedding.shape:
            raise ValueError("detail and context embeddings must have equal shape")
        projected = self.projection(
            torch.cat((detail_embedding, context_embedding), dim=-1)
        )
        return detail_embedding + self.residual_scale * projected


class SharedObjectQualityHead(nn.Module):
    """Predict agent-object-specific scalar geometry quality in ``[0,1]``."""

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

    def forward(
        self,
        object_embedding,
        geometry_embedding,
        roi_coverage=None,
        agent_distance=None,
    ):
        """Return scalar quality for matching valid agent-object pairs."""
        values = [object_embedding, geometry_embedding]
        if self.use_roi_coverage:
            values.append(_quality_scalar(roi_coverage, object_embedding, "roi_coverage"))
        if self.use_agent_distance:
            values.append(_quality_scalar(agent_distance, object_embedding, "agent_distance"))
        return torch.sigmoid(self.network(torch.cat(values, dim=-1))).squeeze(-1)


def doma_is_enabled(args):
    """Return the explicit DOMA enable flag while rejecting ambiguous values."""
    config = args.get("doma")
    if config is None:
        return False
    if not isinstance(config, dict):
        raise TypeError("model.args.doma must be a mapping")
    validate_doma_config(config)
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("doma.enabled must be bool")
    return enabled



def install_doma_modules(model, args):
    """Install only explicitly enabled DOMA modules, preserving RNG isolation."""
    model.doma_enabled = doma_is_enabled(args)
    model._doma_log_printed = False
    if not model.doma_enabled:
        return

    config = validate_doma_config(args["doma"])
    if (
        config["mode"] == "stage2_adapt"
        and config["active_modality"] not in model.modality_name_list
    ):
        raise ValueError("doma.active_modality must be present in the Stage2 model")

    model.doma_config = config
    model.doma_flags = doma_feature_flags(config)
    if not model.doma_flags["object_space"]:
        return

    channels = infer_common_bev_channels(args, model.modality_name_list)
    bev_geometry = DOMABEVGeometry.from_lidar_range(args["lidar_range"])
    roi_config = config["object_space"]["roi"]
    encoder_config = config["object_encoder"]
    geometry_config = config["geometry"]
    refiner_config = config["refiner"]
    multi_config = config["multi_granularity"]

    model.doma_common_bev_channels = channels
    model.doma_bev_geometry = bev_geometry
    detail_output_size = (
        multi_config["detail"]["roi_size"]
        if model.doma_flags["multi_granularity"]
        else roi_config["output_size"]
    )
    model.doma_object_roi = ChunkedRotatedBEVROISampler(
        bev_geometry=bev_geometry,
        output_size=detail_output_size,
        chunk_size=roi_config["chunk_size"],
        min_coverage=roi_config["min_coverage"],
    )
    model.doma_training_proposal_sampler = DOMATrainingProposalSampler(
        config["training_proposals"],
        max_proposals=config["training_proposals"]["max_proposals"],
    )

    if model.doma_flags["object_adapter"]:
        for modality in model.modality_name_list:
            if modality != "m1":
                setattr(
                    model,
                    "doma_object_adapter_%s" % modality,
                    ResidualObjectAdapter(channels),
                )

    if model.doma_flags["object_encoder"]:
        model.doma_shared_object_encoder = SharedObjectEncoder(
            in_channels=channels,
            hidden_channels=encoder_config["hidden_channels"],
            pooled_size=encoder_config["pooled_size"],
            embedding_dim=encoder_config["embedding_dim"],
        )
    if model.doma_flags["geometry"]:
        model.doma_shared_geometry_encoder = SharedGeometryEncoder(
            geometry_config["hidden_dim"]
        )
    if model.doma_flags["refiner"]:
        model.doma_shared_object_refiner = SharedGeometryRefiner(
            embedding_dim=encoder_config["embedding_dim"],
            geometry_dim=geometry_config["hidden_dim"],
            hidden_dim=refiner_config["hidden_dim"],
        )

    if model.doma_flags["context"]:
        context_channels = infer_context_bev_channels(args)
        context_config = multi_config["context"]
        model.doma_context_bev_channels = context_channels
        model.doma_context_roi = ChunkedRotatedBEVROISampler(
            bev_geometry=bev_geometry,
            output_size=context_config["roi_size"],
            chunk_size=roi_config["chunk_size"],
            min_coverage=roi_config["min_coverage"],
        )
        if model.doma_flags["object_adapter"]:
            for modality in model.modality_name_list:
                if modality != "m1":
                    setattr(
                        model,
                        "doma_context_adapter_%s" % modality,
                        ResidualObjectAdapter(context_channels),
                    )
        model.doma_shared_context_encoder = SharedObjectEncoder(
            in_channels=context_channels,
            hidden_channels=encoder_config["hidden_channels"],
            pooled_size=encoder_config["pooled_size"],
            embedding_dim=encoder_config["embedding_dim"],
        )
        model.doma_shared_multigranularity_fusion = SharedMultiScaleFusion(
            encoder_config["embedding_dim"]
        )

    if model.doma_flags["quality"]:
        quality_config = config["quality"]
        model.doma_shared_quality_head = SharedObjectQualityHead(
            embedding_dim=encoder_config["embedding_dim"],
            geometry_dim=geometry_config["hidden_dim"],
            hidden_dim=quality_config["hidden_dim"],
            use_roi_coverage=quality_config["use_roi_coverage"],
            use_agent_distance=quality_config["use_agent_distance"],
        )



def configure_doma_trainability(model):
    """Apply Stage1/Stage2 ownership without changing Official HEAL modules."""
    if not getattr(model, "doma_enabled", False):
        return
    mode = model.doma_config["mode"]
    shared_trainable = mode == "stage1_anchor"
    shared_names = (
        "doma_shared_object_encoder",
        "doma_shared_geometry_encoder",
        "doma_shared_object_refiner",
        "doma_shared_context_encoder",
        "doma_shared_multigranularity_fusion",
        "doma_shared_quality_head",
    )
    for name in shared_names:
        if hasattr(model, name):
            _set_module_trainability(
                getattr(model, name), shared_trainable, model.training
            )

    active_modality = model.doma_config.get("active_modality")
    for modality in model.modality_name_list:
        adapter_trainable = mode == "stage2_adapt" and modality == active_modality
        for namespace in ("object_adapter", "context_adapter"):
            name = "doma_%s_%s" % (namespace, modality)
            if hasattr(model, name):
                _set_module_trainability(
                    getattr(model, name), adapter_trainable, model.training
                )

    if not model._doma_log_printed:
        _print_doma_summary(model, shared_trainable, active_modality)
        model._doma_log_printed = True


def build_collab_doma_context(
    model,
    agent_features,
    record_len,
    affine_matrix,
    agent_modality_list,
    pairwise_t_matrix=None,
):
    """Warp per-agent Common-BEV maps to ego while retaining agent identity."""
    if not getattr(model, "doma_enabled", False):
        return None
    if agent_features.ndim != 4:
        raise ValueError("agent_features must have shape [sum(A),C,H,W]")
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    if sum(lengths) != int(agent_features.shape[0]):
        raise ValueError("record_len does not match Common-BEV agent count")
    if len(agent_modality_list) != int(agent_features.shape[0]):
        raise ValueError("agent_modality_list does not match Common-BEV agent count")

    source_support = _build_source_support(model, agent_features, agent_modality_list)
    scenes = []
    offset = 0
    _, channels, height, width = agent_features.shape
    for batch_index, agent_count in enumerate(lengths):
        if agent_count < 1:
            raise ValueError("every scene must contain at least one agent")
        scene_features = agent_features[offset:offset + agent_count]
        scene_support = source_support[offset:offset + agent_count]
        transforms = affine_matrix[
            batch_index, 0, :agent_count
        ].to(device=agent_features.device, dtype=agent_features.dtype)
        grid = F.affine_grid(
            transforms,
            (agent_count, channels, height, width),
            align_corners=False,
        )
        aligned_features = F.grid_sample(
            scene_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        aligned_support = F.grid_sample(
            scene_support,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        scene = {
            "agent_features": aligned_features,
            "agent_support": aligned_support,
            "agent_modalities": tuple(
                agent_modality_list[offset:offset + agent_count]
            ),
        }
        if (
            model.doma_flags["quality"]
            and model.doma_config["quality"]["use_agent_distance"]
        ):
            if pairwise_t_matrix is None:
                raise ValueError(
                    "quality-aware DOMA context requires pairwise_t_matrix"
                )
            scene["agent_positions"] = _agent_positions_in_ego(
                pairwise_t_matrix[batch_index, 0, :agent_count],
                agent_features,
            )
        scenes.append(scene)
        offset += agent_count
    return {"scenes": tuple(scenes), "box_order": "hwl", "aligned_to": "ego"}


def build_single_doma_context(model, feature, modality_name):
    """Build one-agent-per-sample ego-aligned contexts for HEAL Stage2."""
    if not getattr(model, "doma_enabled", False):
        return None
    if modality_name not in model.modality_name_list:
        raise ValueError("unknown active modality %s" % modality_name)
    modalities = [modality_name] * int(feature.shape[0])
    support = _build_source_support(model, feature, modalities)
    scenes = []
    for batch_index in range(int(feature.shape[0])):
        scene = {
            "agent_features": feature[batch_index:batch_index + 1],
            "agent_support": support[batch_index:batch_index + 1],
            "agent_modalities": (modality_name,),
        }
        if (
            model.doma_flags["quality"]
            and model.doma_config["quality"]["use_agent_distance"]
        ):
            scene["agent_positions"] = feature.new_zeros((1, 2))
        scenes.append(scene)
    return {"scenes": tuple(scenes), "box_order": "hwl", "aligned_to": "ego"}



def attach_collab_doma_pyramid_context(
    model,
    context,
    pre_fusion_features,
    record_len,
    affine_matrix,
    agent_modality_list,
):
    """Attach the pre-fusion level-1 Context feature only when enabled."""
    if context is None or not model.doma_flags["context"]:
        return context
    if not isinstance(pre_fusion_features, (tuple, list)) or len(pre_fusion_features) < 2:
        raise RuntimeError("multi-granularity DOMA requires pyramid level 1")
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    feature = pre_fusion_features[1]
    if int(feature.shape[0]) != sum(lengths):
        raise ValueError("pyramid feature agent count does not match record_len")
    source_support = _build_source_support(model, feature, agent_modality_list)
    aligned_scenes = []
    aligned_support_scenes = []
    offset = 0
    for batch_index, agent_count in enumerate(lengths):
        transforms = affine_matrix[batch_index, 0, :agent_count].to(
            device=feature.device, dtype=feature.dtype
        )
        aligned_scenes.append(
            _warp_scene_features(
                feature[offset : offset + agent_count], transforms, "bilinear"
            )
        )
        aligned_support_scenes.append(
            _warp_scene_features(
                source_support[offset : offset + agent_count], transforms, "nearest"
            )
        )
        offset += agent_count
    for scene_index, scene in enumerate(context["scenes"]):
        scene["context_agent_features"] = aligned_scenes[scene_index]
        scene["context_agent_support"] = aligned_support_scenes[scene_index]
    return context



def attach_single_doma_pyramid_context(
    model, context, pre_fusion_features, modality_name
):
    """Attach Stage2 single-agent Context features without spatial warp."""
    if context is None or not model.doma_flags["context"]:
        return context
    if not isinstance(pre_fusion_features, (tuple, list)) or len(pre_fusion_features) < 2:
        raise RuntimeError("multi-granularity DOMA requires pyramid level 1")
    context_feature = pre_fusion_features[1]
    support = _build_source_support(
        model,
        context_feature,
        [modality_name] * int(context_feature.shape[0]),
    )
    if len(context["scenes"]) != int(context_feature.shape[0]):
        raise ValueError("single pyramid feature batch does not match context")
    for batch_index, scene in enumerate(context["scenes"]):
        scene["context_agent_features"] = context_feature[
            batch_index : batch_index + 1
        ]
        scene["context_agent_support"] = support[batch_index : batch_index + 1]
    return context



def run_doma_training(model, context, data_dict, detector_output=None):
    """Run object-space supervision for Stage1 or independent Stage2."""
    del detector_output
    if (
        not getattr(model, "doma_enabled", False)
        or not model.doma_flags["object_space"]
        or not model.doma_flags["refiner"]
    ):
        return None
    mode = model.doma_config["mode"]
    yaw_mode = model.doma_config["refiner"]["yaw_mode"]
    if mode not in ("stage1_anchor", "stage2_adapt"):
        return None
    if "object_bbx_center" not in data_dict or "object_bbx_mask" not in data_dict:
        raise KeyError("DOMA training requires object_bbx_center and object_bbx_mask")
    gt_boxes = data_dict["object_bbx_center"]
    gt_mask = data_dict["object_bbx_mask"]
    if gt_boxes.ndim != 3 or gt_boxes.shape[-1] != 7:
        raise ValueError("object_bbx_center must have shape [B,M,7] in hwl order")
    if gt_boxes.shape[:2] != gt_mask.shape[:2]:
        raise ValueError("object_bbx_mask must match object_bbx_center [B,M]")
    if context is None or len(context["scenes"]) != int(gt_boxes.shape[0]):
        raise ValueError("DOMA context scene count does not match GT batch")

    if mode == "stage2_adapt":
        for name in (
            "doma_shared_object_encoder",
            "doma_shared_geometry_encoder",
            "doma_shared_object_refiner",
            "doma_shared_context_encoder",
            "doma_shared_multigranularity_fusion",
            "doma_shared_quality_head",
        ):
            if hasattr(model, name):
                getattr(model, name).eval()

    scene_outputs = []
    total_pairs = 0
    valid_pairs = 0
    coverage_sum = gt_boxes.new_zeros(())
    proposal_count = 0
    for scene_index, scene in enumerate(context["scenes"]):
        proposals, targets = model.doma_training_proposal_sampler(
            gt_boxes[scene_index],
            gt_mask[scene_index],
            with_jitter=bool(model.training),
        )
        result = predict_scene_residuals(model, scene, proposals)
        target_residuals = encode_box_residual(
            proposals, targets, yaw_mode=yaw_mode
        )
        pair_indices = result["valid_mask"].nonzero(as_tuple=False)
        individual_targets = (
            target_residuals.index_select(0, pair_indices[:, 0])
            if pair_indices.numel()
            else target_residuals.new_empty((0, 8))
        )
        result.update(
            {
                "targets": targets,
                "target_residuals": target_residuals,
                "individual_targets": individual_targets,
            }
        )
        if model.doma_flags["quality"]:
            if pair_indices.numel():
                selected_proposals = proposals.index_select(0, pair_indices[:, 0])
                selected_targets = targets.index_select(0, pair_indices[:, 0])
                individual_boxes = decode_box_residual(
                    selected_proposals,
                    result["individual_residuals"],
                    yaw_mode=yaw_mode,
                )
                quality_targets = aligned_rotated_bev_iou_hwl(
                    individual_boxes.detach(), selected_targets.detach()
                )
            else:
                quality_targets = proposals.new_empty((0,))
            result["quality_targets"] = quality_targets.detach()
        scene_outputs.append(result)
        proposal_count += int(proposals.shape[0])
        total_pairs += int(result["valid_mask"].numel())
        valid_pairs += int(result["valid_mask"].sum().item())
        coverage_sum = coverage_sum + result["coverage"].detach().sum()

    denominator = max(total_pairs, 1)
    payload = {
        "enabled": True,
        "version": model.doma_config["version"],
        "mode": mode,
        "scenes": tuple(scene_outputs),
        "loss_config": dict(model.doma_config["loss"]),
        "consensus_enabled": bool(model.doma_flags["consensus"]),
        "stats": {
            "object_roi_count": proposal_count,
            "valid_agent_object_pairs": valid_pairs,
            "valid_object_ratio": float(valid_pairs) / float(denominator),
            "mean_roi_coverage": float(coverage_sum.item()) / float(denominator),
        },
    }
    if model.doma_flags["quality"]:
        payload["quality_enabled"] = True
    return payload



def predict_scene_residuals(model, scene, proposals):
    """Predict per-agent residuals and the explicitly selected consensus."""
    if not (model.doma_flags["object_encoder"] and model.doma_flags["refiner"]):
        raise RuntimeError("DOMA object encoder and refiner are required")
    yaw_mode = model.doma_config["refiner"]["yaw_mode"]
    agent_features = scene["agent_features"]
    roi_features, detail_valid, detail_coverage = model.doma_object_roi(
        agent_features, proposals, scene.get("agent_support")
    )
    valid_mask = detail_valid
    coverage = detail_coverage
    context_roi_features = None
    context_coverage = None
    if model.doma_flags["context"]:
        if "context_agent_features" not in scene:
            raise RuntimeError("DOMA scene is missing Context pyramid features")
        context_roi_features, context_valid, context_coverage = (
            model.doma_context_roi(
                scene["context_agent_features"],
                proposals,
                scene.get("context_agent_support"),
            )
        )
        valid_mask = detail_valid & context_valid
        coverage = torch.minimum(detail_coverage, context_coverage)

    proposal_count, agent_count = valid_mask.shape
    valid_indices = valid_mask.nonzero(as_tuple=False)
    individual_quality = None
    if valid_indices.numel():
        proposal_indices = valid_indices[:, 0]
        agent_indices = valid_indices[:, 1]
        selected_modalities = [
            scene["agent_modalities"][int(index)] for index in agent_indices.tolist()
        ]
        selected_rois = roi_features[proposal_indices, agent_indices]
        adapted_rois = route_modality_adapters(
            model, selected_rois, selected_modalities
        )
        detail_embedding = model.doma_shared_object_encoder(adapted_rois)
        object_embedding = detail_embedding
        if model.doma_flags["context"]:
            selected_context_rois = context_roi_features[
                proposal_indices, agent_indices
            ]
            adapted_context_rois = route_modality_adapters(
                model,
                selected_context_rois,
                selected_modalities,
                adapter_namespace="context_adapter",
            )
            context_embedding = model.doma_shared_context_encoder(
                adapted_context_rois
            )
            object_embedding = model.doma_shared_multigranularity_fusion(
                detail_embedding, context_embedding
            )

        selected_proposals = proposals.index_select(0, proposal_indices)
        raw_geometry = proposal_geometry_raw(
            selected_proposals, model.doma_bev_geometry
        ).to(device=object_embedding.device, dtype=object_embedding.dtype)
        geometry_dim = model.doma_config["geometry"]["hidden_dim"]
        if model.doma_flags["geometry"]:
            geometry_embedding = model.doma_shared_geometry_encoder(raw_geometry)
        else:
            geometry_embedding = object_embedding.new_zeros(
                (object_embedding.shape[0], geometry_dim)
            )
        individual_residuals = model.doma_shared_object_refiner(
            object_embedding, geometry_embedding
        )
        per_agent_residuals = individual_residuals.new_zeros(
            (proposal_count, agent_count, 8)
        ).index_put((proposal_indices, agent_indices), individual_residuals)

        if model.doma_flags["quality"]:
            normalized_distances = None
            if model.doma_config["quality"]["use_agent_distance"]:
                normalized_distances = normalized_agent_object_distance(
                    scene, proposals, model.doma_bev_geometry
                )
            individual_quality = model.doma_shared_quality_head(
                object_embedding,
                geometry_embedding,
                roi_coverage=coverage[proposal_indices, agent_indices],
                agent_distance=(
                    normalized_distances[proposal_indices, agent_indices]
                    if normalized_distances is not None
                    else None
                ),
            )
            per_agent_quality = individual_quality.new_zeros(
                (proposal_count, agent_count)
            ).index_put((proposal_indices, agent_indices), individual_quality)
    else:
        individual_residuals = agent_features.new_empty((0, 8))
        per_agent_residuals = agent_features.new_zeros(
            (proposal_count, agent_count, 8)
        )
        if model.doma_flags["quality"]:
            individual_quality = agent_features.new_empty((0,))
            per_agent_quality = agent_features.new_zeros(
                (proposal_count, agent_count)
            )

    if not model.doma_flags["consensus"]:
        any_valid = valid_mask[:, 0] if agent_count else valid_mask.new_zeros(
            (proposal_count,)
        )
        fused_residuals = (
            per_agent_residuals[:, 0]
            if agent_count
            else per_agent_residuals.new_zeros((proposal_count, 8))
        )
        consensus_weights = quality_fallback = None
    elif model.doma_config["consensus"]["mode"] == "quality_weighted":
        quality_config = model.doma_config["quality"]
        fused_residuals, any_valid, consensus_weights, quality_fallback = (
            quality_weighted_geometry_consensus(
                per_agent_residuals,
                valid_mask,
                per_agent_quality,
                min_quality_sum=model.doma_config["consensus"]["min_quality_sum"],
                detach_quality=quality_config["detach_weight_for_consensus"],
            )
        )
    else:
        fused_residuals, any_valid = uniform_geometry_consensus(
            per_agent_residuals, valid_mask
        )
        consensus_weights = quality_fallback = None

    decoded = decode_box_residual(
        proposals, fused_residuals, yaw_mode=yaw_mode
    )
    result = {
        "proposals": proposals,
        "individual_residuals": individual_residuals,
        "per_agent_residuals": per_agent_residuals,
        "fused_residuals": fused_residuals,
        "refined_boxes": torch.where(any_valid[:, None], decoded, proposals),
        "valid_mask": valid_mask,
        "any_valid": any_valid,
        "coverage": coverage,
    }
    if model.doma_flags["context"]:
        result.update(
            {
                "detail_coverage": detail_coverage,
                "context_coverage": context_coverage,
            }
        )
    if model.doma_flags["quality"]:
        result.update(
            {
                "individual_quality": individual_quality,
                "per_agent_quality": per_agent_quality,
                "consensus_weights": consensus_weights,
                "quality_fallback": quality_fallback,
            }
        )
    return result



def route_modality_adapters(
    model, roi_features, modality_names, adapter_namespace="object_adapter"
):
    """Route ROIs by modality; a disabled adapter is a true identity path."""
    if roi_features.shape[0] != len(modality_names):
        raise ValueError("modality_names must match ROI count")
    if roi_features.shape[0] == 0 or not model.doma_flags["object_adapter"]:
        return roi_features
    grouped_indices = OrderedDict()
    for index, modality in enumerate(modality_names):
        grouped_indices.setdefault(modality, []).append(index)

    output_parts = []
    position_parts = []
    for modality, indices in grouped_indices.items():
        attribute = "doma_%s_%s" % (adapter_namespace, modality)
        positions = torch.tensor(indices, dtype=torch.long, device=roi_features.device)
        selected = roi_features.index_select(0, positions)
        output_parts.append(
            getattr(model, attribute)(selected)
            if hasattr(model, attribute)
            else selected
        )
        position_parts.append(positions)
    packed_outputs = torch.cat(output_parts, dim=0)
    packed_positions = torch.cat(position_parts, dim=0)
    return packed_outputs.index_select(0, torch.argsort(packed_positions))


def proposal_geometry_raw(proposals, geometry):
    """Build ``[x_norm,y_norm,z_norm,log(l,w,h),sin(yaw),cos(yaw)]``."""
    if not isinstance(geometry, DOMABEVGeometry):
        raise TypeError("geometry must be DOMABEVGeometry")
    if not torch.is_tensor(proposals) or proposals.ndim != 2 or proposals.shape[1] != 7:
        raise ValueError("proposals must have shape [N,7] in hwl order")
    if not torch.is_floating_point(proposals):
        raise TypeError("proposals must use a floating-point dtype")
    if proposals.numel() and not bool((proposals[:, 3:6] > 0).all()):
        raise ValueError("proposal height, width, and length must be positive")
    x_norm = 2.0 * (proposals[:, 0] - geometry.x_min) / (
        geometry.x_max - geometry.x_min
    ) - 1.0
    y_norm = 2.0 * (proposals[:, 1] - geometry.y_min) / (
        geometry.y_max - geometry.y_min
    ) - 1.0
    z_norm = 2.0 * (proposals[:, 2] - geometry.z_min) / (
        geometry.z_max - geometry.z_min
    ) - 1.0
    return torch.stack(
        (
            x_norm,
            y_norm,
            z_norm,
            torch.log(proposals[:, 5]),
            torch.log(proposals[:, 4]),
            torch.log(proposals[:, 3]),
            torch.sin(proposals[:, 6]),
            torch.cos(proposals[:, 6]),
        ),
        dim=-1,
    )


def uniform_geometry_consensus(per_agent_residuals, valid_mask):
    """Average valid decoded-geometry residual components uniformly."""
    if per_agent_residuals.ndim != 3 or per_agent_residuals.shape[-1] != 8:
        raise ValueError("per_agent_residuals must have shape [N,A,8]")
    if tuple(valid_mask.shape) != tuple(per_agent_residuals.shape[:2]):
        raise ValueError("valid_mask must have shape [N,A]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    weights = valid_mask.to(dtype=per_agent_residuals.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1)
    any_valid = counts[:, 0] > 0
    fused = (per_agent_residuals * weights).sum(dim=1) / counts.clamp_min(1.0)
    fused = torch.where(any_valid[:, None], fused, torch.zeros_like(fused))
    return fused, any_valid


def quality_weighted_geometry_consensus(
    per_agent_residuals,
    valid_mask,
    per_agent_quality,
    min_quality_sum=1e-6,
    detach_quality=True,
):
    """Fuse residuals with scalar quality and deterministic uniform fallback."""
    if per_agent_residuals.ndim != 3 or per_agent_residuals.shape[-1] != 8:
        raise ValueError("per_agent_residuals must have shape [N,A,8]")
    if tuple(valid_mask.shape) != tuple(per_agent_residuals.shape[:2]):
        raise ValueError("valid_mask must have shape [N,A]")
    if tuple(per_agent_quality.shape) != tuple(valid_mask.shape):
        raise ValueError("per_agent_quality must have shape [N,A]")
    if type(detach_quality) is not bool:
        raise TypeError("detach_quality must be bool")
    min_quality_sum = float(min_quality_sum)
    if min_quality_sum <= 0.0:
        raise ValueError("min_quality_sum must be positive")

    quality = per_agent_quality.detach() if detach_quality else per_agent_quality
    valid = valid_mask.to(dtype=per_agent_residuals.dtype)
    quality_weights = quality.clamp(0.0, 1.0) * valid
    quality_sum = quality_weights.sum(dim=1, keepdim=True)
    valid_count = valid.sum(dim=1, keepdim=True)
    any_valid = valid_count[:, 0] > 0
    fallback = any_valid & (quality_sum[:, 0] < min_quality_sum)
    normalized_quality = quality_weights / quality_sum.clamp_min(min_quality_sum)
    uniform_weights = valid / valid_count.clamp_min(1.0)
    weights = torch.where(
        fallback[:, None], uniform_weights, normalized_quality
    )
    weights = torch.where(any_valid[:, None], weights, torch.zeros_like(weights))
    quality_fused = (
        per_agent_residuals * normalized_quality.unsqueeze(-1)
    ).sum(dim=1)
    uniform_fused, _ = uniform_geometry_consensus(
        per_agent_residuals, valid_mask
    )
    fused = torch.where(fallback[:, None], uniform_fused, quality_fused)
    fused = torch.where(any_valid[:, None], fused, torch.zeros_like(fused))
    return fused, any_valid, weights, fallback


def normalized_agent_object_distance(scene, proposals, geometry):
    """Return ``[P,A]`` object-agent distance normalized by BEV diagonal."""
    positions = scene.get("agent_positions")
    if not torch.is_tensor(positions) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("quality-aware scene requires agent_positions [A,2]")
    if positions.device != proposals.device:
        positions = positions.to(device=proposals.device, dtype=proposals.dtype)
    else:
        positions = positions.to(dtype=proposals.dtype)
    diagonal = (
        (geometry.x_max - geometry.x_min) ** 2
        + (geometry.y_max - geometry.y_min) ** 2
    ) ** 0.5
    delta = proposals[:, None, :2] - positions[None, :, :]
    return torch.linalg.vector_norm(delta, dim=-1).div(diagonal).clamp(0.0, 1.0)



@torch.no_grad()
def refine_doma_detections(model, pred_box_tensor, pred_score, context):
    """Refine post-NMS HEAL boxes while preserving score values and ordering."""
    if (
        not getattr(model, "doma_enabled", False)
        or not model.doma_flags["refiner"]
    ):
        return pred_box_tensor, pred_score
    if model.doma_config["mode"] != "inference":
        raise RuntimeError("DOMA detection refinement requires mode=inference")
    if context is None or len(context.get("scenes", ())) != 1:
        raise RuntimeError("DOMA inference requires one same-forward scene context")
    if (pred_box_tensor is None) != (pred_score is None):
        raise ValueError("prediction boxes and scores must both be tensors or both be None")
    if pred_box_tensor is None:
        return pred_box_tensor, pred_score
    if pred_box_tensor.shape[0] != pred_score.shape[0]:
        raise ValueError("prediction box and score counts must match")
    if pred_box_tensor.shape[0] == 0:
        return pred_box_tensor, pred_score

    center_boxes = corners_3d_to_boxes_hwl(pred_box_tensor)
    max_count = model.doma_config["object_space"]["roi"]["max_infer_proposals"]
    refine_count = min(int(center_boxes.shape[0]), max_count)
    top_indices = torch.topk(pred_score, k=refine_count, sorted=False).indices
    selected = center_boxes.index_select(0, top_indices).detach()
    result = predict_scene_residuals(model, context["scenes"][0], selected)
    refined_corners = boxes_hwl_to_corners_3d(result["refined_boxes"])
    output_boxes = pred_box_tensor.clone()
    valid_top_indices = top_indices[result["any_valid"]]
    if valid_top_indices.numel():
        output_boxes[valid_top_indices] = refined_corners[result["any_valid"]]

    return output_boxes, pred_score


def infer_common_bev_channels(args, modality_names):
    """Derive the real post-aligner Common-BEV channel interface from YAML."""
    if not modality_names:
        raise ValueError("DOMA model must define at least one modality")
    channels = []
    for modality in modality_names:
        backbone = args[modality]["backbone_args"]
        upsample = backbone.get("num_upsample_filter", [])
        if upsample:
            value = sum(int(item) for item in upsample)
        else:
            filters = backbone.get("num_filters")
            if not filters:
                raise ValueError("%s backbone_args has no output channel contract" % modality)
            value = int(filters[-1])
        channels.append((modality, value))
    unique = {value for _, value in channels}
    if len(unique) != 1:
        raise ValueError("DOMA Common-BEV channels disagree: %r" % channels)
    return channels[0][1]


def infer_context_bev_channels(args):
    """Return the channel contract of pre-fusion pyramid level 1."""
    fusion = args.get("fusion_backbone")
    if not isinstance(fusion, dict):
        raise TypeError("multi-scale DOMA requires fusion_backbone mapping")
    filters = fusion.get("num_filters")
    if not isinstance(filters, (list, tuple)) or len(filters) < 2:
        raise ValueError(
            "multi-scale DOMA requires fusion_backbone.num_filters level 1"
        )
    value = filters[1]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("fusion_backbone.num_filters[1] must be positive integer")
    return value


def _warp_scene_features(features, transforms, mode):
    agent_count, channels, height, width = features.shape
    grid = F.affine_grid(
        transforms,
        (agent_count, channels, height, width),
        align_corners=False,
    )
    return F.grid_sample(
        features,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def _agent_positions_in_ego(ego_to_agent, reference):
    if not torch.is_tensor(ego_to_agent) or ego_to_agent.ndim != 3:
        raise ValueError("ego-to-agent transforms must have shape [A,4,4]")
    if tuple(ego_to_agent.shape[1:]) != (4, 4):
        raise ValueError("ego-to-agent transforms must have shape [A,4,4]")
    transforms = ego_to_agent.to(device=reference.device, dtype=reference.dtype)
    # pairwise[ego, agent] maps ego coordinates to agent coordinates, exactly
    # the output-to-input direction used by affine_grid.  Its inverse maps the
    # agent origin into ego coordinates.
    agent_to_ego = torch.linalg.inv(transforms)
    return agent_to_ego[:, :2, 3]


def _quality_scalar(value, reference, name):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError("%s must have shape [M]" % name)
    if value.shape[0] != reference.shape[0]:
        raise ValueError("%s count must match object embeddings" % name)
    return value.to(device=reference.device, dtype=reference.dtype).unsqueeze(-1)


def _build_source_support(model, features, modalities):
    support = features.new_ones((features.shape[0], 1, features.shape[2], features.shape[3]))
    height, width = int(features.shape[2]), int(features.shape[3])
    edge_margin = 4 if not model.training else 0
    for index, modality in enumerate(modalities):
        if model.sensor_type_dict.get(modality) != "camera":
            continue
        ratio_h = float(getattr(model, "crop_ratio_H_%s" % modality))
        ratio_w = float(getattr(model, "crop_ratio_W_%s" % modality))
        valid_h = min(height, max(0, int(round(height / ratio_h)) - edge_margin))
        valid_w = min(width, max(0, int(round(width / ratio_w)) - edge_margin))
        start_h = (height - valid_h) // 2
        start_w = (width - valid_w) // 2
        support[index].zero_()
        support[index, :, start_h:start_h + valid_h, start_w:start_w + valid_w] = 1
    return support


def _set_module_trainability(module, trainable, parent_training):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
    module.train(bool(trainable and parent_training))



def _print_doma_summary(model, shared_trainable, active_modality):
    config = model.doma_config
    named_parameters = tuple(model.named_parameters())
    doma_parameters = [
        parameter for name, parameter in named_parameters if name.startswith("doma_")
    ]
    total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
    all_trainable = sum(
        parameter.numel()
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    doma_trainable = sum(
        parameter.numel()
        for parameter in doma_parameters
        if parameter.requires_grad
    )
    print("[DOMA]")
    print("version=%s" % config["version"])
    print("mode=%s" % config["mode"])
    print("modules=%s" % ", ".join(
        key for key, enabled in model.doma_flags.items() if enabled
    ))
    print("shared_trainable=%s" % shared_trainable)
    if active_modality is not None:
        print("active_modality=%s" % active_modality)
    print("doma parameters=%d" % sum(p.numel() for p in doma_parameters))
    print("doma trainable parameters=%d" % doma_trainable)
    print("base parameters=%d" % (
        total_parameters - sum(p.numel() for p in doma_parameters)
    ))
    print("total trainable parameters=%d" % all_trainable)


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
