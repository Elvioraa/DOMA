# -*- coding: utf-8 -*-
"""
Diagnose gradient conflict between the Official HEAL loss and the DOMA
object-space loss during Stage2 modality adaptation.

This script is diagnostic only:
- loads an already-trained Stage2 checkpoint;
- performs forward/backward-style autograd queries;
- NEVER calls optimizer.step();
- NEVER saves or modifies model checkpoints.
"""

import argparse
import csv
import math
import os
import random
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.loss.doma_object_loss import compute_doma_object_loss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools import train_utils


TARGET_MODULES = (
    "encoder_m3",
    "backbone_m3",
    "aligner_m3",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DOMA Stage2 m3 HEAL-vs-object gradient conflict diagnostic"
    )
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fusion_method", "-f", default="intermediate")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Default: <model_dir>/m3_grad_conflict.csv",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_target_parameters(model):
    selected = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in TARGET_MODULES
        ):
            selected.append((name, param))

    if not selected:
        raise RuntimeError(
            "No trainable parameters found in encoder_m3/backbone_m3/aligner_m3"
        )

    return selected


def module_of_parameter(name):
    for prefix in TARGET_MODULES:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    raise ValueError("Unexpected parameter name: %s" % name)


def gradient_metrics(param_items, heal_grads, object_grads, module=None):
    dot = None
    heal_sq = None
    object_sq = None

    common_tensors = 0
    common_numel = 0
    total_numel = 0

    for (name, param), g_heal, g_object in zip(
        param_items, heal_grads, object_grads
    ):
        if module is not None and module_of_parameter(name) != module:
            continue

        total_numel += param.numel()

        if g_heal is None or g_object is None:
            continue

        gh = g_heal.detach().float()
        go = g_object.detach().float()

        local_dot = (gh * go).sum()
        local_heal_sq = (gh * gh).sum()
        local_object_sq = (go * go).sum()

        dot = local_dot if dot is None else dot + local_dot
        heal_sq = local_heal_sq if heal_sq is None else heal_sq + local_heal_sq
        object_sq = (
            local_object_sq
            if object_sq is None
            else object_sq + local_object_sq
        )

        common_tensors += 1
        common_numel += param.numel()

    if dot is None:
        return {
            "cosine": math.nan,
            "heal_grad_norm": 0.0,
            "object_grad_norm": 0.0,
            "object_to_heal_norm_ratio": math.nan,
            "common_tensors": 0,
            "common_numel": 0,
            "total_numel": total_numel,
            "common_numel_ratio": 0.0,
        }

    heal_norm = torch.sqrt(heal_sq)
    object_norm = torch.sqrt(object_sq)

    heal_value = float(heal_norm.item())
    object_value = float(object_norm.item())

    if heal_value == 0.0 or object_value == 0.0:
        cosine = math.nan
    else:
        cosine = float(
            (dot / (heal_norm * object_norm + 1e-12)).item()
        )

    ratio = (
        object_value / heal_value
        if heal_value > 0.0
        else math.nan
    )

    common_ratio = (
        common_numel / total_numel
        if total_numel > 0
        else 0.0
    )

    return {
        "cosine": cosine,
        "heal_grad_norm": heal_value,
        "object_grad_norm": object_value,
        "object_to_heal_norm_ratio": ratio,
        "common_tensors": common_tensors,
        "common_numel": common_numel,
        "total_numel": total_numel,
        "common_numel_ratio": common_ratio,
    }


def finite(values):
    return [x for x in values if math.isfinite(x)]


def print_summary(rows, scope):
    scoped = [r for r in rows if r["scope"] == scope]

    cosines = finite([r["cosine"] for r in scoped])
    heal_norms = finite([r["heal_grad_norm"] for r in scoped])
    object_norms = finite([r["object_grad_norm"] for r in scoped])
    ratios = finite([r["object_to_heal_norm_ratio"] for r in scoped])

    print("\n===== %s =====" % scope)

    if not cosines:
        print("No valid non-zero overlapping gradients.")
        return

    negative_fraction = sum(x < 0.0 for x in cosines) / len(cosines)

    print("valid batches        : %d" % len(cosines))
    print("cosine mean          : %.6f" % statistics.mean(cosines))
    print("cosine median        : %.6f" % statistics.median(cosines))
    print("cosine min           : %.6f" % min(cosines))
    print("cosine max           : %.6f" % max(cosines))
    print("negative fraction    : %.3f" % negative_fraction)

    if heal_norms:
        print("HEAL grad norm mean  : %.6e" % statistics.mean(heal_norms))

    if object_norms:
        print("OBJ grad norm mean   : %.6e" % statistics.mean(object_norms))

    if ratios:
        print("OBJ/HEAL norm mean   : %.6f" % statistics.mean(ratios))


def main():
    opt = parse_args()
    seed_everything(opt.seed)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    mode = hypes.get("model", {}).get("args", {}).get("doma", {}).get("mode")
    if mode is not None and mode != "stage2_adapt":
        raise ValueError(
            "Expected DOMA stage2_adapt config, got mode=%r" % mode
        )

    active_modality = (
        hypes.get("model", {})
        .get("args", {})
        .get("doma", {})
        .get("active_modality")
    )
    if active_modality is not None and active_modality != "m3":
        raise ValueError(
            "This diagnostic expects active_modality=m3, got %r"
            % active_modality
        )

    print("Dataset Building")
    train_dataset = build_dataset(hypes, visualize=False, train=True)

    if getattr(train_dataset, "supervise_single", False):
        raise RuntimeError(
            "This diagnostic assumes Stage2 m3 has no single supervision."
        )

    generator = torch.Generator()
    generator.manual_seed(opt.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=4,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        generator=generator,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)

    print("Loading trained Stage2 checkpoint")
    loaded_epoch, model = train_utils.load_saved_model(
        opt.model_dir,
        model,
    )
    print("Loaded checkpoint epoch:", loaded_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Match the real Stage2 training state.
    model.train()
    try:
        model.model_train_init()
    except AttributeError:
        print("No model_train_init function")

    param_items = collect_target_parameters(model)
    target_params = [p for _, p in param_items]

    print("\nTarget trainable modules:")
    for module in TARGET_MODULES:
        numel = sum(
            p.numel()
            for name, p in param_items
            if module_of_parameter(name) == module
        )
        print("  %-12s %d parameters" % (module, numel))

    # Official HEAL criterion only. This deliberately excludes DOMA object loss.
    heal_criterion = PointPillarPyramidLoss(hypes["loss"]["args"])

    rows = []
    valid_batches = 0

    for loader_i, batch_data in enumerate(train_loader):
        if valid_batches >= opt.num_batches:
            break

        if (
            batch_data is None
            or batch_data["ego"]["object_bbx_mask"].sum() == 0
        ):
            continue

        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["epoch"] = loaded_epoch

        # No optimizer and no optimizer.step() anywhere in this script.
        output_dict = model(batch_data["ego"])

        heal_loss = heal_criterion(
            output_dict,
            batch_data["ego"]["label_dict"],
        )

        if "doma_object" not in output_dict:
            raise RuntimeError(
                "output_dict does not contain 'doma_object'"
            )

        object_loss, object_stats = compute_doma_object_loss(
            output_dict["doma_object"]
        )

        heal_grads = torch.autograd.grad(
            heal_loss,
            target_params,
            retain_graph=True,
            allow_unused=True,
        )

        object_grads = torch.autograd.grad(
            object_loss,
            target_params,
            retain_graph=False,
            allow_unused=True,
        )

        batch_id = valid_batches + 1

        for scope in ("ALL",) + TARGET_MODULES:
            module = None if scope == "ALL" else scope

            metrics = gradient_metrics(
                param_items,
                heal_grads,
                object_grads,
                module=module,
            )

            row = {
                "batch": batch_id,
                "scope": scope,
                "heal_loss": float(heal_loss.detach().item()),
                "object_loss": float(object_loss.detach().item()),
                "cosine": metrics["cosine"],
                "heal_grad_norm": metrics["heal_grad_norm"],
                "object_grad_norm": metrics["object_grad_norm"],
                "object_to_heal_norm_ratio": metrics[
                    "object_to_heal_norm_ratio"
                ],
                "common_tensors": metrics["common_tensors"],
                "common_numel": metrics["common_numel"],
                "total_numel": metrics["total_numel"],
                "common_numel_ratio": metrics["common_numel_ratio"],
                "valid_object_ratio": float(
                    object_stats.get("doma_valid_object_ratio", 0.0)
                ),
                "mean_roi_coverage": float(
                    object_stats.get("doma_mean_roi_coverage", 0.0)
                ),
            }
            rows.append(row)

        all_row = rows[-4]

        print(
            "[%02d/%02d] "
            "L_HEAL=%.6f "
            "L_OBJ=%.6f "
            "cos=%.6f "
            "|gH|=%.3e "
            "|gO|=%.3e "
            "OBJ/HEAL=%.3f "
            "common=%.3f"
            % (
                batch_id,
                opt.num_batches,
                all_row["heal_loss"],
                all_row["object_loss"],
                all_row["cosine"],
                all_row["heal_grad_norm"],
                all_row["object_grad_norm"],
                all_row["object_to_heal_norm_ratio"],
                all_row["common_numel_ratio"],
            )
        )

        valid_batches += 1

    if valid_batches == 0:
        raise RuntimeError("No valid training batches were diagnosed.")

    output_csv = opt.output_csv
    if output_csv is None:
        output_csv = os.path.join(
            opt.model_dir,
            "m3_grad_conflict.csv",
        )

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for scope in ("ALL",) + TARGET_MODULES:
        print_summary(rows, scope)

    print("\nSaved diagnostic CSV:")
    print(output_csv)
    print("\nDiagnostic complete: no optimizer.step() was executed.")


if __name__ == "__main__":
    main()
