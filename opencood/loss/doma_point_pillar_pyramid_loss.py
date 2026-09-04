"""DOMA-specific extension of the byte-identical Official HEAL loss."""

from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.loss.doma_object_loss import compute_doma_object_loss


class DOMAPointPillarPyramidLoss(PointPillarPyramidLoss):
    """Add object-space supervision after the Official HEAL loss path."""

    def forward(self, output_dict, target_dict, suffix=""):
        total_loss = super().forward(output_dict, target_dict, suffix)
        if suffix == "" and "doma_object" in output_dict:
            object_loss, object_stats = compute_doma_object_loss(
                output_dict["doma_object"]
            )
            total_loss = total_loss + object_loss
            self.loss_dict.update(object_stats)
            self.loss_dict["total_loss"] = total_loss.item()
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        super().logging(epoch, batch_id, batch_len, writer, suffix)
        if suffix != "" or not self.loss_dict.get("doma_enabled", False):
            return
        print(
            "[DOMA] Object: %.3e || Individual: %.3e || Consensus: %.3e"
            " || Valid: %.3f || Coverage: %.3f"
            % (
                self.loss_dict.get("doma_object_loss", 0.0),
                self.loss_dict.get("doma_individual_loss", 0.0),
                self.loss_dict.get("doma_consensus_loss", 0.0),
                self.loss_dict.get("doma_valid_object_ratio", 0.0),
                self.loss_dict.get("doma_mean_roi_coverage", 0.0),
            )
        )
        if "doma_protocol_loss" in self.loss_dict:
            print(
                "[DOMA OPA consistency proxy] Loss: %.3e"
                " || Weighted: %.3e || Branch-pairs: %d"
                % (
                    self.loss_dict["doma_protocol_loss"],
                    self.loss_dict["doma_protocol_weighted_loss"],
                    self.loss_dict["doma_protocol_pair_count"],
                )
            )
        if "doma_qar_loss" in self.loss_dict:
            print(
                "[DOMA QAR] Quality: %.3e || Weighted: %.3e"
                " || Pred: %.3f || Target: %.3f"
                % (
                    self.loss_dict["doma_qar_loss"],
                    self.loss_dict["doma_qar_weighted_loss"],
                    self.loss_dict["doma_qar_pred_mean"],
                    self.loss_dict["doma_qar_target_mean"],
                )
            )
        if "doma_delta_iou_count" in self.loss_dict:
            print(
                "[DOMA Delta-IoU diagnostic] N: %d || Finite: %d"
                " || Mean/Std: %.3f/%.3f || P10/P50/P90: %.3f/%.3f/%.3f"
                " || Improve/Zero/Worsen: %.3f/%.3f/%.3f"
                % (
                    self.loss_dict["doma_delta_iou_count"],
                    self.loss_dict["doma_delta_iou_finite_count"],
                    self.loss_dict["doma_delta_iou_mean"],
                    self.loss_dict["doma_delta_iou_std"],
                    self.loss_dict["doma_delta_iou_p10"],
                    self.loss_dict["doma_delta_iou_p50"],
                    self.loss_dict["doma_delta_iou_p90"],
                    self.loss_dict["doma_delta_iou_improve_ratio"],
                    self.loss_dict["doma_delta_iou_zero_ratio"],
                    self.loss_dict["doma_delta_iou_worsen_ratio"],
                )
            )
        if writer is not None:
            step = epoch * batch_len + batch_id
            for name in (
                "doma_object_loss",
                "doma_individual_loss",
                "doma_consensus_loss",
                "doma_quality_loss",
                "doma_valid_object_ratio",
                "doma_mean_roi_coverage",
                "doma_protocol_loss",
                "doma_protocol_weighted_loss",
                "doma_protocol_pair_count",
                "doma_qar_loss",
                "doma_qar_weighted_loss",
                "doma_qar_pred_mean",
                "doma_qar_target_mean",
                "doma_delta_iou_count",
                "doma_delta_iou_finite_count",
                "doma_delta_iou_nonfinite_count",
                "doma_delta_iou_mean",
                "doma_delta_iou_std",
                "doma_delta_iou_min",
                "doma_delta_iou_p10",
                "doma_delta_iou_p25",
                "doma_delta_iou_p50",
                "doma_delta_iou_p75",
                "doma_delta_iou_p90",
                "doma_delta_iou_max",
                "doma_delta_iou_improve_ratio",
                "doma_delta_iou_zero_ratio",
                "doma_delta_iou_worsen_ratio",
                "doma_auxiliary_loss",
                "doma_total_loss",
            ):
                if name in self.loss_dict:
                    writer.add_scalar(name, self.loss_dict[name], step)
