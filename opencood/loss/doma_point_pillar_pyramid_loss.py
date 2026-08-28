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
        if writer is not None:
            step = epoch * batch_len + batch_id
            for name in (
                "doma_object_loss",
                "doma_individual_loss",
                "doma_consensus_loss",
                "doma_quality_loss",
                "doma_valid_object_ratio",
                "doma_mean_roi_coverage",
            ):
                if name in self.loss_dict:
                    writer.add_scalar(name, self.loss_dict[name], step)
