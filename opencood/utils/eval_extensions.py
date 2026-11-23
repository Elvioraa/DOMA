"""Optional pose-noise and distance-stratified detection evaluation."""
import argparse
import math

from opencood.utils import eval_utils


def _std(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative standard deviation")
    return value


def add_arguments(parser):
    parser.add_argument('--pose_noise_xy', type=_std, default=0.0,
                        help='non-ego x/y Gaussian standard deviation in metres')
    parser.add_argument('--pose_noise_yaw', type=_std, default=0.0,
                        help='non-ego yaw Gaussian standard deviation in degrees')
    parser.add_argument('--pose_noise_seed', type=int, default=303)
    parser.add_argument('--eval_distance_buckets', action='store_true',
                        help='print AP by ego-frame box-centre distance')


def configure_noise(hypes, opt):
    if opt.pose_noise_xy > 0 or opt.pose_noise_yaw > 0:
        # Explicit CLI noise replaces the legacy all-agent YAML noise for this run.
        hypes['noise_setting'] = {'add_noise': False, 'eval_pose_noise': (
            opt.pose_noise_xy, opt.pose_noise_yaw, opt.pose_noise_seed)}
        opt.note += f'_noise_xy{opt.pose_noise_xy}_yaw{opt.pose_noise_yaw}_seed{opt.pose_noise_seed}'


class DistanceBucketAP:
    ranges = ((0, 30), (30, 50), (50, float('inf')))

    def __init__(self):
        self.stats = {
            bounds: {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                     for iou in (0.5, 0.7)}
            for bounds in self.ranges
        }

    def update(self, boxes, scores, gt):
        # The postprocessor supplies corners in the ego frame (N, 8, 3).
        gt_distance = gt.mean(dim=1).norm(dim=1)
        if boxes is not None:
            pred_distance = boxes.mean(dim=1).norm(dim=1)
        for (low, high), stats in self.stats.items():
            bucket_gt = gt[(gt_distance >= low) & (gt_distance < high)]
            bucket_boxes = bucket_scores = None
            if boxes is not None:
                mask = (pred_distance >= low) & (pred_distance < high)
                bucket_boxes, bucket_scores = boxes[mask], scores[mask]
            for iou in stats:
                eval_utils.caluclate_tp_fp(bucket_boxes, bucket_scores, bucket_gt, stats, iou)

    def summary(self):
        return {
            bounds: {iou: eval_utils.calculate_ap(stats, iou)[0]
                     if values['gt'] else float('nan')
                     for iou, values in stats.items()}
            for bounds, stats in self.stats.items()
        }

    def print_summary(self):
        for (low, high), ap in self.summary().items():
            print(f'Distance [{low}, {high}) m  AP0.5={ap[0.5]:.4f}  AP0.7={ap[0.7]:.4f}')
