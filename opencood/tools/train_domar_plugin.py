"""Fit shared DOMA-R heads on frozen Stage 1 m1 detections."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml import yaml_utils
from opencood.models.sub_modules.doma_box_coder import corners_3d_to_boxes_hwl
from opencood.models.sub_modules.doma_object import predict_scene_residuals
from opencood.models.sub_modules.domar_plugins import DOMARPlugins, freeze_backbone, plugin_loss


class CollateScenes:
    """Keep full-resolution frozen BEV activations on GPU for just one frame."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, samples):
        return [self.dataset.collate_batch_train([sample])['ego']
                for sample in samples if sample is not None]


@torch.no_grad()
def frozen_candidates(model, ego, processor):
    if set(ego['agent_modality_list']) != {'m1'}:
        raise ValueError('Stage 1.5 expects m1 views exclusively')
    output = model(ego)
    post_input = {'anchor_box': ego['anchor_box'],
                  'transformation_matrix': torch.eye(4, device=ego['anchor_box'].device)}
    corners, scores = processor.post_process({'ego': post_input}, {'ego': output})
    if scores is None or scores.numel() == 0:
        return None
    indices = scores.topk(min(64, len(scores))).indices
    proposals = corners_3d_to_boxes_hwl(corners[indices])
    scene = output['doma_context']['scenes'][0]
    features = predict_scene_residuals(model, scene, proposals, collect_domar=True)
    gt = ego['object_bbx_center'][0][ego['object_bbx_mask'][0].bool()]
    return features['domar_inputs'], proposals, scores[indices], gt


def run_epoch(model, plugins, loader, optimizer, lambda_iou, device):
    from opencood.tools import train_utils

    training = optimizer is not None
    plugins.train(training)
    yaw = model.doma_config['refiner']['yaw_mode']
    count = 0
    loss_sum = 0.
    grid = torch.linspace(0, 1, 21, device=device)
    errors = torch.zeros_like(grid)
    with torch.set_grad_enabled(training):
        for frames in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            used = 0
            for frame in frames:
                data = frozen_candidates(model, train_utils.to_device(frame, device),
                                         loader.dataset.post_processor)
                if data is None:
                    continue
                inputs, boxes, scores, gt = data
                residual, q, _ = plugins(inputs)
                valid = inputs[-1].any(dim=1)
                loss, target = plugin_loss(boxes, gt, residual, q, valid, yaw, lambda_iou)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite DOMA-R loss')
                if training:
                    loss.backward()
                elif valid.any():
                    # Fit only on boxes whose confidence will actually be calibrated.
                    s = scores[valid, None].clamp_min(1e-12)
                    quality = q[valid, None].clamp_min(1e-12)
                    blended = (s.log() * (1 - grid) + quality.log() * grid).exp()
                    errors += (blended - target[valid, None]).square().sum(0)
                loss_sum += loss.item()
                count += 1
                used += 1
            if training and used:
                for param in plugins.parameters():
                    if param.grad is not None:
                        param.grad.div_(used)
                optimizer.step()
    if not count:
        raise RuntimeError('No post-NMS proposals in this split')
    if not training:
        plugins.calibration.gamma.copy_(grid[errors.argmin()])
    return loss_sum / count


def main():
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import seed_utils, train_utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True, help='m1 Stage 1 directory with config.yaml')
    parser.add_argument('--checkpoint', required=True, help='Stage 1 weights, relative to model_dir or absolute')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lambda_iou', type=float, default=1.)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=303)
    opt = parser.parse_args()
    if min(opt.epochs, opt.batch_size, opt.lr) <= 0 or opt.lambda_iou < 0:
        parser.error('epochs, batch size and lr must be positive and lambda_iou nonnegative')
    cfg = yaml_utils.load_yaml(None, opt)
    seed_utils.seed_everything(opt.seed)
    if (cfg['model']['core_method'] != 'doma_heter_pyramid_collab'
            or cfg['model']['args']['doma']['mode'] != 'stage1_anchor'
            or set(cfg['heter']['modality_setting']) != {'m1'}):
        raise ValueError('Expected a Stage 1 collaborative m1 checkpoint configuration')
    device = torch.device('cuda:0')
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(Path(opt.model_dir) / opt.checkpoint, map_location='cpu'), strict=True)
    freeze_backbone(model)
    model.to(device)
    plugins = DOMARPlugins.from_model(model, cfg['train_params'].get('max_cav', 5)).to(device)
    train_cfg = deepcopy(cfg)
    train_cfg['noise_setting'] = {'add_noise': False, 'domar_training_noise': True}
    val_cfg = deepcopy(cfg)
    val_cfg['noise_setting'] = {'add_noise': False}
    loaders = []
    for config, training in ((train_cfg, True), (val_cfg, False)):
        ds = build_dataset(config, visualize=False, train=training)
        loaders.append(DataLoader(ds, batch_size=opt.batch_size, shuffle=training,
                                  collate_fn=CollateScenes(ds), num_workers=opt.workers,
                                  **seed_utils.dataloader_seed_kwargs(opt.seed)))
    out = Path(opt.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(plugins.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    best = float('inf')
    for epoch in range(opt.epochs):
        train_loss = run_epoch(model, plugins, loaders[0], optimizer, opt.lambda_iou, device)
        val_loss = run_epoch(model, plugins, loaders[1], None, opt.lambda_iou, device)
        scheduler.step()
        print(f'epoch={epoch+1} train={train_loss:.6f} val={val_loss:.6f} '
              f'gamma={plugins.calibration.gamma.item():.2f}', flush=True)
        for name in (['last', 'best'] if val_loss < best else ['last']):
            destination = out / f'domar_plugin_{name}.pth'
            temp = destination.with_suffix('.tmp')
            torch.save(plugins.state_dict(), temp)
            temp.replace(destination)
        best = min(best, val_loss)
        loaders[0].dataset.reinitialize()


if __name__ == '__main__':
    main()
