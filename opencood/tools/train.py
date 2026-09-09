# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import uuid

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import seed_utils, train_utils, validation_detection
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    seed = seed_utils.seed_from_hypes(hypes)
    checkpoint_selection = \
        validation_detection.get_checkpoint_selection_config(hypes)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    detection_val_loader = None
    if checkpoint_selection["enabled"]:
        validation_detection.validate_fusion_method(opt.fusion_method)
        detection_val_loader = \
            validation_detection.build_detection_validation_loader(
                opencood_validate_dataset, seed)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2,
                              **seed_utils.dataloader_seed_kwargs(seed))
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2,
                            **seed_utils.dataloader_seed_kwargs(seed))

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    best_detection_metric = -float("inf")
    best_detection_epoch = None
    detection_history = []
    detection_run_id = None
    if checkpoint_selection["enabled"]:
        (best_detection_metric,
         best_detection_epoch,
         detection_history,
         _) = validation_detection.restore_detection_selection_state(
             saved_path, checkpoint_selection)
        detection_run_id = uuid.uuid4().hex

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        if validation_detection.should_evaluate_detection(
                epoch, checkpoint_selection):
            completed_epoch = epoch + 1
            detection_metrics = validation_detection.evaluate_detection_ap(
                model,
                detection_val_loader,
                opencood_validate_dataset,
                device,
                fusion_method=opt.fusion_method,
            )
            selection_metric = checkpoint_selection["metric"]
            current_detection_metric = detection_metrics[selection_metric]
            (best_detection_metric,
             best_detection_epoch,
             detection_history,
             saved_bestdet_path,
             is_new_best) = \
                validation_detection.commit_detection_evaluation(
                    model,
                    saved_path,
                    checkpoint_selection,
                    detection_run_id,
                    completed_epoch,
                    detection_metrics,
                    best_detection_metric,
                    best_detection_epoch,
                    detection_history,
                )

            print("[Detection Validation]")
            print("Split: validation")
            print("Epoch: {}".format(completed_epoch))
            print("AP@0.3: {:.6f}".format(detection_metrics["ap30"]))
            print("AP@0.5: {:.6f}".format(detection_metrics["ap50"]))
            print("AP@0.7: {:.6f}".format(detection_metrics["ap70"]))
            print("Selection metric: {}".format(selection_metric))
            print("Current validation metric: {:.6f}".format(
                current_detection_metric))
            print("Best validation metric: {:.6f}".format(
                best_detection_metric))
            print("Best epoch: {}".format(best_detection_epoch))
            writer.add_scalar("Validation_Detection_AP30",
                              detection_metrics["ap30"], completed_epoch)
            writer.add_scalar("Validation_Detection_AP50",
                              detection_metrics["ap50"], completed_epoch)
            writer.add_scalar("Validation_Detection_AP70",
                              detection_metrics["ap70"], completed_epoch)

            print("[BestDet]")
            if is_new_best:
                print("New best validation AP@{:.1f}".format(
                    validation_detection.SUPPORTED_METRICS[selection_metric]))
                print("Epoch: {}".format(completed_epoch))
                print("{}: {:.6f}".format(
                    selection_metric.upper(), current_detection_metric))
                if saved_bestdet_path is not None:
                    print("Saved: {}".format(saved_bestdet_path))
                else:
                    print("Checkpoint saving disabled by save_bestdet=false")
            else:
                print("No update for validation {}".format(selection_metric))
                print("Current {}: {:.6f}".format(
                    selection_metric.upper(), current_detection_metric))
                print("Best {}: {:.6f}".format(
                    selection_metric.upper(), best_detection_metric))
                print("Best epoch: {}".format(best_detection_epoch))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True
    if run_test:
        if (checkpoint_selection["enabled"] and
                (not checkpoint_selection["save_bestdet"] or
                 best_detection_epoch is None)):
            print("Skipping post-training inference: enabled checkpoint "
                  "selection did not produce a saved bestdet checkpoint.")
        else:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)

if __name__ == '__main__':
    main()
