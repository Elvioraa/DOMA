import argparse
import os
import statistics
import glob
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import seed_utils
from opencood.tools import ddp_safe_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    seed = seed_utils.seed_from_hypes(hypes)
    rank = opt.rank if opt.distributed else 0
    ddp_safe_enabled = ddp_safe_utils.ddp_safe_validation_enabled(hypes)
    ddp_safe_active = ddp_safe_enabled and opt.distributed

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_seed_kwargs = seed_utils.distributed_sampler_seed_kwargs(seed)
        sampler_train = DistributedSampler(opencood_train_dataset,
                                           **sampler_seed_kwargs)
        if ddp_safe_active:
            sampler_val = ddp_safe_utils.DistributedEvalSampler(
                opencood_validate_dataset,
                num_replicas=dist.get_world_size(),
                rank=rank)
        else:
            sampler_val = DistributedSampler(opencood_validate_dataset,
                                             shuffle=False,
                                             **sampler_seed_kwargs)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  **seed_utils.dataloader_seed_kwargs(seed, rank))
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False,
                                **seed_utils.dataloader_seed_kwargs(seed, rank))
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  **seed_utils.dataloader_seed_kwargs(seed, rank))
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                **seed_utils.dataloader_seed_kwargs(seed, rank))

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ddp_safe_active:
        print('rank=%d local_rank=%d device=cuda:%d' %
              (rank, opt.gpu, opt.gpu), force=True)
        if rank == 0:
            per_gpu_batch = hypes['train_params']['batch_size']
            print('[DDP Training]')
            print('world_size=%d' % dist.get_world_size())
            print('per_gpu_batch=%d' % per_gpu_batch)
            print('global_batch=%d' %
                  (per_gpu_batch * dist.get_world_size()))
            print('train_dataset_size=%d' % len(opencood_train_dataset))
            print('train_batches_per_rank=%d' % len(train_loader))
            print('validation_dataset_size=%d' %
                  len(opencood_validate_dataset))
            validation_batches_per_rank = [
                len(ddp_safe_utils.DistributedEvalSampler(
                    opencood_validate_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=validation_rank))
                for validation_rank in range(dist.get_world_size())
            ]
            print('validation_batches_per_rank=%s' %
                  validation_batches_per_rank)
            print('checkpoint_writer=rank0')
            print('validation_reduction=global_sum_count')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        if ddp_safe_active:
            saved_path_holder = [None]
            if rank == 0:
                saved_path_holder[0] = train_utils.setup_train(hypes)
            dist.broadcast_object_list(saved_path_holder, src=0)
            saved_path = saved_path_holder[0]
        else:
            saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module

    # Safe validation uses the underlying module so ranks may skip None batches
    # independently without mismatched DDP forward-time collectives.
    validation_model = model_without_ddp if ddp_safe_active else model


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            local_batch_valid = batch_data is not None
            if local_batch_valid:
                local_batch_valid = (
                    batch_data['ego']['object_bbx_mask'].sum().item() != 0)
            if opt.distributed:
                train_batch_valid = \
                    ddp_safe_utils.all_ranks_have_valid_training_batch(
                        local_batch_valid, device)
            else:
                train_batch_valid = local_batch_valid
            if not train_batch_valid:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()


        # torch.cuda.empty_cache() # it will destroy memory buffer
        if epoch % hypes['train_params']['save_freq'] == 0:
            if ddp_safe_active:
                if ddp_safe_utils.checkpoint_writer_enabled(
                        ddp_safe_enabled, opt.distributed, rank):
                    torch.save(model_without_ddp.state_dict(),
                               os.path.join(
                                   saved_path,
                                   'net_epoch%d.pth' % (epoch + 1)))
                dist.barrier()
            else:
                torch.save(model_without_ddp.state_dict(),
                           os.path.join(saved_path,
                                        'net_epoch%d.pth' % (epoch + 1)))
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            local_loss_sum = 0.0
            local_count = 0

            if ddp_safe_active:
                model.eval()
                ddp_safe_utils.sync_module_buffers_from_rank0(
                    model_without_ddp)

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    if not ddp_safe_active:
                        model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = validation_model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                    if ddp_safe_active:
                        local_loss_sum += final_loss.item()
                        local_count += 1

            if ddp_safe_active:
                valid_ave_loss, _, _ = \
                    ddp_safe_utils.reduce_validation_sum_count(
                        local_loss_sum, local_count, device)
            else:
                valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if ddp_safe_active:
                is_new_best = torch.tensor(
                    [rank == 0 and valid_ave_loss < lowest_val_loss],
                    dtype=torch.bool,
                    device=device)
                dist.broadcast(is_new_best, src=0)
                if is_new_best.item():
                    previous_lowest_val_epoch = lowest_val_epoch
                    lowest_val_loss = valid_ave_loss
                    lowest_val_epoch = epoch + 1
                    if ddp_safe_utils.checkpoint_writer_enabled(
                            ddp_safe_enabled, opt.distributed, rank):
                        torch.save(
                            model_without_ddp.state_dict(),
                            os.path.join(
                                saved_path,
                                'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                        previous_bestval_path = os.path.join(
                            saved_path,
                            'net_epoch_bestval_at%d.pth' %
                            previous_lowest_val_epoch)
                        if (previous_lowest_val_epoch != -1 and
                                os.path.exists(previous_bestval_path)):
                            os.remove(previous_bestval_path)
                    dist.barrier()
            else:
                if valid_ave_loss < lowest_val_loss:
                    lowest_val_loss = valid_ave_loss
                    torch.save(model_without_ddp.state_dict(),
                           os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                    if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                        if opt.rank == 0:
                            os.remove(os.path.join(saved_path,
                                            'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                    lowest_val_epoch = epoch + 1

        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    if ddp_safe_active:
        dist.barrier()
    print('Training Finished, checkpoints saved to %s' % saved_path)

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            import numpy as np
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        if run_test:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()
