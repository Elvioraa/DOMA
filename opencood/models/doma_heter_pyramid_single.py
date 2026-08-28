""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.fuse_modules.doma_pyramid_fuse import DOMAPyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.models.sub_modules.doma_object import (
    attach_single_doma_pyramid_context,
    build_single_doma_context,
    configure_doma_trainability,
    install_doma_modules,
    run_doma_training,
)

class DOMAHeterPyramidSingle(nn.Module):
    def __init__(self, args):
        super(DOMAHeterPyramidSingle, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if args.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

        """
        Would load from pretrain base.
        """
        self.pyramid_backbone = DOMAPyramidFusion(args['fusion_backbone'])
        """
        Shrink header, Would load from pretrain base.
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.fix_modules.append('shrink_conv')

        """
        Shared Heads, Would load from pretrain base.
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        install_doma_modules(self, args)
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)
        configure_doma_trainability(self)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        if getattr(self, 'doma_enabled', False):
            configure_doma_trainability(self)
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop(
                    (int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}")))
                )(feature)

            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })

        doma_context = None
        if getattr(self, 'doma_enabled', False):
            active_modality = self.doma_config.get('active_modality')
            if active_modality is not None and active_modality != modality_name:
                raise RuntimeError(
                    'DOMA active_modality=%s but input key routes %s'
                    % (active_modality, modality_name)
                )
            doma_context = build_single_doma_context(
                self, feature, modality_name
            )

        # multiscale fusion.
        capture_pyramid = bool(
            getattr(self, 'doma_enabled', False)
            and self.doma_flags['context']
        )
        if capture_pyramid:
            feature, occ_map_list, pre_fusion_features = (
                self.pyramid_backbone.forward_single(
                    feature, return_pre_fusion_features=True
                )
            )
            attach_single_doma_pyramid_context(
                self,
                doma_context,
                pre_fusion_features,
                modality_name,
            )
        else:
            feature, occ_map_list = self.pyramid_backbone.forward_single(feature)

        if self.shrink_flag:
            feature = self.shrink_conv(feature)

        cls_preds = self.cls_head(feature)
        reg_preds = self.reg_head(feature)
        dir_preds = self.dir_head(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list': 
                            occ_map_list})
        if getattr(self, 'doma_enabled', False):
            doma_object = run_doma_training(
                self, doma_context, data_dict, detector_output=output_dict
            )
            if doma_object is not None:
                output_dict['doma_object'] = doma_object
            if self.doma_config['mode'] == 'inference':
                output_dict['doma_context'] = doma_context
        return output_dict

