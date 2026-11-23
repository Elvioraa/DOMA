# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import hashlib
import numpy as np
import torch
import torch.distributions as dist

def add_eval_pose_noise(data_dict, xy_std, yaw_std, seed, sample_idx):
    """Perturb reported poses in metres/degrees, retaining clean label poses."""
    for cav_id, content in data_dict.items():
        # Some base datasets share params with an in-memory scene cache.
        params = content['params'].copy()
        data_dict[cav_id] = dict(content, params=params)
        clean = params['lidar_pose']
        params['lidar_pose_clean'] = clean
        if content['ego'] or (xy_std == 0 and yaw_std == 0):
            continue
        # A stable per-agent stream, independent of Python's hash seed and workers.
        key = f'{seed}/{sample_idx}/{cav_id}'.encode('utf-8')
        rng_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')
        rng = np.random.RandomState(rng_seed)
        noise = np.zeros(6)
        noise[[0, 1, 4]] = rng.standard_normal(3) * [xy_std, xy_std, yaw_std]
        params['lidar_pose'] = np.asarray(clean) + noise
    return data_dict

def add_noise_data_dict(data_dict, noise_setting, sample_idx=None):
    """ Update the base data dict. 
        We retrieve lidar_pose and add_noise to it.
        And set a clean pose.
    """
    if 'eval_pose_noise' in noise_setting:
        return add_eval_pose_noise(data_dict, *noise_setting['eval_pose_noise'], sample_idx)
    if noise_setting['add_noise']:
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

            if "laplace" in noise_setting['args'].keys() and noise_setting['args']['laplace'] is True:
                cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                        generate_noise_laplace( # we just use the same key name
                                                            noise_setting['args']['pos_std'],
                                                            noise_setting['args']['rot_std'],
                                                            noise_setting['args']['pos_mean'],
                                                            noise_setting['args']['rot_mean']
                                                        )
            else:
                cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                            generate_noise(
                                                                noise_setting['args']['pos_std'],
                                                                noise_setting['args']['rot_std'],
                                                                noise_setting['args']['pos_mean'],
                                                                noise_setting['args']['rot_mean']
                                                            )

    else:
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

            
    return data_dict

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.
    
    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    
    return pose_noise



def generate_noise_laplace(pos_b, rot_b, pos_mu=0, rot_mu=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use laplace distribution to generate noise.
    
    Args:

        pos_b : float 
            parameter b of laplace dist, in meter

        rot_b : float
            parameter b of laplace dist, in degree

        pos_mu : float
            mean of laplace dist, in meter

        rot_mu : float
            mean of laplace dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.laplace(pos_mu, pos_b, size=(2))
    yaw = np.random.laplace(rot_mu, rot_b, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])
    return pose_noise


def generate_noise_torch(pose, pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ only used for v2vnet robust.
        rotation noise is sampled from von_mises distribution
    
    Args:
        pose : Tensor, [N. 6]
            including [x, y, z, roll, yaw, pitch]

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noisy: Tensor, [N, 6]
            noisy pose
    """

    N = pose.shape[0]
    noise = torch.zeros_like(pose, device=pose.device)
    concentration = (180 / (np.pi * rot_std)) ** 2

    noise[:, :2] = torch.normal(pos_mean, pos_std, size=(N, 2), device=pose.device)
    noise[:, 4] = dist.von_mises.VonMises(loc=rot_mean, concentration=concentration).sample((N,)).to(noise.device)


    return noise


def remove_z_axis(T):
    """ remove rotation/translation related to z-axis
    Args:
        T: np.ndarray
            [4, 4]
    Returns:
        T: np.ndarray
            [4, 4]
    """
    T[2,3] = 0 # z-trans
    T[0,2] = 0
    T[1,2] = 0
    T[2,0] = 0
    T[2,1] = 0
    T[2,2] = 1
    
    return T