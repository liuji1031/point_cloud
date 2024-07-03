import os
import numpy as np
from numpy.linalg import multi_dot
from copy import copy, deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from nuscenes.utils.geometry_utils import transform_matrix

def get_lidar_pts(nusc,sample):
    """retrieves raw lidar points from filename stored in sample data

    Args:
        sample (_type_): _description_
        data_root (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_info = nusc.get('sample_data', lidar_token)
    
    lidar_path = os.path.join(nusc.dataroot,lidar_info["filename"])
    lidar_data = LidarPointCloud.from_file(str(lidar_path))
    return lidar_data

def get_ego_pose(nusc,sample,sensor="LIDAR_TOP"):
    """retrieve ego pose from the current sample

    Args:
        sample (_type_): _description_
    """
    sensor_token = sample["data"][sensor]
    sample_data = nusc.get("sample_data",sensor_token)
    ego_pose = nusc.get("ego_pose",sample_data["ego_pose_token"])
    return ego_pose

def transform_matrix_from_pose(pose,inverse):
    """wrapper function for getting the transformation matrix from pose dict

    Args:
        pose (_type_): _description_
        inverse (_type_): _description_

    Returns:
        _type_: _description_
    """
    return transform_matrix(pose['translation'], 
                                Quaternion(pose['rotation']),
                                inverse=inverse)

def get_sensor_data(nusc,sample,sensor):
    return nusc.get("sample_data",sample["data"][sensor])

def inv_transform(T):
    """returns inverse transformation matrix

    Args:
        T (_type_): _description_

    Returns:
        _type_: _description_
    """
    R = T[:3,:3]
    t = T[:3,[3]]

    T_inv = np.eye(4)
    T_inv[:3,:3] = R.T
    T_inv[:3,[3]] = -multi_dot([R.T,t])
    return T_inv

def get_lidar_pts_multisweep(scene,
                              nusc : "NuScene",
                              sensor="LIDAR_TOP",
                              win_len=2,
                              min_dist=1.0,
                              nkeep=26000,
                              norm_factor=[100.0,100.0,10.0]):
    """returns the lidar points 

    Args:
        scene (_type_): _description_
        nusc (NuScene): _description_
        sensor (str, optional): _description_. Defaults to "LIDAR_TOP".
        win_len (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    sample_token = scene["first_sample_token"]
    data_out = []
    sample_tokens = []
    pc_win = []
    norm_factor = np.array([*norm_factor,1.0])[np.newaxis,:]
    car_ref_global_prev = None

    while sample_token!="":
        sample = nusc.get("sample",sample_token)
        sample_data = get_sensor_data(nusc,sample,sensor)

        # get ego pose and transformation
        ego_pose = get_ego_pose(nusc,sample,sensor)
        car_ref_global = transform_matrix_from_pose(ego_pose,inverse=False)

        # get sensor ref car transformation
        sensor_ref_car = transform_matrix_from_pose(
                            nusc.get("calibrated_sensor",
                                    sample_data["calibrated_sensor_token"]),
                            inverse=False
                        )
        
        pc = get_lidar_pts(nusc,sample)
        pc.remove_close(min_dist)
        # convert sensor points to car body frame
        pc.transform(sensor_ref_car)
        # plt.hist(pc.points[2,:],50)
        # plt.show()
        if car_ref_global_prev is None:
            car_ref_global_prev = car_ref_global
            pc_win.append(pc)

        else:
            transform_between = multi_dot([inv_transform(car_ref_global),
                                           car_ref_global_prev])
            # print(transform_between)
            for pc_ in pc_win:
                pc_ : LidarPointCloud
                pc_.transform(transform_between)
            
            car_ref_global_prev = car_ref_global
            pc_win.append(pc)
        
        if len(pc_win) > win_len:
            pc_win.pop(0)

        if len(pc_win) == win_len:
            # copy points
            tmp = []
            for pc_ in pc_win:
                data = deepcopy(pc_.points.T)
                dist = np.linalg.norm(data[:,:2],axis=-1,keepdims=False)
                ind = np.argsort(dist)
                data = data[ind[:nkeep],:]
                data = data/norm_factor
                tmp.append(data)
            data_out.append(tmp)
            sample_tokens.append(sample_token)
            break

        sample_token = sample["next"]

    return data_out, sample_tokens

def plot_lidar_multisweep(lidar_pts):

    plt.figure(figsize=(12,12))
    for pc in lidar_pts:
        plt.scatter(pc[:,0],pc[:,1],0.5)

    plt.gca().set_aspect("equal")

def next_sample(nusc,sample,skip=1):
    """return the sample by skipping "skip" number of samples

    Args:
        nusc (_type_): _description_
        sample (_type_): _description_
        skip (int, optional): _description_. Defaults to 1.
    """
    for _ in range(skip):
        if sample["next"] != "":
            sample = nusc.get("sample", sample["next"])
        
    return sample

def prev_sample(nusc,sample,skip=1):
    for _ in range(skip):
        if sample["prev"] != "":
            sample = nusc.get("sample", sample["prev"])
    return sample