import os
import numpy as np
from numpy.linalg import multi_dot
from copy import copy, deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
from loguru import logger
from pyquaternion import Quaternion
from easydict import EasyDict as edict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
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
            # break

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

def parameterize_gt_boxes(box : Box):
    x,y,z = box.center
    l,w,h = box.wlh
    yaw = box.orientation.yaw_pitch_roll[0]
    return edict(x=x,y=y,z=z,l=l,w=w,h=h,yaw=yaw)

def get_gt_boxes(nusc : NuScenes, sample, target_class="vehicle"):
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    
    annotations = sample["anns"]
    assert(len(annotations) == len(boxes))

    gt_boxes = []

    for k,ann in enumerate(annotations):
        ann = nusc.get('sample_annotation', ann)
        # print(k,ann)
        class_name = ann["category_name"].split(".")[0]
        print(class_name)
        if class_name == target_class:
            gt_boxes.append(parameterize_gt_boxes(boxes[k]))

    return gt_boxes

def gt_boxes_to_array(gt_boxes):
    """save gt_boxes to numpy array; each key mapped to a single numpy array

    Args:
        gt_boxes (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = []
    y = []
    z = []
    l = []
    w = []
    h = []
    yaw = []
    for b in gt_boxes:
        b : edict
        x.append(b.x)
        y.append(b.y)
        z.append(b.z)
        l.append(b.l)
        w.append(b.w)
        h.append(b.h)
        yaw.append(b.yaw)
    
    # convert to numpy array
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    l = np.array(l)
    w = np.array(w)
    h = np.array(h)
    yaw = np.array(yaw)

    return x,y,z,l,w,h,yaw

class NusceneDataset(data.Dataset):

    def __init__(self,path,
                 canvas_h,
                 canvas_w,
                 anchors,
                 target_class="vehicle",
                 pos_neg_iou_threh=[0.6,0.45],
                 version="v1.0-mini", ) -> None:
        super().__init__()

        self.nusc = NuScenes(version=version,dataroot=path,verbose=True)

        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.anchors = anchors # list of 3d anchor boxes
        self.target_class = target_class
        self.pos_neg_iou_threh = pos_neg_iou_threh

        self.n_anchor = len(anchors)

        # go through all scenes and get all the sample tokens
        self.sample_tokens = []
        for scene in self.nusc.scene:
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                self.sample_tokens.append(sample_token)
                sample_token = self.nusc.get("sample",sample_token)["next"]

        self.n_samples = len(self.sample_tokens)
        logger.info(f"Total number of samples: {self.n_samples}")

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, ind):

        sample_token = self.sample_tokens[ind]

        # get 3D bounding box
        gt_boxes = get_gt_boxes(self.nusc,self.nusc.get("sample",sample_token),
                                self.target_class)
        x,y,z,l,w,h,yaw = gt_boxes_to_array(gt_boxes)

        # from canvas and anchors, calculate iou's

        
        
    def cal_target(self, ):
        pass
