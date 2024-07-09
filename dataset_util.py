import os
import numpy as np
from numpy.linalg import multi_dot
from copy import copy, deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
from loguru import logger
from einops import rearrange, pack, einsum, reduce, repeat
from pyquaternion import Quaternion
from easydict import EasyDict as edict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
from box_overlaps import bbox_overlaps

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
    w,l,h = box.wlh
    yaw = box.orientation.yaw_pitch_roll[0]
    return [x,y,z,l,w,h,yaw]

def get_gt_boxes(nusc : NuScenes, sample, target_class="vehicle", render=False):
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    
    annotations = sample["anns"]
    assert(len(annotations) == len(boxes))

    gt_boxes = []

    for k,ann_token in enumerate(annotations):
        ann = nusc.get('sample_annotation', ann_token)
        # print(k,ann)
        class_name = ann["category_name"].split(".")[0]
        # print(class_name)
        if class_name == target_class:
            gt_boxes.append(parameterize_gt_boxes(boxes[k]))
            if render:
                print("======",k,ann,"======")
                nusc.render_annotation(ann_token)
                plt.show()
                print(gt_boxes[-1])

    return np.array(gt_boxes)

# def gt_boxes_to_array(gt_boxes):
#     """save gt_boxes to numpy array; each key mapped to a single numpy array

#     Args:
#         gt_boxes (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     x = []
#     y = []
#     z = []
#     l = []
#     w = []
#     h = []
#     yaw = []
#     for b in gt_boxes:
#         b : edict
#         x.append(b.x)
#         y.append(b.y)
#         z.append(b.z)
#         l.append(b.l)
#         w.append(b.w)
#         h.append(b.h)
#         yaw.append(b.yaw)
    
#     # convert to numpy array
#     x = np.array(x)[:,np.newaxis]
#     y = np.array(y)[:,np.newaxis]
#     z = np.array(z)[:,np.newaxis]
#     l = np.array(l)[:,np.newaxis]
#     w = np.array(w)[:,np.newaxis]
#     h = np.array(h)[:,np.newaxis]
#     yaw = np.array(yaw)[:,np.newaxis]

#     return x,y,z,l,w,h,yaw

def get_2d_corners(x,y,l,w,yaw):

    rot_mat = np.hstack([np.cos(yaw),-np.sin(yaw),np.sin(yaw),np.cos(yaw)])
    rot_mat = rearrange(rot_mat,"n (a b) -> n a b",a=2,b=2)

    xg,yg = np.meshgrid([-1,1],[-1,1])
    xg = xg.reshape((1,-1)) # row vector
    yg = yg.reshape((1,-1)) # row vector

    # in nuscene w:l is roughly 2:1
    x_ = rearrange(w/2*xg,"n d -> n 1 d", d=4) # w is associated with x
    y_ = rearrange(l/2*yg,"n d -> n 1 d", d=4) # l is associated with y
    xy,_ = pack([x_,y_],"n * d") # n x 2 x 4
    
    # batch matrix multiplication
    xy = einsum(rot_mat, xy, "n a b, n b c -> n a c") # n x 2 x 4
    center = rearrange(np.hstack([x,y]),"n d -> n d 1") # n x 2 x 1

    corners = center + xy # broadcasted addition

    return corners

def get_2d_upper_lower_corners(corners):
    """returns the upper and lower corners of the 2d bounding box

    Args:
        corners (_type_): _description_

    Returns:
        _type_: _description_
    """
    # corners size: "n 2 4"
    min_xy = reduce(corners,"n a b -> n a","min")
    max_xy = reduce(corners,"n a b -> n a","max")
    out = np.hstack([min_xy,max_xy])
    return out

class NusceneDataset(data.Dataset):

    def __init__(self,
                 path,
                 canvas_h,
                 canvas_w,
                 canvas_res_hw,
                 canvas_offset_hw,
                 anchors,
                 target_class="vehicle",
                 pos_neg_iou_thresh=[0.6,0.45],
                 version="v1.0-mini", ) -> None:
        super().__init__()

        self.nusc = NuScenes(version=version,dataroot=path,verbose=True)

        self.canvas_h = canvas_h # lateral axis, eg 400 (as in voxelnet paper)
        self.canvas_w = canvas_w # anterior-posterior axis, eg 352 
        self.canvas_res_hw = canvas_res_hw # resolution of the canvas
        self.canvas_offset_hw = canvas_offset_hw # offset of the canvas
        self.anchors = anchors # list of 3d anchor boxes, specified as l,w,h,yaw
        self.target_class = target_class
        self.pos_neg_iou_thresh = pos_neg_iou_thresh

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

        # compute the diagonal length of each anchor box
        for a in self.anchors:
            a["diag"] = np.sqrt(a["l"]**2 + a["w"]**2)

        self.anchor_z = np.array([a["z"] for a in self.anchors])
        self.anchor_l = np.array([a["l"] for a in self.anchors])
        self.anchor_w = np.array([a["w"] for a in self.anchors])
        self.anchor_d = np.array([a["diag"] for a in self.anchors])
        self.anchor_h = np.array([a["h"] for a in self.anchors])
        self.anchor_yaw = np.array([a["yaw"] for a in self.anchors])

        # create bounding boxes at each of the canvas locations
        self.anchor_box_2corners, self.anchor_center_xy = \
            self.create_2d_bbox_on_canvas()

    def create_2d_bbox_on_canvas(self,):
        h = self.canvas_res_hw[0]/2*np.arange(self.canvas_h)+\
            self.canvas_offset_hw[0]
        w = self.canvas_res_hw[1]/2*np.arange(self.canvas_w)+\
            self.canvas_offset_hw[1]

        hg, wg = np.meshgrid(h,w) # h x w
        hg = rearrange(hg,"h w -> (h w) 1")
        wg = rearrange(wg,"h w -> (h w) 1")
        canvas_grid = rearrange(np.hstack([hg,wg]),"m n -> m 1 n") # m by 1 by 2

        # by m by n_anchor by 2
        anchor_center = repeat(canvas_grid,"m 1 n -> m na n",na=self.n_anchor)

        # create box offsets
        L = np.array([a["l"] for a in self.anchors])[:,np.newaxis]# nanchor by 1
        W = np.array([a["w"] for a in self.anchors])[:,np.newaxis]
        yaw = np.array([a["yaw"] for a in self.anchors])[:,np.newaxis]

        rot_mat = np.hstack([np.cos(yaw),-np.sin(yaw),np.sin(yaw),np.cos(yaw)])
        rot_mat = rearrange(rot_mat,"n (a b) -> n a b",a=2,b=2)

        xg,yg = np.meshgrid([-1,1],[-1,1])
        xg = xg.reshape((1,-1)) # row vector
        yg = yg.reshape((1,-1)) # row vector

        x_ = rearrange(L/2*xg,"na d -> na 1 d", d=4)
        y_ = rearrange(W/2*yg,"na d -> na 1 d", d=4)
        xy,_ = pack([x_,y_],"na * d") # nanchor x 2 x 4
        xy = einsum(rot_mat, xy, "na a b, na b c -> na a c")# nx2x4

        # broadcast addition
        anchor_box = rearrange(anchor_center,"m na c -> m na c 1",c=2) + \
            rearrange(xy,"na c d -> 1 na c d",c=2,d=4) # m x nanchor x 2 x 4

        # calculate upper and lower corners
        anchor_box_2corners,_ = pack([
            reduce(anchor_box,"m na n d -> m na n","min",n=2),
            reduce(anchor_box,"m na n d -> m na n","max",n=2)],
            "m na *"
        ) # (h w) by nanchor by 4; xmin, ymin, xmax, ymax
        anchor_box_2corners = rearrange(anchor_box_2corners,
                                        "m na n -> (m na) n")
        
        anchor_center = rearrange(anchor_center, "(h w) na n -> h w na n",
                                  h=self.canvas_h)
        return anchor_box_2corners, anchor_center
            

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, ind):

        sample_token = self.sample_tokens[ind]

        # get 3D bounding box
        gt_boxes = get_gt_boxes(self.nusc,self.nusc.get("sample",sample_token),
                                self.target_class)
        n_gt_box = gt_boxes.shape[0]
        gt_x,gt_y,gt_z,gt_l,gt_w,gt_h,gt_yaw = \
            [gt_boxes[:,[_]] for _ in range(gt_boxes.shape[1])]
        gt_2d_4corners = get_2d_corners(gt_x,gt_y,gt_l,gt_w,gt_yaw)

        # from canvas and anchors, calculate iou's
        # get 2d corners, upper left and lower right bounding box
        gt_2d_2corners = get_2d_upper_lower_corners(gt_2d_4corners) # n by 2

        # calculate upper and lower corner for the anchors
        iou = bbox_overlaps(
            self.anchor_box_2corners,
            gt_2d_2corners,
        ) # nanchor by ngtbox

        # for each ground truth box, find the highest iou anchor
        highest_iou_anchor_id = np.argmax(iou,axis=0)
        highest_gt_id = np.arange(gt_boxes.shape[0])
        mask = iou[highest_iou_anchor_id,highest_gt_id] > 0.0 # keep positive iou
        highest_iou_anchor_id = highest_iou_anchor_id[mask]
        highest_gt_id = highest_gt_id[mask]

        # go through the rest of the anchors
        ind_tmp = np.argwhere(iou > self.pos_neg_iou_thresh[0])
        pos_anchor_id, pos_gt_id = ind_tmp[:,0], ind_tmp[:,1]

        # combine all positive ones
        pos_anchor_id = np.concatenate([highest_iou_anchor_id, pos_anchor_id])
        pos_gt_id = np.concatenate([highest_gt_id, pos_gt_id])

        # keep the unique ones
        pos_anchor_id, ind_tmp = np.unique(pos_anchor_id,return_index=True)
        pos_gt_id = pos_gt_id[ind_tmp]

        mask = np.zeros(iou.shape,dtype=bool)
        mask[pos_anchor_id,pos_gt_id] = True

        # get the neg ones, i.e., iou with all gt boxes fall below the threshold
        neg_anchor_id = np.argwhere(
            np.sum(iou<self.pos_neg_iou_thresh[1],axis=1) == n_gt_box)
        neg_anchor_id = neg_anchor_id.flatten()

        # create regression target
        ind_x_p, ind_y_p, ind_a_p = np.unravel_index(pos_anchor_id,
                                (self.canvas_h,self.canvas_w,self.n_anchor))
        
        # compute the difference
        pos_anchor_center_x = self.anchor_center_xy[ind_x_p,ind_y_p,ind_a_p,0]
        pos_anchor_center_y = self.anchor_center_xy[ind_x_p,ind_y_p,ind_a_p,1]
        pos_anchor_center_z = self.anchor_z[ind_a_p]
        pos_anchor_diag = self.anchor_d[ind_a_p]
        pos_anchor_l = self.anchor_l[ind_a_p]
        pos_anchor_w = self.anchor_w[ind_a_p]
        pos_anchor_h = self.anchor_h[ind_a_p]
        pos_anchor_yaw = self.anchor_yaw[ind_a_p]

        pos_gt_center_x, pos_gt_center_y = gt_x[pos_gt_id], gt_y[pos_gt_id]
        pos_gt_center_z = gt_z[pos_gt_id]
        pos_gt_l, pos_gt_w, pos_gt_h = gt_l[pos_gt_id], gt_w[pos_gt_id], \
            gt_h[pos_gt_id]
        pos_gt_yaw = gt_yaw[pos_gt_id]
        
        dx = (pos_gt_center_x - pos_anchor_center_x)/pos_anchor_diag
        dy = (pos_gt_center_y - pos_anchor_center_y)/pos_anchor_diag
        dz = (pos_gt_center_z - pos_anchor_center_z)/pos_anchor_h
        dl = np.log(pos_gt_l/pos_anchor_l)
        dw = np.log(pos_gt_w/pos_anchor_w)
        dh = np.log(pos_gt_h/pos_anchor_h)
        dyaw = pos_gt_yaw - pos_anchor_yaw

        # convert variables to column vectors
        dx = dx[:, np.newaxis]
        dy = dy[:, np.newaxis]
        dz = dz[:, np.newaxis]
        dl = dl[:, np.newaxis]
        dw = dw[:, np.newaxis]
        dh = dh[:, np.newaxis]
        dyaw = dyaw[:, np.newaxis]

        # reg_target = np.hstack([dx, dy, dz, dl, dw, dh, dyaw])
        reg_target = np.zeros((self.canvas_h,self.canvas_w,self.n_anchor,7))
        reg_target[ind_x_p,ind_y_p,ind_a_p,:] = np.hstack([dx,dy,dz,dl,dw,dh,dyaw])
        reg_target = rearrange(reg_target,"h w na d -> h w (na d)",d=7)

        # convert indices to mask
        pos_anchor_id_mask = np.zeros(
            (self.canvas_h,self.canvas_w,self.n_anchor))
        pos_anchor_id_mask[ind_x_p,ind_y_p,ind_a_p] = 1.0

        neg_anchor_id_mask = np.zeros(
            (self.canvas_h,self.canvas_w,self.n_anchor))
        ind_x_n, ind_y_n, ind_a_n = np.unravel_index(neg_anchor_id,
                                (self.canvas_h,self.canvas_w,self.n_anchor))
        neg_anchor_id_mask[ind_x_n,ind_y_n,ind_a_n] = 1.0

        return pos_anchor_id_mask, neg_anchor_id_mask, reg_target

    def cal_target(self, ):
        pass
