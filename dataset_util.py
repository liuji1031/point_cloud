import os
import numpy as np
from numpy.linalg import multi_dot
from copy import copy, deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data as data
import torch
import pickle
from copy import deepcopy
import blosc
from tqdm import tqdm
from loguru import logger
from einops import rearrange, pack, einsum, reduce, repeat
from pyquaternion import Quaternion
from easydict import EasyDict as edict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
from iou import iou_box_array

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

def get_lidar_pts_singlesweep(nusc, sample, sensor="LIDAR_TOP",
                              convert_to_ego_frame=True,
                              min_dist=1.0,
                              nkeep="all"):
    """returns the lidar points for a single sweep

    Args:
        nusc (_type_): _description_
        sample (_type_): _description_
        sensor (str, optional): _description_. Defaults to "LIDAR_TOP".
        convert_to_ego_frame (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    sample_data = get_sensor_data(nusc,sample,sensor)

    pc = get_lidar_pts(nusc,sample)
    pc.remove_close(min_dist)

    if not convert_to_ego_frame:
        return pc.points.T

    # get ego pose and transformation
    ego_pose = get_ego_pose(nusc,sample,sensor)
    # car_ref_global = transform_matrix_from_pose(ego_pose,inverse=False)

    # get sensor ref car transformation
    sensor_ref_car = transform_matrix_from_pose(
                        nusc.get("calibrated_sensor",
                                sample_data["calibrated_sensor_token"]),
                        inverse=False
                    )
    
    # convert sensor points to car body frame
    pc.transform(sensor_ref_car)

    data = pc.points.T
    dist = np.linalg.norm(data[:,:2],axis=-1,keepdims=False)
    ind = np.argsort(dist)
    if nkeep != "all":
        if nkeep <= len(ind):
            data = data[ind[:nkeep],:]
        else:
            data = data[ind,:]
            data = np.vstack([data,np.zeros((nkeep-len(ind),data.shape[1]))])

    return data

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
                if nkeep != "all":
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

def plot_lidar(lidar_pts,show=True):
    plt.figure(figsize=(12,12),dpi=300)
    if isinstance(lidar_pts, LidarPointCloud):
        lidar_pts = lidar_pts.points.T
    plt.scatter(lidar_pts[:,0],lidar_pts[:,1],0.02)
    plt.gca().set_aspect("equal")
    if show:
        plt.show()

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
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'],
                                       use_flat_vehicle_coordinates=True)
    
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

    return np.array(gt_boxes), boxes

def plot_box_2d(gt_boxes,linespec="b--"):
    x,y,z,l,w,h,yaw = gt_boxes[:,[0]], gt_boxes[:,[1]], gt_boxes[:,[2]], \
        gt_boxes[:,[3]], gt_boxes[:,[4]], gt_boxes[:,[5]], gt_boxes[:,[6]]
    corners = get_2d_corners(x,y,l,w,yaw)

    plot_box_2d_corners(corners,linespec)

def plot_box_2d_corners(corners,linespec="b--"):
    seq = [0,1,3,2]
    for i in range(corners.shape[0]):
        x = np.concatenate([corners[i,0,seq], corners[i,0,0][np.newaxis]])
        y = np.concatenate([corners[i,1,seq], corners[i,1,0][np.newaxis]])
        plt.plot(x,y,linespec)

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
    x_ = rearrange(l/2*xg,"n d -> n 1 d", d=4) # w is associated with x
    y_ = rearrange(w/2*yg,"n d -> n 1 d", d=4) # l is associated with y
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
                 nusc,
                 canvas_h,
                 canvas_w,
                 canvas_res,
                 canvas_min,
                 anchors,
                 voxelizer,
                 max_pts_per_cloud=25000,
                 target_class="vehicle",
                 pos_neg_iou_thresh=[0.6,0.45],
                 train_val_test_split=[0.8,0.1,0.1],
                 mode="train",
                 seed=0,
                 aug_angles=None,
                 load_preprocessed=False,
                 load_lidar_from_disk=None,
                 sample_pkl_path="/home/ji/Dropbox/Robotics/PointCloud/Project/point_cloud/sample_tokens_by_scene.pkl") -> None:
        super().__init__()

        self.load_preprocessed = load_preprocessed
        self.load_lidar_from_disk = load_lidar_from_disk
        self.nusc = nusc
        self.max_pts_per_cloud = max_pts_per_cloud
        self.voxelizer = voxelizer
        self.aug_angles = aug_angles if aug_angles is not None else [0]

        self.canvas_h = canvas_h # lateral axis, eg 400 (as in voxelnet paper)
        self.canvas_w = canvas_w # anterior-posterior axis, eg 352
        self.canvas_res = canvas_res # resolution of the canvas
        self.canvas_min = canvas_min # min of the canvas
        self.canvas_max = {"h":canvas_min["h"]+canvas_res["h"]*canvas_h,
                           "w":canvas_min["w"]+canvas_res["w"]*canvas_w}
        self.anchors = anchors # list of 3d anchor boxes, specified as l,w,h,yaw
        self.target_class = target_class
        self.pos_neg_iou_thresh = pos_neg_iou_thresh

        self.n_anchor = len(anchors)

        # go through all scenes and get all the sample tokens
        self.sample_tokens = []
        self.sample_tokens_by_scene = []
        if self.nusc is None:
            # load from disk
            # use pickle to load from sample_tokens.pkl
            logger.info("Loading sample tokens from disk")
            self.sample_tokens_by_scene = \
                load_pickle_compressed(sample_pkl_path)
        else:
            for scene in self.nusc.scene:
                tokens = []
                sample_token = scene["first_sample_token"]
                while sample_token != "":
                    tokens.append(sample_token)
                    sample_token = self.nusc.get("sample",sample_token)["next"]
                self.sample_tokens_by_scene.append(tokens)
                save_pickle_compressed(self.sample_tokens_by_scene,
                                       "sample_tokens_by_scene.pkl")

        n_samples_total = np.sum([len(t) for t in self.sample_tokens_by_scene])
        logger.info(f"Total number of samples: {n_samples_total}")

        # use the seed to shuffle the sample tokens
        if seed is not None:
            np.random.seed(seed)

        # now use stratified sampling to split samples from each scene into
        # train, val, and test
        for tokens in self.sample_tokens_by_scene:
            n_samples = len(tokens)
            split_ind = (np.cumsum(train_val_test_split)*n_samples).astype(int)
            ind = np.random.permutation(n_samples)

            if mode == "train":
                self.sample_tokens += \
                    [ tokens[i] for i in ind[:split_ind[0]] ]
            elif mode == "val":
                self.sample_tokens += \
                    [ tokens[i] for i in ind[split_ind[0]:split_ind[1]] ]
            elif mode == "test":
                self.sample_tokens += \
                    [ tokens[i] for i in ind[split_ind[1]:] ]
            elif mode == "all":
                self.sample_tokens += tokens

        logger.info(
            f"Number of samples for mode {mode}: {len(self.sample_tokens)}")
        self.n_samples = len(self.sample_tokens)

        # compute the diagonal length of each anchor box
        for a in self.anchors:
            a["diag"] = np.sqrt(a["l"]**2 + a["w"]**2)

        self.anchor_z = np.array([a["z"] for a in self.anchors])[:,np.newaxis]
        self.anchor_l = np.array([a["l"] for a in self.anchors])[:,np.newaxis]
        self.anchor_w = np.array([a["w"] for a in self.anchors])[:,np.newaxis]
        self.anchor_d = np.array([a["diag"] for a in self.anchors])[:,np.newaxis]
        self.anchor_h = np.array([a["h"] for a in self.anchors])[:,np.newaxis]
        self.anchor_yaw = np.array([a["yaw"] for a in self.anchors])[:,np.newaxis]

        # create bounding boxes at each of the canvas locations
        self.anchor_box, self.anchor_box_2corners, self.anchor_center_xy = \
            self.create_2d_bbox_on_canvas()

    def create_2d_bbox_on_canvas(self,):
        # h->y w->x
        h = self.canvas_res["h"]*np.arange(self.canvas_h)+\
            self.canvas_min["h"]+self.canvas_res["h"]/2
        w = self.canvas_res["w"]*np.arange(self.canvas_w)+\
            self.canvas_min["w"]+self.canvas_res["w"]/2
        
        hg, wg = np.meshgrid(h,w, indexing="ij") # h x w
        hg = rearrange(hg,"h w -> (h w) 1")
        wg = rearrange(wg,"h w -> (h w) 1")
        canvas_grid = np.hstack([wg,hg]) # m by 2; notice wg comes before hg
                                         # as w is x and h is y

        # by m by n_anchor by 2
        anchor_center = repeat(canvas_grid,"m n -> m na n",na=self.n_anchor)

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
        anchor_box = rearrange(anchor_box, "(h w) na n d -> h w na n d",
                                h=self.canvas_h)
        return anchor_box, anchor_box_2corners, anchor_center
            
    def __len__(self):
        return self.n_samples
    
    def exclude_out_of_range_gtbox(self,gt_boxes):
        # h->y w->x

        if gt_boxes.shape[0] == 0:
            return gt_boxes

        mask = (gt_boxes[:,0] >= self.canvas_min["w"]) & \
               (gt_boxes[:,0] < self.canvas_max["w"]) & \
               (gt_boxes[:,1] >= self.canvas_min["h"]) & \
               (gt_boxes[:,1] < self.canvas_max["h"])

        return gt_boxes[mask,:] # n by 7
    
    def compute_target(self, gt_boxes):
        """_summary_

        Args:
            gt_boxes (_type_): 2D array contains the ground truth boxes. The
            columns contains x, y, z, l, h, w, yaw respectively. Each row is one
            box

        Returns:
            _type_: _description_
        """
        if gt_boxes.shape[0]==0:
            n_gt_box = 0
        else:
            n_gt_box = gt_boxes.shape[0]

        if n_gt_box == 0:
            pos_anchor_id_mask = np.zeros(
            (self.canvas_h,self.canvas_w,self.n_anchor))
            neg_anchor_id_mask = np.ones(
                (self.canvas_h,self.canvas_w,self.n_anchor))
            reg_target = np.zeros(
                (self.canvas_h,self.canvas_w,self.n_anchor*7))
            
            return pos_anchor_id_mask, neg_anchor_id_mask, reg_target

        gt_x,gt_y,gt_z,gt_l,gt_w,gt_h,gt_yaw = \
            [gt_boxes[:,[_]] for _ in range(gt_boxes.shape[1])]
        gt_2d_4corners = get_2d_corners(gt_x,gt_y,gt_l,gt_w,gt_yaw)

        # from canvas and anchors, calculate iou's
        # get 2d corners, upper left and lower right bounding box
        gt_2d_2corners = get_2d_upper_lower_corners(gt_2d_4corners) # n by 2

        # print(gt_2d_2corners)

        # calculate upper and lower corner for the anchors
        n_anchor_box = self.anchor_box_2corners.shape[0]
        iou = iou_box_array(
            self.anchor_box_2corners.astype(np.float32),
            gt_2d_2corners.astype(np.float32),
        ) # nanchor by ngtbox

        # print(iou.shape)
        # print(iou)

        # for each ground truth box, find the highest iou anchor
        highest_iou_anchor_id = np.argmax(iou,axis=0)
        highest_gt_id = np.arange(n_gt_box)

        # keep positive iou
        mask = iou[highest_iou_anchor_id,highest_gt_id] > 0.0
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
        pos_anchor_center_x = \
            self.anchor_center_xy[ind_x_p,ind_y_p,ind_a_p,0].reshape((-1,1))
        pos_anchor_center_y = \
            self.anchor_center_xy[ind_x_p,ind_y_p,ind_a_p,1].reshape((-1,1))
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
        
        # all delta variables are column vectors
        dx = (pos_gt_center_x - pos_anchor_center_x)/pos_anchor_diag
        dy = (pos_gt_center_y - pos_anchor_center_y)/pos_anchor_diag
        dz = (pos_gt_center_z - pos_anchor_center_z)/pos_anchor_h
        dl = np.log(pos_gt_l/pos_anchor_l)
        dw = np.log(pos_gt_w/pos_anchor_w)
        dh = np.log(pos_gt_h/pos_anchor_h)
        dyaw = pos_gt_yaw - pos_anchor_yaw

        # reg_target = np.hstack([dx, dy, dz, dl, dw, dh, dyaw])
        reg_target = np.zeros((self.canvas_h,self.canvas_w,self.n_anchor,7))
        reg_target[ind_x_p,ind_y_p,ind_a_p] = \
            np.hstack([dx,dy,dz,dl,dw,dh,dyaw])
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

        # make sure pos and neg mask have no overlap
        neg_anchor_id_mask = neg_anchor_id_mask*(1-pos_anchor_id_mask)

        return pos_anchor_id_mask, neg_anchor_id_mask, reg_target
    
    def get_lidar_pts(self, ind):
        sample_token = self.sample_tokens[ind]
        sample = self.nusc.get("sample",sample_token)

        lidar_pts = get_lidar_pts_singlesweep(self.nusc,sample,
                                              convert_to_ego_frame=True,
                                              min_dist=1.0,
                                              nkeep=self.max_pts_per_cloud)
    
        return lidar_pts
    
    def rotate_lidar_pts(self, lidar_pts, angle, return_copy=True):
        if angle == 0:
            return lidar_pts if not return_copy else deepcopy(lidar_pts)
        
        rot_mat, _ = self.rot_mat_from_angle(angle)
        lidar_pts_ = deepcopy(lidar_pts) if return_copy else lidar_pts
        # rotate the lidar points
        lidar_pts_[:,:2] = np.linalg.multi_dot([lidar_pts_[:,:2],rot_mat.T])

        return lidar_pts_
    
    def rot_mat_from_angle(self,angle):
        angle_rad = np.deg2rad(angle)
        rot_mat = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],
                            [np.sin(angle_rad),np.cos(angle_rad)]])
        return rot_mat, angle_rad

    def rotate_gt_boxes(self,gt_boxes,angle,return_copy=True):
        if angle == 0:
            return gt_boxes if not return_copy else deepcopy(gt_boxes)
        rot_mat, angle_rad = self.rot_mat_from_angle(angle)
        gt_boxes_ = gt_boxes if not return_copy else deepcopy(gt_boxes)
        gt_boxes_[:,:2] = np.linalg.multi_dot([gt_boxes_[:,:2],rot_mat.T])
        gt_boxes_[:,6] += angle_rad

        return gt_boxes_

    def __getitem__(self, ind):

        sample_token = self.sample_tokens[ind]
        # print(sample_token)

        angle = 0 # default no augmentation
        if len(self.aug_angles)>0: # more than 0 value specified
            # choose an angle from self.aug_angles, which is a list
            # of angles in degrees
            angle = np.random.choice(self.aug_angles)
            # print(f"random angle: {angle}")
        
        if self.load_lidar_from_disk is None:
            # get the lidar points
            lidar_pts = self.get_lidar_pts(ind)
            # get 3D bounding box
            gt_boxes, _  = get_gt_boxes(self.nusc, self.nusc.get("sample",sample_token),
                                        self.target_class, render=False)
        else:
            path = Path(self.load_lidar_from_disk) / f"{sample_token}.pkl"
            # print(f"loading {path}")
            data = load_pickle_compressed(path)
            lidar_pts = data["lidar_pts"]
            gt_boxes = data["gt_boxes"]
        
        gt_boxes = self.exclude_out_of_range_gtbox(gt_boxes)
            
        if angle != 0:
            lidar_pts = self.rotate_lidar_pts(lidar_pts,angle,return_copy=False)
            
        if angle != 0 and gt_boxes.shape[0] > 0:
            gt_boxes = self.rotate_gt_boxes(gt_boxes,angle,return_copy=False)

        pos_anchor_id_mask, neg_anchor_id_mask, reg_target = \
            self.compute_target(gt_boxes)

        # perform voxelization here
        voxel, coord, mask, counts = self.voxelizer(lidar_pts)

        # convert to numpy array of similar precision
        voxel = voxel.astype(np.float32)
        coord = coord.astype(np.int32)
        mask = mask.astype(np.float32)
        # counts = counts.astype(np.int32)
        pos_anchor_id_mask = pos_anchor_id_mask.astype(np.float32)
        neg_anchor_id_mask = neg_anchor_id_mask.astype(np.float32)
        reg_target = reg_target.astype(np.float32)

        out = {"voxel":voxel,
                "coord":coord,
                "mask":mask,
                "pos_anchor_id_mask":pos_anchor_id_mask,
                "neg_anchor_id_mask":neg_anchor_id_mask,
                "reg_target":reg_target,
                "gt_boxes":gt_boxes,
                "token":sample_token}

        # return a dictionary of the following variables with the same key name:
        # lidar_pts, pos_anchor_id_mask, neg_anchor_id_mask, reg_target
        return out
    
    def retrieve_by_token(self, sample_token):
        for i, token in enumerate(self.sample_tokens):
            if token == sample_token:
                return self.__getitem__(i)
        return None
    
    def get_lidar_pts_by_token(self, sample_token):
        sample = self.nusc.get("sample",sample_token)

        lidar_pts = get_lidar_pts_singlesweep(self.nusc,sample,
                                              convert_to_ego_frame=True,
                                              min_dist=1.0,
                                              nkeep=self.max_pts_per_cloud)
        return lidar_pts
    
    def save_lidar_gtbox_to_disk(self, save_disk_path):
        for ind, sample in tqdm(enumerate(self.sample_tokens),
                                total=len(self.sample_tokens)):
            lidar_pts = self.get_lidar_pts(ind)
            gt_boxes, _ = get_gt_boxes(self.nusc,self.nusc.get("sample",sample),
                                        self.target_class, render=False)
            save_path = Path(save_disk_path) / f"{sample}.pkl"
            save_pickle_compressed({"lidar_pts":lidar_pts, "gt_boxes":gt_boxes},
                                   save_path)

def to_device(data,device):
    for k,v in data.items():
        if k=="gt_boxes" or k=="tokens":
            continue
        if isinstance(v,np.ndarray):
            v = torch.from_numpy(v)
        data[k] = v.to(device)
    return data

def save_pickle_compressed(data,save_path):
    pickled_data = pickle.dumps(data)
    compressed_pickle = blosc.compress(pickled_data)
    with open(save_path,"wb") as f:
        f.write(compressed_pickle)

def load_pickle_compressed(load_path):
    with open(load_path,"rb") as f:
        compressed_pickle = f.read()
    depressed_pickle = blosc.decompress(compressed_pickle)
    return pickle.loads(depressed_pickle)

from einops import pack
def collate_fn(batches):
    out = {}
    voxel = []
    coord = []
    mask = []
    pos_anchor_id_mask = []
    neg_anchor_id_mask = []
    reg_target = []
    gt_boxes = []
    tokens = []
    for batch_ind, batch in enumerate(batches):

        voxel.append(batch["voxel"])
        coord.append(
            np.pad(batch["coord"],
                   pad_width=((0,0),(1,0)),
                   mode="constant",constant_values=batch_ind)
                   )
        mask.append(batch["mask"])
        pos_anchor_id_mask.append(batch["pos_anchor_id_mask"][np.newaxis,:])
        neg_anchor_id_mask.append(batch["neg_anchor_id_mask"][np.newaxis,:])
        reg_target.append(batch["reg_target"][np.newaxis,:])
        gt_boxes.append(batch["gt_boxes"])
        tokens.append(batch["token"])

    voxel,_ = pack(voxel, "* p d")
    coord,_ = pack(coord, "* d")
    mask,_ = pack(mask, "* p")
    pos_anchor_id_mask,_ = pack(pos_anchor_id_mask, "* h w na")
    neg_anchor_id_mask,_ = pack(neg_anchor_id_mask, "* h w na")
    reg_target,_ = pack(reg_target, "* h w naxd")
    
    # voxel = np.concatenate(voxel,axis=0)
    # coord = np.concatenate(coord,axis=0)
    # mask = np.concatenate(mask,axis=0)
    # pos_anchor_id_mask = np.concatenate(pos_anchor_id_mask,axis=0)
    # neg_anchor_id_mask = np.concatenate(neg_anchor_id_mask,axis=0)
    # reg_target = np.concatenate(reg_target,axis=0)

    out["voxel"] = torch.from_numpy(voxel)
    out["coord"] = torch.from_numpy(coord)
    out["mask"] = torch.from_numpy(mask)
    out["pos_anchor_id_mask"] = torch.from_numpy(pos_anchor_id_mask)
    out["neg_anchor_id_mask"] = torch.from_numpy(neg_anchor_id_mask)
    out["reg_target"] = torch.from_numpy(reg_target)
    out["gt_boxes"] = gt_boxes
    out["tokens"] = tokens
        
    return out

from config import cfg
def get_database(mode, nusc=None):
    nd = NusceneDataset(nusc=nusc,
                    canvas_h=cfg.canvas_size["h"],
                    canvas_w=cfg.canvas_size["w"],
                    canvas_res=cfg.canvas_res,
                    canvas_min=cfg.canvas_min,
                    anchors=cfg.anchors,
                    voxelizer=cfg.voxelizer,
                    max_pts_per_cloud=cfg.max_points_per_cloud,
                    pos_neg_iou_thresh=cfg.pos_neg_iou_thresh,
                    train_val_test_split=cfg.train_val_test_split,
                    aug_angles=cfg.aug_angles[mode],
                    load_lidar_from_disk=cfg.lidar_path,
                    mode=mode)
    logger.info("First 5 sample tokens:")
    for _ in range(5):
        logger.info(nd.sample_tokens[_])

    return nd

from torch.utils.data import DataLoader
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True,pin_memory_device="cuda",
                      num_workers=16, collate_fn=collate_fn)

def plot_gt_data(lidar_pts, gt_boxes):
    plt.figure(figsize=(10,10))
    plot_lidar(lidar_pts,show=False)
    plot_box_2d(gt_boxes,linespec="g-")