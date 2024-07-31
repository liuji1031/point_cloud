import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat, pack
from voxelization import Voxelization
from voxel_feature import VoxelFeatureExtraction
from spatial_conv import SpatialConvolution
from rpn import RegionProposalNet
from loss import VoxelNetLoss
from loguru import logger

class VoxelNet(nn.Module):
    """define the full voxelnet architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 canvas_spatial_shape_DHW,
                 anchors,
                 pos_cls_weight=2.0,
                 neg_cls_weight=1.0,
                 l1_weight=1.0,
                 device="cuda"
                 ):
        super().__init__()
        self.name="VoxelNet"

        # x: front, y: left, z: up
        self.canvas_spatial_shape_DHW = canvas_spatial_shape_DHW
        # 4 features in: x,y,z,intensity
        # after decoration, 7 features in
        self.voxel_feature = VoxelFeatureExtraction(n_feat_in=7,
                                                    n_hidden_per_layer=[32,128],
                                                    n_feat_out=128,)
        
        # spatial convolution
        self.scn = SpatialConvolution(n_feat_in=128, to_BEV=True)
        self.scn = torch.compile(self.scn)

        # region proposal network
        self.rpn = RegionProposalNet(nanchor=len(anchors))

        # loss function
        logger.info(f"pos_cls_weight: {pos_cls_weight}, neg_cls_weight: {neg_cls_weight}, l1 weight: {l1_weight}")
        self.loss = torch.compile(VoxelNetLoss(pos_cls_weight=pos_cls_weight,
                                               neg_cls_weight=neg_cls_weight,
                                               l1_weight=l1_weight))
        
        self.device = device
    
    def rearrange_batch(self, voxel, mask, coord, counts):
        # voxel size: (batch_size, max_voxel_num, max_voxel_pts, n_feat_in)
        # mask size: (batch_size, max_voxel_num, max_voxel_pts)
        # coord size: (batch_size, max_voxel_num, 3)
        # counts size: (batch_size, max_voxel_num)
        n_batch = voxel.shape[0]
        max_voxel_num = voxel.shape[1]

        voxel = rearrange(voxel, "b n p f -> (b n) p f")
        mask = rearrange(mask, "b n p -> (b n) p")
        coord = rearrange(coord, "b n d -> (b n) d")
        counts = rearrange(counts, "b n -> (b n)")
        ind_batch = torch.arange(n_batch, dtype=torch.int32, device=self.device)
        ind_batch = repeat(ind_batch, "b -> (b n) 1", n=max_voxel_num)
        coord,_ = pack([ind_batch, coord],"b *")
        # coord.to(torch.int32)

        return voxel[counts>0], mask[counts>0], coord[counts>0]

    def forward(self, input_dict):

        voxel_batch = input_dict["voxel"]
        mask_batch = input_dict["mask"]
        coord_batch = input_dict["coord"]
        # counts_batch = input_dict["counts"]

        # voxel_batch, mask_batch, coord_batch = \
        #     self.rearrange_batch(voxel_batch,mask_batch,
        #                          coord_batch,counts_batch)
        
        # print(voxel_batch.shape, mask_batch.shape, coord_batch.shape)

        pos_anchor_id_mask = input_dict["pos_anchor_id_mask"]
        neg_anchor_id_mask = input_dict["neg_anchor_id_mask"]
        reg_target = input_dict["reg_target"]

        # voxelization moved into dataset util
        
        # voxel feature extraction
        # first dim of coord indicates which batch the point comes from
        batch_size = int(
            torch.max(coord_batch[:,0]).detach().cpu().numpy().item()+1
            )
        voxel_feat = self.voxel_feature(voxel_batch, mask_batch)

        # print(voxel_feat.shape)

        # spatial convolution
        voxel_feat = self.scn(voxel_feat,
                              coord_batch,
                              self.canvas_spatial_shape_DHW,
                              batch_size)
        
        # print(voxel_feat.shape)
        
        # region proposal network
        cls_head, reg_head = self.rpn(voxel_feat)

        # print(cls_head.shape, reg_head.shape)

        # calculate loss
        loss = self.loss(cls_head, reg_head,
                         pos_anchor_id_mask, neg_anchor_id_mask, reg_target)
        
        return (cls_head, reg_head), loss


from dataset_util import get_2d_corners

def gen_bbox(cls_head,reg_head,anchors,anchor_center_xy,
              prob_threshold=0.7, sort_by_prob=True):
    """_summary_

    Args:
        cls_head (_type_): h by w by nanchor
        reg_head: h by w by nanchor by 7
        anchor_center_xy (_type_): h by w by nanchor by 2
        anchor_boxes (_type_): h by w by nanchor by 4: xmin, ymin, xmax, ymax
        prob_threshold (float, optional): _description_. Defaults to 0.7.
    """

    cls_head = cls_head.detach().cpu().numpy()
    reg_head = reg_head.detach().cpu().numpy() # h by w by nanchor by 7
    ind_pos = np.argwhere(cls_head>prob_threshold)

    if ind_pos.shape[0]==0:
        return None
    
    if sort_by_prob:
        score = cls_head[ind_pos[:,0],ind_pos[:,1],ind_pos[:,2]]
        isort = np.argsort(score)[::-1]
        ind_pos = ind_pos[isort]

    pred_val = []
    for i in range(ind_pos.shape[0]):
        anchor_id,ih,iw = ind_pos[i]
        # print(ih,iw,anchor_id)
        l,w,yaw = anchors[anchor_id]["l"],\
            anchors[anchor_id]["w"],anchors[anchor_id]["yaw"]
        
        # from regression head calculates the offset
        ax, ay = anchor_center_xy[ih,iw,anchor_id]
        dx, dy, dz, dl, dw, dh, dyaw = reg_head[anchor_id*7:(anchor_id+1)*7, ih,iw]

        diag = np.sqrt(l**2+w**2)
        pred_x = ax + dx*diag
        pred_y = ay + dy*diag
        pred_l = np.exp(dl)*l
        pred_w = np.exp(dw)*w
        pred_yaw = yaw + dyaw

        pred_val.append([pred_x,pred_y,pred_l,pred_w,pred_yaw])

    pred_val = np.array(pred_val)

    return get_2d_corners(*[pred_val[:,[k]] for k in range(5)])

from iou import iou_box_array

def nonmax_suppression(bbox_2corners, threshold=0.5):
    # compute pairwise ious
    n = bbox_2corners.shape[0]
    iou_mat = iou_box_array(bbox_2corners, bbox_2corners)

    remaining = set(range(n))
    keep = []
    i = 0

    while i < n and len(remaining)>0:
        if i not in remaining:
            i += 1
            continue
        keep.append(i)
        # remove all boxes with iou > 0.5
        remaining = remaining - set(np.argwhere(iou_mat[i]>threshold).flatten())
        i += 1

    return keep

from dataset_util import get_2d_upper_lower_corners, get_2d_corners
from iou import iou_box_array

def pred_bbox(cls_head, reg_head, anchors, anchor_center_xy,
              prob_threshold=0.95):
    bbox_4corners = gen_bbox(cls_head, reg_head, anchors,
                            anchor_center_xy, prob_threshold=prob_threshold)
    if bbox_4corners is None:
        return None, None

    bbox_2corners = get_2d_upper_lower_corners(bbox_4corners)
    ind = nonmax_suppression(bbox_2corners, 0.1)

    return bbox_4corners[ind], bbox_2corners[ind]

def iou_measure(cls_head, reg_head, gt_boxes, anchors, anchor_center_xy,
                prob_threshold=0.95):

    if gt_boxes.shape[0]==0:
        return 0.0

    bbox_4corners, bbox_2corners = pred_bbox(cls_head, reg_head, anchors,
                                            anchor_center_xy,
                                            prob_threshold=prob_threshold)
    
    if bbox_4corners is None:
        return 0.0

    gt_boxes_ = get_2d_corners(gt_boxes[:,[0]],gt_boxes[:,[1]],gt_boxes[:,[3]],
                               gt_boxes[:,[4]],gt_boxes[:,[6]])
    gt_boxes_ = get_2d_upper_lower_corners(gt_boxes_)

    # compute iou between gt boxes and predicted boxes
    iou_array = iou_box_array(bbox_2corners, gt_boxes_)

    # max over the predicted boxes
    iou_max = np.max(iou_array, axis=0)
    mean_iou = np.mean(iou_max)
    return mean_iou