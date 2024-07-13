import torch
from torch import nn
from voxelization import Voxelization
from voxel_feature import VoxelFeatureExtraction
from spatial_conv import SpatialConvolution
from rpn import RegionProposalNet
from loss import VoxelNetLoss

class VoxelNet(nn.Module):
    """define the full voxelnet architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 x_range,
                 y_range,
                 z_range,
                 voxel_size,
                 max_voxel_pts,
                 anchors,
                 ):
        super().__init__()
        self.name="VoxelNet"

        # x: front, y: left, z: up
        self.voxelization = None
        # 4 features in: x,y,z,intensity
        # after decoration, 7 features in
        self.voxel_feature = VoxelFeatureExtraction(n_feat_in=7,
                                                    n_hidden_per_layer=[32,128],
                                                    n_feat_out=128,)
        
        # spatial convolution
        self.scn = SpatialConvolution(n_feat_in=128, to_BEV=True)
        self.scn = torch.comple(self.scn)

        # region proposal network
        self.rpn = RegionProposalNet(nanchor=len(anchors))

        # loss function
        self.loss = VoxelNetLoss()

    def forward(self, input_dict):

        voxel_batch = input_dict["voxel_batch"]
        mask_batch = input_dict["mask_batch"]
        coord_batch = input_dict["coord_batch"]

        

        pos_anchor_id_mask = input_dict["pos_anchor_id_mask"]
        neg_anchor_id_mask = input_dict["neg_anchor_id_mask"]
        reg_target = input_dict["reg_target"]

        # voxelization moved into dataset util
        
        # voxel feature extraction
        # first dim of coord indicates which batch the point comes from
        batch_size = torch.max(coord_batch[:,0]).detach().cpu().numpy().item()+1
        voxel_feat = self.voxel_feature(voxel_batch, mask_batch)

        # spatial convolution
        voxel_feat = self.scn(voxel_feat,
                              coord_batch,
                              self.voxelization.spatial_shape_DHW,
                              batch_size)
        
        # region proposal network
        cls_head, reg_head = self.rpn(voxel_feat)

        print(cls_head.shape, reg_head.shape)

        # calculate loss
        loss = self.loss(cls_head, reg_head,
                         pos_anchor_id_mask, neg_anchor_id_mask, reg_target)
        
        return (cls_head, reg_head), loss
