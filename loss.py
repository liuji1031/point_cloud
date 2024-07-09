import torch
from torch import nn
from einops import rearrange, repeat

class VoxelNetLoss(nn.Module):
    def __init__(self, 
                 pos_cls_weight=1.5,
                 neg_cls_weight=1.0,

                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.name = "VoxelNetLoss"
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
    
    def forward(self, cls_head, reg_head,
                pos_anchor_id_mask, neg_anchor_id_mask, reg_target):
        
        # cls_head shape: (b, n_anchor,h,w) with n_anchor=2 in paper
        # reg_head shape: (b, n_anchor*7,h,w) with n_anchor=2 in paper
        # pos_anchor_id_mask shape: (b,h,w,n_anchor)
        # neg_anchor_id_mask shape: (b,h,w,n_anchor)
        # reg_target shape: (b, h,w,n_anchor*7)

        cls_head = rearrange(cls_head, "b na h w -> b h w na")
        reg_head = rearrange(reg_head, "b na7 h w -> b h w na7")

        # find the cross entropy loss for positive anchors
        n_pos = torch.sum(pos_anchor_id_mask)+1e-6
        loss_cls_pos = (-torch.log(cls_head+1e-6))*pos_anchor_id_mask
        loss_cls_pos = torch.sum(loss_cls_pos)/n_pos

        # find the cross entropy loss for negaitve anchors
        n_neg = torch.sum(neg_anchor_id_mask)+1e-6
        loss_cls_neg = (-torch.log(1-cls_head+1e-6))*neg_anchor_id_mask
        loss_cls_neg = torch.sum(loss_cls_neg)/n_neg

        # find the smooth L1 loss for positive anchors
        pos_mask = repeat(pos_anchor_id_mask, "b h w na -> b h w (na d)", d=7)
        loss_smooth_l1 = nn.functional.smooth_l1_loss(reg_head*pos_mask,
                                           reg_target*pos_mask,
                                           reduction="sum")
        loss_smooth_l1 = loss_smooth_l1 / n_pos
        
        total_loss = self.pos_cls_weight*loss_cls_pos + \
                        self.neg_cls_weight*loss_cls_neg + \
                        loss_smooth_l1
        
        return total_loss