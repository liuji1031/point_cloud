import numpy as np
import torch
from torch import nn
from einops import rearrange, pack
from einops.layers.torch import Rearrange, Reduce

class VoxelFeatureExtractionLayer(nn.Module):
    """Implements the VFE layer in VoxelNet

    Args:
        nn (_type_): _description_
    """
    def __init__(self,n_feat_in, n_feat_out,append_aggregate=True):
        super().__init__()

        self.name = "VFELayer"
        self.n_feat_in = n_feat_in
        self.n_feat_out = n_feat_out
        self.append_aggregate = append_aggregate

        if append_aggregate:
            # tries to keep feature length constant
            n_hidden = int(n_feat_out/2)
        else:
            n_hidden = n_feat_out

        self.linear = nn.Linear(n_feat_in, n_hidden, bias=True)
        self.seq = nn.Sequential(
            Rearrange("b p d->b d p"), # such that 2nd dim is feature, for BN1d
            nn.BatchNorm1d(num_features=n_hidden, momentum=0.01),
            Rearrange("b d p-> b p d"), # rearrange back
            nn.ReLU()
        )

        # max over all points in a voxel
        self.maxpool = Reduce("b p d -> b 1 d", "max")

    def forward(self, voxel_batch, mask):
        """compute the initial decorated point cloud, including offset towards
        vox center
        """
        out = self.linear(voxel_batch)

        # apply masking using broadcasting, necessary if allow bias in previous
        # linear layer
        mask_ = rearrange(mask,"nvox npt -> nvox npt 1")
        out = out*mask_

        # apply the rest of the pipeline
        out = self.seq(out)

        # before doing the max pool, set the masked out points to -inf
        out = out + (1.0-mask_)*(-1.0e9)
        out_max:torch.Tensor = self.maxpool(out)

        # print(out.shape, out_max.shape)

        if self.append_aggregate:
            # concatenate over feature dimension
            out,_ = pack([out, out_max.expand_as(out)],"b p *")
            assert(out.shape[-1]==self.n_feat_out)
            return out
        else: # just return aggregated feature
            out_max = rearrange(out_max, "b 1 d->b d") # squeeze
            return out_max

class VoxelFeatureExtraction(nn.Module):

    def __init__(self,
                 n_feat_in,
                 n_hidden_per_layer,
                 n_feat_out):
        super().__init__()

        self.n_feat_in = n_feat_in
        self.n_hidden_per_layer = n_hidden_per_layer
        self.n_feat_out = n_feat_out

        n_hidden_layer = len(n_hidden_per_layer)

        self.layers = []
        n_in = n_feat_in
        for i in range(n_hidden_layer):
            n_out = n_hidden_per_layer[i]
            self.layers.append(VoxelFeatureExtractionLayer(n_in,n_out,
                                                      append_aggregate=True))
            n_in = n_out
        
        # last layer, without appending aggregate
        self.layers.append(VoxelFeatureExtractionLayer(n_in,n_feat_out,
                                                  append_aggregate=False))
        
    def forward(self, voxel_batch, mask):
        x = voxel_batch
        for vfe in self.layers:
            x = vfe(x, mask)
        return x