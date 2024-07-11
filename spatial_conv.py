import torch
from einops import rearrange
from torch import nn, jit
import spconv.pytorch as spconv


class SpatialConvolution(jit.ScriptModule):
    """implement some 3D spatial convolution layers as in the original voxelnet
    paper

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 n_feat_in,
                 to_BEV=True):
        super().__init__()

        # self.conv1 = spconv.SparseConv3d(n_feat_in, 64,kernel_size=3,
        #                                  stride=[2,1,1],padding=[1,1,1],
        #                                  indice_key="sp1",
        #                                  algo=spconv.ConvAlgo.Native)
        
        # self.conv2 = spconv.SparseConv3d(64, 64,kernel_size=3,
        #                                  stride=[1,1,1],padding=[0,1,1],
        #                                  indice_key="sp2",
        #                                  algo=spconv.ConvAlgo.Native)
        
        # self.conv3 = spconv.SparseConv3d(64, 64,kernel_size=3,
        #                                  stride=[2,1,1],padding=[1,1,1],
        #                                  indice_key="sp3",
        #                                  algo=spconv.ConvAlgo.Native)
        
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(n_feat_in, 64,kernel_size=3,
                                         stride=[2,1,1],padding=[1,1,1],
                                         indice_key="sp1",
                                         algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64,kernel_size=3,
                                         stride=[1,1,1],padding=[0,1,1],
                                         indice_key="sp2",
                                         algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64,kernel_size=3,
                                         stride=[2,1,1],padding=[1,1,1],
                                         indice_key="sp3",
                                         algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.to_BEV = to_BEV
                                         
    @jit.script_method
    def forward(self, feature, coords, spatial_shape, batch_size):

        input_sp = spconv.SparseConvTensor(feature, coords,
                                           spatial_shape, batch_size)

        # x = self.conv1(input_sp)
        # print(x.dense().shape)
        
        # x = self.conv2(x)
        # print(x.dense().shape)
        # x = self.conv3(x)
        # print(x.dense().shape)

        x = self.net(input_sp)

        if self.to_BEV:
            # concatenate channel and depth dimension
            x : spconv.SparseConvTensor
            x = rearrange(x.dense(),"b c d h w -> b (c d) h w")
        return x