# implement the region proposal net from voxelnet paper

import torch
from torch import nn
from einops import pack

class RPN(nn.Module):
    """implement the region proposal network from the voxelnet paper

    Args:
        nn (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.name = "RPN"

        def conv_block(n_in, n_out, kernel_size, stride, padding, repeat=1):
            layers = []
            for _ in range(repeat):
                layers+=[
                nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(n_out),
                nn.ReLU()
                ]
            return layers
        
        def deconv_block(n_in, n_out, kernel_size, stride, padding, repeat=1):
            layers = []
            for _ in range(repeat):
                layers+=[
                nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(n_out),
                nn.ReLU()
                ]
            return layers

        self.block1 = nn.Sequential(*conv_block(128, 128, 3, 2, 1,repeat=1),
                                    *conv_block(128, 128, 3, 1, 1,repeat=3)
                                )
        
        self.block2 = nn.Sequential(*conv_block(128, 128, 3, 2, 1,repeat=1),
                                    *conv_block(128, 128, 3, 1, 1,repeat=5)
                                )
        
        self.block3 = nn.Sequential(*conv_block(128, 256, 3, 2, 1,repeat=1),
                                    *conv_block(256, 256, 3, 1, 1,repeat=5)
                                )
        
        self.deconv1 = nn.Sequential(*deconv_block(128, 256, 3, 1, 1,repeat=1))
        self.deconv2 = nn.Sequential(*deconv_block(128, 256, 2, 2, 0,repeat=1))
        self.deconv3 = nn.Sequential(*deconv_block(256, 256, 4, 4, 0,repeat=1))

        self.head1 = nn.Sequential(*conv_block(768, 2, 1, 1, 0,repeat=1))
        self.head2 = nn.Sequential(*conv_block(768, 14, 1, 1, 0,repeat=1))

    def forward(self, x):

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        print(x1.shape, x2.shape, x3.shape)

        # deconvolution
        x1_ = self.deconv1(x1)
        x2_ = self.deconv2(x2)
        x3_ = self.deconv3(x3)

        print(x1_.shape, x2_.shape, x3_.shape)

        # concatenate
        x,_ = pack([x1_,x2_,x3_],"b * h w")

        cls = self.head1(x)
        reg = self.head2(x)

        return cls, reg
        
