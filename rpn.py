# implement the region proposal net from voxelnet paper

import torch
from torch import nn, jit
from einops import pack

class RegionProposalNet(jit.ScriptModule):
    """implement the region proposal network from the voxelnet paper

    Args:
        nn (_type_): _description_
    """
    def __init__(self,*args,nanchor=2, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.name = "RPN"

        def conv_block(n_in, n_out, kernel_size, stride, padding, repeat=1,
                       add_bn=True, add_relu=True):
            layers = []
            for _ in range(repeat):
                layers.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding))
                
                if add_bn:
                    layers.append(nn.BatchNorm2d(n_out))
                
                if add_relu:
                    layers.append(nn.ReLU())
                
            return layers
        
        def deconv_block(n_in, n_out, kernel_size, stride, padding, repeat=1,
                         add_bn=True, add_relu=True):
            layers = []
            for _ in range(repeat):
                
                layers.append(nn.ConvTranspose2d(n_in, n_out,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding))

                if add_bn:
                    layers.append(nn.BatchNorm2d(n_out))

                if add_relu:
                    layers.append(nn.ReLU())

            return layers

        self.nanchor = nanchor
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

        head1_layers =  conv_block(768, 128, 3, 1, 1,repeat=1) + \
                        conv_block(128, 128, 1, 1, 0,repeat=2) + \
                        conv_block(128, nanchor, 1, 1, 0,repeat=1,
                                               add_bn=False,
                                               add_relu=False)
        self.head1 = nn.Sequential(*head1_layers)

        head2_layers = conv_block(768, 128, 3, 1, 1,repeat=1) + \
                        conv_block(128, 128, 1, 1, 0,repeat=2) + \
                        conv_block(128, 7*nanchor, 1, 1, 0,repeat=1,
                                               add_bn=False,
                                               add_relu=False)
        self.head2 = nn.Sequential(*head2_layers)

    @jit.script_method
    def forward(self, x):

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # deconvolution
        x1_ = self.deconv1(x1)
        x2_ = self.deconv2(x2)
        x3_ = self.deconv3(x3)

        # concatenate
        # equivalent einops code:
        # x,_ = pack([x1_,x2_,x3_],"b * h w")
        x = torch.cat((x1_,x2_,x3_),dim=1)

        cls = self.head1(x)
        # apply sigmoid on cls
        cls = torch.sigmoid(cls)

        reg = self.head2(x)
        # for every pos anchor, we have 7 values, which are: dx dy dz dl dw dh 
        # dyaw. These can be positive or negative, thus no relu

        return cls, reg
        
