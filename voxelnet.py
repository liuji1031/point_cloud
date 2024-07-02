import numpy as np
from typing import List
import torch
from torch import nn
from einops import rearrange, pack, reduce

class Voxelization(nn.Module):
    """implement a module for voxelizaing point cloud data

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 x_min,x_max,
                 y_min,y_max,
                 z_min,z_max,
                 voxel_size,
                 max_voxel_pts,
                 init_decoration=True
                 ):
        super().__init__()

        # if True, append diff to vox center
        self.init_decoration = init_decoration

        dx,dy,dz = voxel_size

        nx = int((x_max-x_min)//dx)
        ny = int((y_max-y_min)//dy)
        nz = int((z_max-z_min)//dz)

        # make the range integer copy of the interval
        x_max = x_min + nx*dx
        y_max = y_min + ny*dy
        z_max = z_min + nz*dz

        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.z_range = [z_min, z_max]
        self.voxel_size = voxel_size
        self.max_voxel_pts = max_voxel_pts

        # expand axis
        self.lb = rearrange(torch.tensor([x_min,y_min,z_min]),"d->1 d")
        self.vox_sz = rearrange(torch.tensor(self.voxel_size),"d->1 d")

    def process_single_batch_pc(self, pc:torch.Tensor, ibatch):
        # exclude out of range points
        x,y,z = pc[:,0],pc[:,1],pc[:,2]
        n_feat = pc.shape[-1] # feature dim

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        keep = (x>=x_min) & (x<x_max) & (y>=y_min) & (y<y_max) & \
        (z>=z_min) & (z<z_max)
        keep = torch.argwhere(keep).squeeze()

        pc = pc[keep,:]
        coord = ((pc[:,:3] - self.lb)/self.vox_sz).to(torch.int64)
        coord : torch.Tensor

        unique_coord, inverse_ind, counts = coord.unique(
                                              sorted=True,
                                              return_inverse=True,
                                              return_counts=True,
                                              dim=0)
        
        # inverse_ind maps from coord to unique_coord, in other words, signifies
        # which coord corresponds to each point in point cloud
        
        # construct voxel
        n_vox = unique_coord.shape[0]
        voxel = torch.zeros(n_vox,self.max_voxel_pts,n_feat,dtype=pc.dtype)
        
        unique_coord : torch.Tensor
        # keep_ind = []
        for i_vox in range(unique_coord.shape[0]):
            ind = torch.argwhere(inverse_ind==i_vox).numpy().flatten()

            if counts[i_vox] > self.max_voxel_pts:
                # subsample
                ind = np.random.choice(ind,self.max_voxel_pts,replace=False)
                counts[i_vox] = self.max_voxel_pts

            # keep_ind.append(ind)
            voxel[i_vox, :counts[i_vox],:] = pc[ind,:]

        # keep_ind,_ = pack(keep_ind, "*")

        # generate masking for voxel entries not populated by points by
        # broadcasting
        mask = rearrange(torch.arange(self.max_voxel_pts),"d -> 1 d") < \
            rearrange(counts,"d->d 1")
        mask.to(pc.dtype)
        
        # add axis at feature dimension
        mask_ = rearrange(mask,"nvox npt -> nvox npt 1")

        # calculate voxel center
        vox_center = unique_coord*self.vox_sz + (self.lb + self.vox_sz/2)

        # add batch number; after shape: nv by 4, i.e. (ibatch, ix, iy, iz)
        unique_coord = nn.functional.pad(unique_coord,(1,0),mode="constant",
                                         value=ibatch)
        
        if self.init_decoration:
            # compute diff with vox center and append as feature
            pt_center = reduce(voxel[:,:,:3],"nvox npt d -> nvox 1 d","sum")/ \
            rearrange(counts, "nvox -> nvox 1 1")
            diff = (voxel - pt_center)*mask_
            voxel,_ = pack([voxel, diff],"nvox npt *")

        return voxel, unique_coord, mask, vox_center

    @torch.no_grad()
    def forward(self, point_cloud):
        """parse point cloud into voxels

        Args:
            point_cloud (_type_): a list (batched) of point cloud data or one 
            batch (tensor of shape NxD)
        """

        batched = isinstance(point_cloud, list)

        if not batched:
            point_cloud = [point_cloud]
        
        voxel_batch = []
        coord_batch = []
        mask_batch = []
        vox_center_batch = []

        for ibatch, pc in enumerate(point_cloud):
            voxel, coord, mask, vox_center = \
                self.process_single_batch_pc(pc, ibatch)
            
            # append all output
            voxel_batch.append(voxel)
            coord_batch.append(coord)
            mask_batch.append(mask)
            vox_center_batch.append(vox_center)

        # concatenate along the batch dimension
        voxel_batch,_ = pack(voxel_batch,"* np m")
        coord_batch,_ = pack(coord_batch,"* np n")
        mask_batch,_ = pack(mask_batch, "* np k")
        vox_center_batch,_ = pack(vox_center_batch, "* np i")

        return voxel_batch, coord_batch, mask_batch, vox_center_batch

class VoxelFeatureExtractionLayer(nn.Module):
    def __init__(self,):
        super().__init__()



    def forward(self, pc_batch, ):
        """compute the initial decorated point cloud, including offset towards
        vox center
        """

        pass

