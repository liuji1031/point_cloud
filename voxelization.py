import numpy as np
import numba
from einops import rearrange, pack, reduce

@numba.njit
def _voxelization(unique_coord, inverse_ind, counts, pc, voxel,
                      max_voxel_pts):
    # keep_ind = []
    for i_vox in range(unique_coord.shape[0]):
        ind = np.argwhere(inverse_ind==i_vox).flatten()

        if counts[i_vox] > max_voxel_pts:
            # subsample
            ind = np.random.choice(ind,max_voxel_pts,replace=False)
            counts[i_vox] = max_voxel_pts

        # keep_ind.append(ind)
        voxel[i_vox, :counts[i_vox],:] = pc[ind,:]

class Voxelization():
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
                 init_decoration=True,
                 ):
        super().__init__()

        self.name = "Voxelization"

        # if True, append diff to vox center
        self.init_decoration = init_decoration

        dx,dy,dz = voxel_size

        nx = int(np.round((x_max-x_min)/dx))
        ny = int(np.round((y_max-y_min)/dy))
        nz = int(np.round((z_max-z_min)/dz))

        self.spatial_shape_WHD = [nx,ny,nz]
        self.spatial_shape_DHW = self.spatial_shape_WHD[::-1]

        # make the range integer copy of the interval
        x_max = x_min + nx*dx
        y_max = y_min + ny*dy
        z_max = z_min + nz*dz

        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.z_range = [z_min, z_max]
        self.voxel_size_DHW = voxel_size[::-1] # dx,dy,dz
        self.max_voxel_pts = max_voxel_pts

        # expand axis
        self.lb_DHW = rearrange(np.array([ z_min,y_min,x_min]),
                                "d->1 d")
        self.vox_sz_DHW = rearrange(np.array(self.voxel_size_DHW),
                                    "d->1 d")

    def process_single_batch_pc(self, pc:np.ndarray, ibatch):
        # exclude out of range points
        x,y,z = pc[:,0],pc[:,1],pc[:,2]
        n_feat = pc.shape[-1] # feature dim

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        keep = (x>=x_min) & (x<x_max) & (y>=y_min) & (y<y_max) & \
        (z>=z_min) & (z<z_max)
        keep = np.argwhere(keep).squeeze()

        pc = pc[keep,:]
        # rearrange to DHW
        pc[:,:3] = pc[:,[2,1,0]]
        coord = ((pc[:,:3] - self.lb_DHW)/self.vox_sz_DHW).astype(np.int32)

        unique_coord, inverse_ind, counts = np.unique(coord,
                                              return_inverse=True,
                                              return_counts=True,
                                              axis=0)
        
        # inverse_ind maps from coord to unique_coord, in other words, signifies
        # which coord corresponds to each point in point cloud
        
        # construct voxel
        n_vox = unique_coord.shape[0]
        voxel = np.zeros((n_vox,self.max_voxel_pts,n_feat))
        
        _voxelization(unique_coord, inverse_ind, counts, pc, voxel,
                      self.max_voxel_pts)
        # keep_ind,_ = pack(keep_ind, "*")

        # generate masking for voxel entries not populated by points by
        # broadcasting
        mask = rearrange(np.arange(self.max_voxel_pts),"d -> 1 d") < \
            rearrange(counts,"d->d 1")
        mask = mask.astype(pc.dtype)
        
        # calculate voxel center
        vox_center = unique_coord*self.vox_sz_DHW + \
            (self.lb_DHW+self.vox_sz_DHW/2)

        # add batch number; after shape: nv by 4, i.e. (ibatch, ix, iy, iz)
        unique_coord = np.pad(unique_coord,pad_width=((0,0),(1,0)),
                              mode="constant",constant_values=ibatch)
        # unique_coord.to(torch.int32)
        
        if self.init_decoration:
            # compute diff with vox center and append as feature
            pt_center = reduce(voxel[:,:,:3],"nvox npt d -> nvox 1 d","sum")/ \
            rearrange(counts, "nvox -> nvox 1 1")
            diff = (voxel[:,:,:3]-pt_center)*\
                rearrange(mask,"nvox npt -> nvox npt 1")
            voxel,_ = pack([voxel, diff],"nvox npt *")

        return voxel, unique_coord, mask, vox_center

    def __call__(self, point_cloud):
        """parse point cloud into voxels

        Args:
            point_cloud (_type_): a list (batched) of point cloud data or one 
        """

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
        voxel_batch = np.concatenate(voxel_batch,axis=0)
        coord_batch = np.concatenate(coord_batch,axis=0)
        mask_batch = np.concatenate(mask_batch, axis=0)
        vox_center_batch = np.concatenate(vox_center_batch, axis=0)

        return voxel_batch, coord_batch, mask_batch, vox_center_batch