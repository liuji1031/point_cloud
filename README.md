# Point Cloud Processing for Vehicle Detection

In this project, I aim to first implement VoxelNet to explore some backbone encoding strategies for downstream tasks. VoxelNet depends on Voxel Feature Extraction layers to encode the point cloud, which is first voxelized. Next, within each voxel, the points are decorated with other features such as the delta distance to the average center, before going through Point Pillar style processing such as going through MLP, maxpool over the points, concatenating each point feature with the voxel-level feature. 

## Environment

The training script requires the following: <br />

PyTorch <br />
Spconv  <br />
einops  <br />

