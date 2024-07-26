import numpy as np
from easydict import EasyDict as edict
from voxelization import Voxelization

cfg = edict()
cfg.x_range = [-40.0,40.0]
cfg.y_range = [-40.,40.]
cfg.z_range = [-1.,3.]
cfg.voxel_size = {'x':0.4, 'y':0.4, 'z':0.4}
cfg.max_voxel_pts = 35
cfg.max_points_per_cloud = 25000
cfg.pos_neg_iou_thresh = [0.6,0.45]
# cfg.max_voxel_num = 8000

def get_canvas_size(rng, res):
    return int(np.round((rng[1]-rng[0])/res))

cfg.canvas_size = {"h":get_canvas_size(cfg.y_range, 2*cfg.voxel_size["y"]),
                   "w":get_canvas_size(cfg.x_range, 2*cfg.voxel_size["x"])}
cfg.canvas_res = {"w":2*cfg.voxel_size["x"], "h":2*cfg.voxel_size["y"]}
cfg.canvas_min = {"h":cfg.y_range[0], "w":cfg.x_range[0]}

cfg.anchors = [{"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":0.0},
               {"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":np.pi*0.25},
               {"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":np.pi*0.5},
               {"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":np.pi*0.75},]

# cfg.anchors = [{"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":0.0},
#                {"z":0.,"l":3.9,"w":1.6,"h":1.56,"yaw":np.pi*0.5}]

cfg.voxelizer = Voxelization(x_range=cfg.x_range,y_range=cfg.y_range,
                        z_range=cfg.z_range,
                        voxel_size=cfg.voxel_size,
                        max_voxel_pts=cfg.max_voxel_pts,
                        # max_voxel_num=cfg.max_voxel_num,
                        init_decoration=True)

cfg.train_val_test_split = [0.9,0.05,0.05]
cfg.lidar_path = "/media/ji/volume2/PointCloud/dataset_lidar_gtbox"

cfg.aug_angles = {"train":[0,-5,5,-10,10],
                  "val":None,
                  "test":None}
