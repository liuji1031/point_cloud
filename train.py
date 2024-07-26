# %%
# %%
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from easydict import EasyDict as edict
from nuscenes.nuscenes import NuScenes
from dataset_util import get_database, get_dataloader
from voxelization import Voxelization
from voxelnet import VoxelNet
from config import cfg

torch.set_default_dtype(torch.float32)

logger.info(str(cfg.anchors))
assert(len(cfg.anchors) > 2)

# retrieve the database
nd_train = get_database(mode="train")
nd_val = get_database(mode="val")

train_dataloader = get_dataloader(nd_train, batch_size=8, shuffle=True)
val_dataloader = get_dataloader(nd_val, batch_size=8, shuffle=True)

# %%
vnet = VoxelNet(canvas_spatial_shape_DHW=cfg.voxelizer.spatial_shape_DHW,
                anchors=cfg.anchors)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
vnet.to(device)

# %%
import time
from datetime import datetime
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from dataset_util import to_device
from voxelnet import iou_measure

# training loop
def train(n_epoch, vnet, train_dataloader, val_dataloader, chkpt=None,
          summary_path=None, device="cuda:0", raise_oom=False):

    optimizer = optim.Adam(vnet.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.1, patience=2,
                                               threshold=1e-2,
                                               min_lr=1e-6)
    
    start_epoch = 0
    best_val_loss = np.inf
    if chkpt is not None:
        logger.info(f"loading checkpoint from {chkpt}")
        chkpt_states = torch.load(chkpt)
        if "vnet" not in chkpt_states:
            vnet.load_state_dict(torch.load(chkpt))
        else:
            vnet.load_state_dict(chkpt_states["vnet"])
            optimizer.load_state_dict(chkpt_states["optimizer"])
            scheduler.load_state_dict(chkpt_states["scheduler"])
            start_epoch = chkpt_states["epoch"]+1

            logger.info(f"resuming from epoch {start_epoch}")
            
    logger.info(f"current learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(f"current regularization strength: {optimizer.param_groups[0]['weight_decay']}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if summary_path is None:
        writer = SummaryWriter(f"training_logs/{timestamp}")
    else:
        writer = SummaryWriter(summary_path)
    
    for epoch in range(start_epoch, start_epoch+n_epoch):
        
        vnet.train(True)
        loss_train = []
        batch_time = []
        t0 = time.time_ns()

        pbar = tqdm(iter(train_dataloader), total=len(train_dataloader),
                    desc=f"Epoch {epoch}, training",ncols=100)
        for input in pbar:
            input = to_device(input, device)
            optimizer.zero_grad(set_to_none=True)
            
            mdl_output, loss = vnet(input)
            try:
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    logger.info('| WARNING: ran out of memory, skipping batch')
                    for p in vnet.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    raise e
            t1 = time.time_ns()
            optimizer.step()

            loss_train.append(loss.item())
            batch_time.append((t1-t0)/1e9)
            t0 = t1

            pbar.set_postfix({"loss":f"{np.mean(loss_train[-20:]):.2f}",
                "mini batch time":f"{np.mean(batch_time[-20:]):.2f} sec"})
            
        avg_train_loss = np.mean(loss_train)
        scheduler.step(avg_train_loss)

        # evaluate on validation set
        vnet.eval()
        loss_val = []
        iou_val = []
        pbar = tqdm(iter(val_dataloader), total=len(val_dataloader),
                          desc=f"Epoch {epoch}, validation",ncols=100)
        for input in pbar:
            with torch.no_grad():
                input = to_device(input, device)
                mdl_output, loss = vnet(input)
                for ibatch in range(mdl_output[0].shape[0]):
                    cls_head = mdl_output[0][ibatch]
                    reg_head = mdl_output[1][ibatch]
                    iou = iou_measure(cls_head,reg_head,
                                      input["gt_boxes"][ibatch],
                                      cfg.anchors,
                                      nd_train.anchor_center_xy)
                    iou_val.append(iou)
                    # writer.add_scalar("iou", iou, epoch)
                loss_val.append(loss.item())
            
            pbar.set_postfix({"loss":f"{np.mean(loss_val):.2f}",
                              "iou":f"{np.mean(iou_val):.3f}"}
                              )
        avg_val_loss = np.mean(loss_val)

        writer.add_scalars("loss", {"train":avg_train_loss,
                                   "validation":avg_val_loss}, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"],
                          epoch)
        writer.add_scalar("iou", np.mean(iou_val),
                          epoch)
        writer.flush()

        states_dict = {"vnet":vnet.state_dict(),
                          "optimizer":optimizer.state_dict(),
                          "scheduler":scheduler.state_dict(),
                          "epoch":epoch}
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"saving best model at epoch {epoch}")
            torch.save(states_dict, f"chkpt/best_model_{timestamp}.pth")
        
        # save every epoch
        torch.save(states_dict, f"chkpt/last_model_{timestamp}.pth")


# %%
# train(50, vnet, train_dataloader, val_dataloader)

train(50, vnet, train_dataloader, val_dataloader,
      chkpt="chkpt/last_model_20240723_121502.pth",
      summary_path="training_logs/20240723_121502",)

