from aedat4_dataset import TurningDiskDataset
from model import TurningDiskSiamFC
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time


def train_siamfc():
    epoch_num = 800
    save_period = 2
    load_ckpt_path = ""
    save_ckpt_path = "ckpt/TurningDiskSiamFC_snn.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/train_predictor_{}'.format(int(time.time())))
    net = TurningDiskSiamFC().cuda()
    train_data = DataLoader(TurningDiskDataset(), batch_size=32, pin_memory=True, shuffle=True, num_workers=8)

    if load_ckpt_path:
        net.load_state_dict(torch.load(load_ckpt_path, map_location=torch.device("cuda:0")))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)

    for epoch in tqdm(range(epoch_num)):
        for step, data in enumerate(train_data):
            net_out = net(*[x.cuda() for x in data])
            optimizer.zero_grad()
            loss = net_out['loss'].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            writer.add_scalar('training loss', loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print("\rsaving at epoch {}, loss={:.3f}, path: {}".format(epoch + 1, loss.item(), save_ckpt_path))
            torch.save(net.state_dict(), save_ckpt_path)


def test_siamfc():
    load_ckpt_path = "ckpt/TurningDiskSiamFC_snn.ckpt"
    net = TurningDiskSiamFC().cuda()
    net.load_state_dict(torch.load(load_ckpt_path, map_location=torch.device("cuda:0")))
    train_data = DataLoader(TurningDiskDataset(test=True), batch_size=1, pin_memory=True, shuffle=False)
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (145, 145))

    for step, data in enumerate(train_data):
        aps, dvs, aps_loc, dvs_loc = data
        net_out = net(aps.cuda(), dvs.cuda(), aps_loc.cuda(), dvs_loc.cuda(), training=False)

        pred_loc = net_out['pred_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        gt_loc = net_out['gt_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        aps = aps.squeeze().data.cpu().numpy().astype(np.uint8)
        dvs = dvs.squeeze()[1].data.cpu().numpy().astype(np.uint8)

        for t in range(gt_loc.shape[2]):
            plt.gca().clear()
            img = cv2.cvtColor(aps, cv2.COLOR_GRAY2RGB)
            img[dvs[..., t] != 0, 0] = 255
            for o in range(gt_loc.shape[0]):
                px, py = pred_loc[o, :, t]
                gx, gy = gt_loc[o, :, t]
                img = cv2.rectangle(img, (px-10, py-10), (px+10, py+10), (0, 0, 255), 1)
                img = cv2.rectangle(img, (gx-10, gy-10), (gx+10, gy+10), (0, 255, 0), 1)
                # label = "cross" if o == 0 else "triangle" if o == 1 else "circle"
                # cv2.putText(img, label, (gx + 2, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

            plt.imshow(img)
            plt.pause(0.1)
            video.write(img[:, :, ::-1])
    video.release()


if __name__ == "__main__":
    # train_siamfc()
    test_siamfc()
