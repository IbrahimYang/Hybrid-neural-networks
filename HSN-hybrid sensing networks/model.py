import torch
import torch.nn as nn
import torch.nn.functional as F
from snn_model import ResNet2StageSNN


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=False, downsample=None, stride2=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = out[:, :, 1:-1, 1:-1].contiguous()
        return out if not self.last_relu else self.relu(out)


class ResNet2Stage(nn.Module):
    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        self.inplanes = firstchannels
        super(ResNet2Stage, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_layer(channels[0], block_num[0], last_relu=True, stride2=True)
        self.stage2 = self._make_layer(channels[1], block_num[1], last_relu=True, stride2=True)
        self.conv_out = nn.Conv2d(channels[1] * 4, channels[1] * 4, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, planes, blocks, last_relu, stride2=False):
        block = Bottleneck
        downsample = None
        if self.inplanes != planes * block.expansion or stride2:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3,
                          stride=2 if stride2 else 1, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, last_relu=True, downsample=downsample, stride2=stride2)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, last_relu=(last_relu if i == (blocks-1) else True)))

        return nn.Sequential(*layers)

    def step(self, x):
        x = self.conv1(x)  # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)  # stride = 4
        x = self.stage2(x)  # stride = 8
        x = self.conv_out(x)
        return x

    def forward(self, net_in):
        return torch.stack([self.step(net_in[..., step]) for step in range(net_in.shape[-1])], -1)


class TurningDiskSiamFC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aps_net = ResNet2Stage(inchannel=1, block_num=[1, 1])
        self.dvs_net = ResNet2StageSNN(inchannel=2, block_num=[1, 1])

    def corr_up(self, x, k):
        c = torch.nn.functional.conv2d(x, k).unflatten(1, (x.shape[0], k.shape[0]//x.shape[0])).diagonal().permute(3, 0, 1, 2)
        return c

    @staticmethod
    def extract_clip(ff, clip_loc, clip_size):
        bs, fs, h, w = ff.shape
        ch, cw = clip_size

        tenHorizontal = torch.linspace(-1.0, 1.0, cw).expand(1, 1, ch, cw) * cw / w
        tenVertical = torch.linspace(-1.0, 1.0, ch).unsqueeze(-1).expand(1, 1, ch, cw) * ch / h
        tenGrid = torch.cat([tenHorizontal, tenVertical], 1).to(ff.device)

        clip_loc[..., 0] /= w / 2
        clip_loc[..., 1] /= h / 2
        tenDis = clip_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)

        tenGrid = (tenGrid.unsqueeze(1) + tenDis).permute(1, 0, 3, 4, 2)
        target_list = [F.grid_sample(input=ff, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True) for grid in tenGrid]

        return torch.stack(target_list, 1).flatten(end_dim=1)

    @staticmethod
    def gen_gt_cm(target_loc, map_size):
        w, h = map_size

        tenHorizontal = torch.arange(0, w).expand(1, 1, 1, h, w) - w / 2 + 0.5
        tenVertical = torch.arange(0, h).unsqueeze(-1).expand(1, 1, 1, h, w) - h / 2 + 0.5
        tenGrid = torch.stack([tenHorizontal, tenVertical], 2).to(target_loc.device)

        target_loc = target_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
        dist = torch.norm(tenGrid - target_loc, dim=2)
        gt_cm = -1 + (dist < 2) * 1 + (dist < 1) * 1
        return gt_cm.permute(0, 1, 3, 4, 2)

    def get_target_loc(self, cm, img_size):
        iw, ih = img_size
        bs, ns, h, w, ts = cm.shape
        tenHorizontal = (torch.arange(0, w).expand(1, 1, 1, h, w) - w / 2 + 0.5) * 8 + iw / 2
        tenVertical = (torch.arange(0, h).unsqueeze(-1).expand(1, 1, 1, h, w) - h / 2 + 0.5) * 8 + ih / 2
        tenGrid = torch.stack([tenHorizontal, tenVertical], 2).to(cm.device).expand(bs, ns, 2, ts, h, w)
        index = cm.permute(0, 1, 4, 2, 3).flatten(start_dim=-2).argmax(dim=-1, keepdim=True).unsqueeze(2).expand(bs, ns, 2, ts, 1)
        target_loc = tenGrid.flatten(start_dim=-2).gather(dim=-1, index=index).squeeze(dim=-1)
        return target_loc

    def forward(self, aps, dvs, aps_loc, dvs_loc, training=True):
        bs, _, h, w, ts = dvs.shape
        aps_feature = self.aps_net.step(aps)
        dvs_feature = self.dvs_net(dvs)
        kernel = self.extract_clip(aps_feature, aps_loc, (3, 3))
        cm = torch.stack([self.corr_up(dvs_feature[..., t], kernel) for t in range(ts)], -1)

        if training:
            l_reg = 0.1
            gt_cm = self.gen_gt_cm(dvs_loc, cm.shape[2:4])
            loss = - (gt_cm * cm).sum(dim=(2, 3)) + l_reg * torch.pow(cm * (gt_cm != 0), 2).sum(dim=(2, 3))
            loss = loss.mean(dim=(1, 2))
            return {"loss": loss, "cm": cm}
        else:
            pred_loc = self.get_target_loc(cm, aps.shape[-2:][::-1])
            dvs_loc = dvs_loc * 8 + torch.tensor(aps.shape[-2:][::-1]).view(1, 1, 2, 1).to(dvs_loc.device) / 2 - 0.5
            return {"cm": cm, "pred_loc": pred_loc, "aps": aps, "dvs": dvs, "gt_loc": dvs_loc}
