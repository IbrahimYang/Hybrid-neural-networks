import torch
import torch.nn as nn
import numpy as np

update_v = 'default'
expansion = 4


class ActFun(torch.autograd.Function):
    """A Heaviside step function that is made differentiable with surrogate gradient"""
    lens = 0.5

    @staticmethod
    def forward(ctx, _input):
        ctx.save_for_backward(_input)
        return _input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        _input, = ctx.saved_tensors
        return grad_output * _input.abs().lt(ActFun.lens).float()


class SNNLayer(nn.Module):
    def __init__(self, layer, bn=True, thresh=None, thresh_grad=True, decay=0.0, decay_grad=False, bypass_in=False, update_v=update_v):
        super(SNNLayer, self).__init__()
        self.layer = layer
        self.state = [0., 0.]  # [mem, spike]

        if thresh is None:
            thresh = 0.0 if update_v == 'rnn' else 0.5

        self.thresh = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * thresh, requires_grad=thresh_grad)
        self.decay = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * decay, requires_grad=decay_grad)

        self.bn = nn.BatchNorm2d(layer.out_channels) if bn else None
        self.bypass_bn = nn.BatchNorm2d(layer.out_channels) if bn and bypass_in else None

        if bn and thresh:
            self.bn.weight.data = self.thresh.data.view(-1) / (2**0.5 if bypass_in else 1)
            if bypass_in:
                self.bypass_bn.weight.data = self.thresh.data.view(-1) / 2**0.5

        self.act_func = nn.ReLU(inplace=False) if update_v == 'rnn' else ActFun.apply
        self.update_v = update_v

    def update_state(self, x, bypass_in):

        layer_in = self.bn(self.layer(x)) if self.bn is not None else self.layer(x)
        if bypass_in is not None:
            layer_in += self.bypass_bn(bypass_in) if self.bn is not None else bypass_in

        if self.update_v == 'default':
            self.state[0] = self.state[0] * (1. - self.state[1]) * self.decay + layer_in
        elif self.update_v == 'bursting':
            self.state[0] = self.state[0] * self.decay - self.state[1] * self.thresh + layer_in
        elif self.update_v == 'rnn':
            self.state[0] = self.state[0] * self.decay + layer_in

        self.state[1] = self.act_func(self.state[0] - self.thresh)

    def reset_state(self, history):
        self.state = [self.state[0].detach(), self.state[1].detach()] if history else [0., 0.]
        # self.thresh.data = self.thresh.clamp(min=-1., max=1.).data
        self.decay.data = self.decay.clamp(min=0., max=1.).data

    def forward(self, x, bypass_in=None):
        self.update_state(x, bypass_in)
        return self.state[1]


class BottleneckSNN(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, stride2=False):
        super().__init__()

        self.layers = nn.Sequential(
            SNNLayer(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)),
            SNNLayer(nn.Conv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)),
            # SNNLayer(nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False))
        )
        self.residual_layer = SNNLayer(nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False), bypass_in=True)

        self.downsample = downsample

    def reset_state(self, history):
        for layer in self.layers:
            layer.reset_state(history)
        self.residual_layer.reset_state(history)

    def forward(self, x):
        out = self.layers(x)
        residual = self.downsample(x) if self.downsample is not None else x

        out = self.residual_layer(out, residual)
        out = out[:, :, 1:-1, 1:-1].contiguous()
        return out


class ResNet2StageSNN(nn.Module):
    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        super().__init__()

        self.layers = nn.Sequential(
            SNNLayer(nn.Conv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)),
            *self._make_layer(firstchannels, channels[0], block_num[0], stride2=True),
            *self._make_layer(channels[0] * expansion, channels[1], block_num[1], stride2=True),
            SNNLayer(nn.Conv2d(channels[1] * expansion, channels[1] * expansion, kernel_size=1, bias=False),
                     decay=0., decay_grad=False, update_v='rnn', bn=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
        # self.layers[-1].layer.weight.data = torch.zeros_like(self.layers[-1].layer.weight.data)

    def _make_layer(self, inplanes, planes, blocks, stride2=False):

        downsample = nn.Conv2d(inplanes, planes * expansion, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)

        layers = [BottleneckSNN(inplanes, planes, downsample=downsample, stride2=stride2)]
        layers += [BottleneckSNN(planes * expansion, planes) for i in range(1, blocks)]

        return layers

    def reset_state(self, history=False):
        for layer in self.layers:
            layer.reset_state(history)

    def step(self, x):
        self.layers(x)
        out = self.layers[-1].state[0]
        return out

    def forward(self, net_in):
        self.reset_state()
        out_list = []
        for t in range(net_in.shape[-1]):
            net_out = self.step(net_in[..., t])
            out_list.append(net_out)
        return torch.stack(out_list, -1)
