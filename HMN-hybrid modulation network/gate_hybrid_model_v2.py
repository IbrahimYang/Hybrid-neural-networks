import torch
import torch.nn as nn
import torch.nn.functional as F
channels =512
batch_size  = 200
wins = 10
# v_th_scales = 0.2
v_th_scales = 0.
lens = 0.5
device = 'cuda'

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

probs = 0.0 
act_fun = ActFun.apply


def mem_update(fc,   inputs, spike, mem, v_th,gate):
    state = fc(inputs)
    mem = mem * (1  - spike) + state
    # mem+=state
    temp=mem - v_th
    now_spike = act_fun(temp)*gate
    # now_spike = torch.sigmoid(temp) * gate

    return mem, now_spike.float()


class main_net(nn.Module):
    def __init__(self):
        super(main_net, self).__init__()
        # self.linear_1 = nn.Linear(34*34*2, channels,bias=False)
        # self.linear_2 = nn.Linear(channels, channels,bias=False)
        # self.linear_3 = nn.Linear(channels, 10,bias=False)

        self.linear_1 = nn.Linear(34 * 34 * 2, channels, bias=True)
        self.linear_2 = nn.Linear(channels, channels, bias=True)
        self.linear_3 = nn.Linear(channels, 10, bias=True)

        self.v_th1 = v_th_scales
        self.v_th2 = v_th_scales
        self.v_th3 = v_th_scales

    def forward(self, x, gate):
        h1_mem =  torch.zeros(batch_size, channels, device=device)
        h1_spike =  torch.zeros(batch_size, channels, device=device)
        h1_sumspike = torch.zeros(batch_size, channels, device=device)
        h2_mem = torch.zeros(batch_size, channels, device=device)
        h2_spike =torch.zeros(batch_size, channels, device=device)
        h2_sumspike = torch.zeros(batch_size, channels, device=device)
        h3_mem =  torch.zeros(batch_size, 10, device=device)
        h3_spike = torch.zeros(batch_size, 10, device=device)
        h3_sumspike = torch.zeros(batch_size, 10, device=device)

        for step in range(wins):
            y = x[:,step,:]
            h1_mem, h1_spike = mem_update(self.linear_1, y, h1_spike, h1_mem, self.v_th1,gate[0] )

            h2_mem, h2_spike = mem_update(self.linear_2, h1_spike , h2_spike, h2_mem,self.v_th2, gate[1] )

            h3_mem, h3_spike = mem_update(self.linear_3, h2_spike , h3_spike, h3_mem, self.v_th3,1 )

            h1_sumspike = h1_sumspike + h1_spike
            h2_sumspike = h2_sumspike + h2_spike
            h3_sumspike = h3_sumspike + h3_spike

        outs_1 = h1_sumspike / wins
        outs_2 = h2_sumspike / wins
        outs = h3_sumspike / wins
        return outs,(outs_1,outs_2)

class aux_net(nn.Module):
    def __init__(self):
        super(aux_net, self).__init__()
        self.linear_1 = nn.Linear(34*34*2, channels)
        self.linear_2 = nn.Linear(channels, channels)
        self.linear_3 = nn.Linear(channels, channels)
        self.linear_4 = nn.Linear(channels, channels)

    def forward(self, x):
        scale=100
        temp_1 = self.linear_1(x)
        temp_1 = F.relu(temp_1)

        temp_2 = self.linear_2(temp_1)
        temp_2 = F.relu(temp_2)

        buf=self.linear_3(temp_2)
        # buf = torch.mean(buf, dim=0).unsqueeze(0)
        head1 = F.sigmoid(buf*scale )

        buf = self.linear_4(temp_2)
        # buf = torch.mean(buf, dim=0).unsqueeze(0)
        head2 = F.sigmoid(buf*scale )
        return (head1, head2)

