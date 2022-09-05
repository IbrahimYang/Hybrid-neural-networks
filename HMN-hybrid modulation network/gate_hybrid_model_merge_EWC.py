import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
channels =8192*2
batch_size  = 200
wins = 10
v_th_scales = 0.2
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
    temp=mem - v_th
    now_spike = act_fun(temp)*gate
    return mem, now_spike.float()


class main_net(nn.Module):
    def __init__(self):
        super(main_net, self).__init__()
        self.ewc_lambda = 5000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = 5  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0

        self.linear_1 = nn.Linear(34*34*2, channels,bias=False)
        self.linear_2 = nn.Linear(channels, channels,bias=False)
        self.linear_3 = nn.Linear(channels, 10,bias=False)

        # self.v_th1 = nn.Parameter(torch.randn(channels)*v_th_scales)
        # self.v_th2 = nn.Parameter(torch.randn(channels)*v_th_scales)
        # self.v_th3 = nn.Parameter(torch.randn(10)*v_th_scales)

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


    def estimate_fisher(self, dataset,permutted_paramer,gate):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        gate= [torch.FloatTensor(gate[0]).cuda(), torch.FloatTensor(gate[1]).cuda()]
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        # data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
        data_loader = dataset

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            # x = x.to(self._device())
            x = x[:,:, permutted_paramer].cuda()
            outputs = self(x,gate)
            output = outputs[0]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y) == int else y
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=device)

    # def update_omega(self, W, epsilon=1e-8):
    #     '''After completing training on a task, update the per-parameter regularization strength.
    #
    #     [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
    #     [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''
    #
    #     # Loop over all parameters
    #     for n, p in self.named_parameters():
    #         if p.requires_grad:
    #             n = n.replace('.', '__')
    #
    #             # Find/calculate new values for quadratic penalty on parameters
    #             p_prev = getattr(self, '{}_SI_prev_task'.format(n))
    #             p_current = p.detach().clone()
    #             p_change = p_current - p_prev
    #             omega_add = W[n] / (p_change ** 2 + epsilon)
    #             try:
    #                 omega = getattr(self, '{}_SI_omega'.format(n))
    #             except AttributeError:
    #                 omega = p.detach().clone().zero_()
    #             omega_new = omega + omega_add
    #
    #             # Store these new values in the model
    #             self.register_buffer('{}_SI_prev_task'.format(n), p_current)
    #             self.register_buffer('{}_SI_omega'.format(n), omega_new)

class aux_net(nn.Module):
    def __init__(self):
        super(aux_net, self).__init__()
        # self.linear_1 = nn.Linear(34*34*2, channels)
        # self.linear_2 = nn.Linear(channels, channels)
        self.linear_3 = nn.Linear(34*34*2, channels)
        self.linear_4 = nn.Linear(34*34*2, channels)

    def forward(self, x):
        scale=100.
        # temp_1 = self.linear_1(x)
        # temp_1 = F.relu(temp_1)

        # temp_2 = self.linear_2(temp_1)
        # temp_2 = F.relu(temp_2)

        temp_2=x

        buf=self.linear_3(temp_2)
        head1 = F.sigmoid(buf*scale )

        buf = self.linear_4(temp_2)
        head2 = F.sigmoid(buf*scale )
        return (head1, head2)

