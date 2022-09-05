import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse
import utils
import torch
import matplotlib.pyplot as plt
import torch.functional as F
from gate_hybrid_model_merge_EWC import *
from tqdm import tqdm
import mat_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr_s1', default=0.02, type=float, help='learning rate')
parser.add_argument('--lr_s2', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch_size', default=batch_size, type=int)
parser.add_argument('--task_permutation_up', default=400, type=int)
parser.add_argument('--task_permutation_down', default=300, type=int)
parser.add_argument('--task_times', default=699, type=int)
parser.add_argument('--task_per_group', default=4, type=int)
parser.add_argument('--test_task_num', default=40, type=int)
parser.add_argument('--stage_1_percentage', default=0.5, type=float)
parser.add_argument('--stage_2_epoch_per_task', default=10, type=int)
# parser.add_argument('--stage_2_index_permute', default=False, type=bool)
parser.add_argument('--xdg', action="store_true")
parser.add_argument('--gate_thr_xdg', default=-1, type=float)
parser.add_argument('--ewc',action="store_true")
parser.add_argument('--stage_1_epoch', default=30, type=int)
parser.add_argument('--inputs_size', default=34*34*2, type=int)
parser.add_argument('--root_path', default='test', type=str)
parser.add_argument('--seed', default=430, type=int)
parser.add_argument('--gpu', default=4, type=int)
args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

def gate_manual(task_num):
    gate_list=[]
    for i in range(task_num):
        temp=np.random.rand(channels*2)
        temp=temp>args.gate_thr_xdg
        temp=temp.astype(np.float)
        gate_list.append([temp[:channels],temp[channels:]])
    return np.array(gate_list)

gate_manual_mm=gate_manual(args.test_task_num)

def create_folders():
    if (os.path.exists(args.root_path)):
        print('alreadly exist')
    else:
        os.mkdir(args.root_path)
        os.mkdir(args.root_path + '/models')

def permute_index(item):
    p_num=int(np.random.rand()*(args.task_permutation_up-args.task_permutation_down))+args.task_permutation_down
    start=int(np.random.rand()*(args.inputs_size-p_num))
    index=item.copy()
    result=index[:start]+ np.random.permutation(index[start:start+p_num]).tolist()+index[start+p_num:]
    return result

def get_permutation_index(times):
    ori_index = [i for i in range(args.inputs_size)]
    ori_index=np.random.permutation(ori_index).tolist()
    index=[ori_index]
    for i in range(times):
        if((i+1)%args.task_per_group==0):
            temp=np.random.permutation(index[-1].copy()).tolist()
        else:
            temp=index[-1]
        index.append(permute_index(temp))
    index = np.array(index)
    print('task num:{}'.format(index.shape[0]))
    np.save(args.root_path + '/p_index.npy', index)
    similarity_matrix=utils.show_task_similarity(index)
    np.save(args.root_path + '/p_similarity.npy', similarity_matrix)
    return index

def get_net():
    net = main_net().cuda()
    return net

def get_gate_net():
    net = aux_net().cuda()
    return net

def get_data_set():
    trainset = mat_loader.dataset_from_mat_dvs('data/npy/train',wins)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset = mat_loader.dataset_from_mat_dvs('data/npy/test',wins)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    return trainloader, testloader

def train_ann_task(net, gate_net, data_set, permutation_index_list):
    def get_train_index():
        start_index=len(permutation_index_list)-args.test_task_num
        result_1=[i for i in range(start_index)]
        result_2=[start_index+int(args.task_per_group* group_index + inner_group_index+1)
                  for group_index in range(int( args.test_task_num/args.task_per_group))
                  for inner_group_index in range(int(args.task_per_group * args.stage_1_percentage))]
        result=result_1+result_2
        print('train index:',result_2)
        return result
    train_index=get_train_index()
    optimizer_aux = optim.SGD(gate_net.parameters(), lr=args.lr_s1)
    print('stage_1:-------------------')
    for i in range(args.stage_1_epoch):
        train_ann_epoch(net, gate_net, i, data_set[0], permutation_index_list, optimizer_aux,train_index)
    acc,test_gate=test_task(net, gate_net, data_set[1], permutation_index_list[-args.test_task_num:,:], '/test_gate.npy',None,True,0)
    test_gate = test_gate.astype(np.bool)
    test_gate=[[test_gate[i,0,:channels],test_gate[i,0,channels:]] for i in range(args.test_task_num)]
    test_gate=np.array(test_gate)
    return test_gate

def train_ann_epoch(net,gate_net,epoch,train_set,permutation_index_list,optimizer_aux,train_index):
    net.train()
    gate_net.train()
    def adjust_lr(epoch):
        lr = args.lr_s1 * (0.1 ** (epoch //25))
        return lr
    lr = adjust_lr(epoch)
    for p in optimizer_aux.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f' % (epoch, lr))
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_set):
        inputs, targets = inputs.cuda(), targets.cuda()

        task_index_1 = int(np.random.rand()*len(train_index))
        task_index_1=train_index[task_index_1]

        task_index_2 = int(np.random.rand() * len(train_index))
        task_index_2 = train_index[task_index_2]

        inputs=inputs.mean(dim=1,keepdim=False)

        inputs_1 = inputs[:, permutation_index_list[task_index_1]]
        inputs_2 = inputs[:,permutation_index_list[task_index_2]]

        optimizer_aux.zero_grad()
        gate_1 = gate_net(inputs_1)
        gate_2 = gate_net(inputs_2)

        regularizer_1=utils.get_conjugate_loss(gate_1[0],gate_2[0],permutation_index_list[task_index_1],permutation_index_list[task_index_2])
        regularizer_2=utils.get_conjugate_loss(gate_1[1],gate_2[1],permutation_index_list[task_index_1],permutation_index_list[task_index_2])

        loss=regularizer_1+regularizer_2
        loss.backward()

        optimizer_aux.step()
        train_loss += loss.item()
        total += targets.size(0)
        indicator = int(len(train_set) / 3)
        if ((batch_idx + 1) % indicator == 0):
            print('conjugate loss:{}'.format(train_loss/(batch_idx+1)))
    if (torch.rand(1) < 0.2):
        plt.hist(gate_1[0].detach().cpu().numpy().reshape(-1), bins=100)
        plt.savefig('gate.png')
        plt.close()

def train_snn_task(net, gate_net, data_set, permutation_index_list,test_gate):
    test_acc_list = []
    optimizer_main = optim.SGD(net.parameters(), lr=args.lr_s2,momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()
    stage_2_index=[i for i in range(args.test_task_num)]
    print('stage_2:-------------------')
    # if (args.stage_2_index_permute == True):
    #     stage_2_index = np.random.permutation(stage_2_index)
    print('train:', stage_2_index)
    # if(args.ewc==True):
    #     W = {}
    #     p_old = {}
    #     for n, p in net.named_parameters():
    #         if p.requires_grad:
    #             print(n)
    #             n = n.replace('.', '__')
    #             net.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
    #             net.register_buffer('{}_SI_omega'.format(n), 0 * p.data.clone())

    for index in stage_2_index:
        print('train task index {}---------{}------------------------------------------'.format(index,args.root_path))
        # if(args.ewc==True):
        #     for n, p in net.named_parameters():
        #         if p.requires_grad:
        #             n = n.replace('.', '__')
        #             W[n] = p.data.clone().zero_()
        #             p_old[n] = p.data.clone()

        for i in range(args.stage_2_epoch_per_task):
            train_snn_epoch(net, test_gate, i, data_set[0], permutation_index_list, optimizer_main, criterion, index)
        # if(args.ewc==True):
        #     print('consolidating')
        #     net.estimate_fisher(data_set[0], permutted_paramer=permutation_index_list[index],gate=test_gate[index])
        if ((index + 1) % 4==0):#args.task_per_group == 0):
            test_acc,_ = test_task(net, gate_net, data_set[1], permutation_index_list,'/test_gate.npy',test_gate,False,index+1)
            print('accuracy: ', test_acc)
            test_acc_list.append(test_acc)
            # plt.plot(test_acc_list)
            # plt.savefig(args.root_path + '/acc.png')
            # plt.close()
            utils.save_dict({'acc': test_acc_list}, args.root_path + '/test_acc.pkl')
            # torch.save(net.state_dict(), args.root_path + '/models/parameters_{}.pkl'.format(index))

def train_snn_epoch(net, train_gate, epoch, train_set, permutation_index_list, optimizer_main,criterion, task_index):
    net.train()
    def adjust_lr(epoch):
        lr = args.lr_s2 * (0.1 ** (epoch // 5))
        return lr

    lr = adjust_lr(epoch)
    for p in optimizer_main.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f' % (epoch, lr))
    train_loss = 0
    correct = 0
    total = 0
    gate = train_gate[task_index]
    gate = [torch.FloatTensor(gate[0]).cuda(), torch.FloatTensor(gate[1]).cuda()]
    for batch_idx, (inputs, targets) in enumerate(train_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs[:,:,permutation_index_list[task_index]]
        optimizer_main.zero_grad()

        pred,_ = net(inputs, gate)
        loss = criterion(pred, targets)
        if(args.ewc==True):
            ewc_loss = net.ewc_loss() * net.ewc_lambda
            loss = loss + ewc_loss
        loss.backward()

        optimizer_main.step()
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        indicator = int(len(train_set) / 3)
        if ((batch_idx + 1) % indicator == 0):
            if(args.ewc==True):
                print(ewc_loss)
            print(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test_task(net, gate_net, test_set, permutation_index_list,path,test_gate,generate_gate,tasks_to_now):
    print('testing-------------------------------------')
    def unfold(gate):
        gate_1 = []
        gate_2 = []
        for g in gate:
            gate_1.append(g[0].detach().cpu().numpy())
            gate_2.append(g[1].detach().cpu().numpy())
        gate_1 = np.array(gate_1)
        gate_2 = np.array(gate_2)
        gate_1 = gate_1.reshape(-1, channels)
        gate_2 = gate_2.reshape(-1, channels)
        gate = np.concatenate((gate_1, gate_2), axis=1)
        return gate

    acc_list = []
    gate_list = []
    gate_list_save=[]
    if(generate_gate==True):
        tasks_to_now=args.test_task_num
    for i in tqdm(range(tasks_to_now)):
        acc, gate = test_epoch(net, gate_net, test_set, permutation_index_list,i,test_gate,generate_gate)
        acc_list.append(acc)
        temp=unfold(gate)
        # gate_list_save.append(temp.copy())
        temp=np.mean(temp,axis=0,keepdims=True)
        gate_list.append(temp)
    gate = np.array(gate_list)
    # gate_save=np.array(gate_list_save)
    gate[gate>=0.5]=1
    gate[gate<0.5]=0
    # np.save(args.root_path +path, gate_save)
    print('mean acc:', np.mean(acc_list))
    return acc_list,gate

def test_epoch(net, gate_net, test_set, permutation_index_list,task_index,test_gate,generate_gate):
    net.eval()
    correct = 0
    total = 0
    acc=0
    gate_list = []
    for batch_idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs[:,:, permutation_index_list[task_index]]
        temp=inputs.mean(dim=1,keepdim=False)
        gate = gate_net(temp)
        gate_list.append(gate)
        if(generate_gate==False):
            gate_temp=test_gate[task_index]
            gate_temp = [torch.FloatTensor(gate_temp[0]).cuda(), torch.FloatTensor(gate_temp[1]).cuda()]
            pred ,_= net(inputs,gate_temp)
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    if(generate_gate==False):
        acc = 100. * correct / total
    return acc, gate_list

def print_args(args):
    dict = vars(args)
    print('arguments:--------------------------')
    for key in dict.keys():
        print('{}:{}'.format(key, dict[key]))
    utils.save_dict(dict, args.root_path + '/args.pkl')
    print('-----------------------------------')

def main():
    print(args.root_path)
    create_folders()
    print_args(args)
    print('--------pre_task---------')
    permutation_index_list = get_permutation_index(args.task_times)
    net = get_net()
    gate_net = get_gate_net()
    data_set = get_data_set()
    print('--------train_task---------')
    test_gate=train_ann_task(net, gate_net, data_set, permutation_index_list)
    if(args.xdg==True):
        test_gate=gate_manual_mm
    np.save(args.root_path+'/gate_to_train.npy',test_gate)
    train_snn_task(net, gate_net, data_set, permutation_index_list[-args.test_task_num:,:],test_gate)
main()
