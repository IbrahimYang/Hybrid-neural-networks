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
from gate_hybrid_model_v2 import *
from tqdm import tqdm
import mat_loader
import gate_analysis

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--task_per_group', default=4, type=int)
parser.add_argument('--test_task_num', default=40, type=int)
parser.add_argument('--batch_size', default=batch_size, type=int)
parser.add_argument('--inputs_size', default=34*34*2, type=int)
parser.add_argument('--root_path', default='ewc/ewc_25', type=str)

args = parser.parse_args()
save_path=''
def get_net():
    net = main_net().cuda()
    net.load_state_dict(torch.load(save_path+args.root_path+'/models/parameters_39.pkl'))
    return net


def get_data_set():
    trainset = mat_loader.dataset_from_mat_dvs('data/npy/train',wins)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset = mat_loader.dataset_from_mat_dvs('data/npy/test',wins)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    return trainloader, testloader

def permute_index(item,ratio):
    p_num=int(args.inputs_size*ratio)
    if p_num==0:
        return item
    start=int(np.random.rand()*(args.inputs_size-p_num))
    index=item.copy().tolist()
    result=index[:start]+ np.random.permutation(index[start:start+p_num]).tolist()+index[start+p_num:]
    return np.array(result)

def test_task(net, test_set, permutation_index_list,test_gate,ratio):
    print('testing-------------------------------------')
    acc_list = []
    for i in tqdm(range(args.test_task_num)):
        acc = test_epoch(net, test_set, permutation_index_list,i,test_gate,ratio)
        acc_list.append(acc)
    print('mean acc:', np.mean(acc_list),acc_list)
    return acc_list

def test_epoch(net, test_set, permutation_index_list,task_index,test_gate,ratio):
    net.eval()
    correct = 0
    total = 0
    acc=0
    p_i=permute_index(permutation_index_list[task_index],ratio)
    for batch_idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs[:,:,p_i]

        gate_temp=test_gate[task_index]
        gate_temp = [torch.FloatTensor(gate_temp[0]).cuda(), torch.FloatTensor(gate_temp[1]).cuda()]
        pred,hidden_act = net(inputs,gate_temp)
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
    return acc

def main():
    print(args.root_path)
    permutation_index_list = np.load(save_path+args.root_path+'/p_index.npy')
    print(permutation_index_list.shape)

    net = get_net()
    data_set = get_data_set()
    test_gate=np.load(save_path+args.root_path+'/gate_to_train.npy')
    print(test_gate.shape)
    result=[]

    for i in range(10):
        ratio=i/10
        acc=test_task(net, data_set[1], permutation_index_list[-args.test_task_num:,:],test_gate,ratio)
        result.append(np.mean(acc))
    result=np.array(result)
    np.save(save_path+args.root_path+'/similarity_testtask_1.npy',result)
    # print(act.shape)
    # act_similarity=gate_analysis.calculate_similarity(act[:,0,:])
    # plt.imshow(act_similarity, interpolation='nearest', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.savefig(save_path + args.root_path+'_act_similarity.png', dpi=1000)
    # plt.close()
    # np.save(save_path + args.root_path+'_act_similarity.npy',act_similarity)
main()
