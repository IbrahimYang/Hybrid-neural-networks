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
import cv2
from scipy.io import loadmat,savemat
def get_data_set():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    return trainloader, testloader

def generate_spikes(data_set,t_window,size,path):
    images = []
    label=[]
    for batch_idx, (inputs, targets) in tqdm(enumerate(data_set)):
        temp=inputs.detach().cpu().numpy()[0,0]
        targets=targets.detach().cpu().numpy()
        temp=cv2.resize(temp,(size,size),interpolation=cv2.INTER_CUBIC)
        temp[temp>1]=1
        temp[temp<0]=0
        spike=np.zeros([size,size,t_window])
        for i in range(t_window):
            spike[:,:,i]=temp>np.random.rand()
        images.append(spike)
        label.append(targets)
    images=np.array(images).astype(np.float32)
    label=np.array(label).reshape(-1)
    print(images.shape,label.shape)
    savemat(path,{'images':images,'label':label})

train_loader,test_loader=get_data_set()
generate_spikes(train_loader,6,34,'data/mat/MNIST_PYTHON_TRAIN_4D.mat')
generate_spikes(test_loader,6,34,'data/mat/MNIST_PYTHON_TEST_4D.mat')