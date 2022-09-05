import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
# class dataset_from_mat(data.Dataset):
#     def __init__(self,file_path,transform):
#         super(dataset_from_mat, self).__init__()
#         self.data=loadmat(file_path)
#         self.images=self.data['images'].astype(np.float32)
#         self.label=self.data['label']
#         self.transform=transform
#     def __getitem__(self, index):
#         inputs=self.images[index,:,:,:6]
#         targets=self.label[0,index]
#         if(self.transform!=None):
#             inputs=self.transform(inputs)
#         return inputs,targets

#     def __len__(self):
#         return self.images.shape[0]


class dataset_from_mat_dvs(data.Dataset):
    def __init__(self,file_path,wins):
        super(dataset_from_mat_dvs, self).__init__()
        self.wins=wins
        self.images=np.load(file_path+'/images.npy').astype(np.float32)[:,:,:,:,:wins]
        self.images=np.transpose(self.images,(0,4,3,1,2))
        self.images=self.images.reshape(-1,self.wins,34*34*2)
        self.label=np.load(file_path+'/labels.npy')
    def __getitem__(self, index):
        inputs=self.images[index,:,:]
        targets=self.label[index]
        inputs=torch.Tensor(inputs)
        return inputs,targets
    def __len__(self):
        return self.images.shape[0]