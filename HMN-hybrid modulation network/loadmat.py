import h5py
import matplotlib.pyplot as plt
file='data/NMNIST_train_data_2ms.mat'
data=h5py.File(file,'r')

print(data['label'].shape)
data=data['image'][:,0,:,:,12]
for i in range(50):
    plt.imshow(data[i,:,:])
    plt.show()

import numpy as np
data=np.load('data/npy/train/images.npy')
print(data.shape)
data=data[0,:,:,0,:]

for i in range(50):
    plt.imshow(data[:,:,i])
    plt.show()