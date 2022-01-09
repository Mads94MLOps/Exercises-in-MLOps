# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:22:03 2022

@author: Mads_
"""

from typing_extensions import Concatenate
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data as torchdata
from torch.utils.data import TensorDataset


path = "C:/Users/Mads_/Desktop/dtu_mlops/data/corruptmnist/"



list_of_files = os.listdir(path) #list of files in the current directory

                
train_images = []
train_labels =[]

for i,each_file in enumerate(list_of_files):
    
    if each_file.startswith('train_'):
        
        # Find the path of the file
        data = np.load(os.path.join(path, each_file))
        images = data['images']
        labels = data['labels']
        train_images.append(images)
        train_labels.append(labels)
   


   
train_images = np.concatenate(train_images,axis=0)
train_labels = np.concatenate(train_labels,axis=0)    

train_images = torch.from_numpy(train_images).type(torch.FloatTensor)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)

trainset = TensorDataset(train_images, train_labels)
torch.save(trainset, f'{path}/training_set.pt')


data = np.load(os.path.join(path,'test.npz'))
images = data['images']
labels = data['labels']
test_images = torch.from_numpy(images).type(torch.FloatTensor)
test_labels = torch.from_numpy(labels).type(torch.LongTensor)

testset = TensorDataset(test_images, test_labels)
torch.save(testset, f'{path}/test_set.pt')

"""
class final_exercise_data(torchdata.Dataset):
    
    def __init__(self, npz_file):
        self.file = torch.load(npz_file)
        #self.transform = transform
        
        
        
    def __len__(self):
        return self.file['images'].shape[0]
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            

        image = self.file['images'][index]
        label = self.file['labels'][index]
        return image,label
            


#transform = transforms.Compose([transforms.ToTensor(),
                               # transforms.Normalize((0.5,), (0.5,))])

"""

trainset = torch.load(path+'training_set.pt')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torch.load(path+'test_set.pt')
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)



