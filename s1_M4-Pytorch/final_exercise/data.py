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
                
def mnist():
    
    #transform = transforms.Compose([transforms.ToTensor(),
              #              transforms.Normalize((0.5,), (0.5,))])

    #path = "C:/Users/Mads_/Desktop/dtu_mlops/data/corruptmnist/"



    list_of_files = os.listdir(path) #list of files in the current directory


    # Create training set                
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

    # Create test-set
    data = np.load(os.path.join(path,'test.npz'))
    images = data['images']
    labels = data['labels']
    test_images = torch.from_numpy(images).type(torch.FloatTensor)
    test_labels = torch.from_numpy(labels).type(torch.LongTensor)

    testset = TensorDataset(test_images, test_labels)
    torch.save(testset, f'{path}/test_set.pt')

    #Load training set
    trainset = torch.load(path+'training_set.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    #Load test set
    testset = torch.load(path+'test_set.pt')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        
    return trainloader, testloader

