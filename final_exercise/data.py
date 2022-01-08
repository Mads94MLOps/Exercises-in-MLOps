import torch
import os
import torch.utils.data as torchdata
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset

path = "C:/Users/Mads_/OneDrive/Anvendt Kemi/Machine Learning Operations/dtu_mlops/data/corruptmnist/"

list_of_files = os.listdir(path) #list of files in the current directory
                
def mnist():
    
    #transform = transforms.Compose([transforms.ToTensor(),
              #              transforms.Normalize((0.5,), (0.5,))])

    trainset = torch.load(path+'training_set.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torch.load(path+'test_set.pt')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        
    return trainloader, testloader

