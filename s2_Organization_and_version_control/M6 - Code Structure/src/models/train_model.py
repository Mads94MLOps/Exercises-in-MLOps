import torch
import numpy as np

import matplotlib.pyplot as plt
from src.models.model import MyAwesomeModel
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler


print('hello')

def train():
    print("Training day and night")
    '''       
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    '''    
    # TODO: Implement training loop here
    model = MyAwesomeModel()

    dataset = torch.load('data/processed/data_set_processed.pt')
    
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       sampler=train_sampler)
 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)


    epochs = 5
    losses = []
    print('Before training loop')
    for i in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
                
            images = images.view(images.shape[0], -1)
                
            optimizer.zero_grad()
            #import pdb
            #pdb.set_trace()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
                
            optimizer.step()
                
            running_loss += loss.item()
                
        else:
            print(f'Traning loss: {running_loss/len(trainloader)}')
        losses.append(running_loss)
        
        #torch.save(model, "C:/Users/Mads_/OneDrive/Anvendt Kemi/Machine Learning Operations/dtu_mlops/s1_getting_started/exercise_files/final_exercise/s1-M4_Pytorch/training_model.pt")
    plt.plot(losses)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    plt.savefig('reports/figures/accuracy.png')
    # plt.savefig('C:/Users/Mads_/OneDrive/Anvendt Kemi/Machine Learning Operations/Exercises-in-MLOps/s2_Organization_and_version_control/M6 - Code Structure/reports/figures/accuracy.png')
    torch.save(model.state_dict(), "models/training_model.pt")

if __name__ == '__main__':
    train()
