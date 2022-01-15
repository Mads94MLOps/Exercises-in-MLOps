import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.model import MyAwesomeModel
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler


def train():
    print("Training day and night")

    """
        Train a model with a Convolutional Neural Network defined in MyAwesomeModel(), with a NLLLoss
        loss function, and backward tracking of gradients.

            Arguments: 
                - Filepath of train_model followed by 'train'

    """

    model = MyAwesomeModel()

    dataset = torch.load("data/processed/data_set_processed.pt")

    # initializing parameters
    batch_size = 64

    trainloader=torch.utils.data.DataLoader(dataset['Trainset'],
        batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Training loop
    epochs = 5
    losses = []
    print("Before training loop")
    for i in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            # Return the tensor that has been flattened with images.shape[0] no. of rows and as many
            # columns as it takes to flatten the tensor
            images = images.view(images.shape[0], -1)
            print(images.shape)

            # Setting gradient to zero
            optimizer.zero_grad()

            output = model(images)
            print(output.shape)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        else:
            print(f"Traning loss: {running_loss/len(trainloader)}")
        losses.append(running_loss)

    # Creates plot of losses and stores them in reports/figures
    plt.plot(losses)
    plt.savefig("reports/figures/accuracy.png")

    torch.save(model.state_dict(), "models/training_model.pt")


if __name__ == "__main__":
    train()
