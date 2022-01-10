from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):

    '''
        Convolutional Neural Network with three Linear transformations and two hidden layers,
        with dropout.
        The output layer is with softmax activation.

            Parameters:
                Variable to be predicted
            Returns:
                Prediction 
    '''

    def __init__(self):
        super().__init__()
        # Inputs to hidden layer 1 linear transformation
        self.fc1 = nn.Linear(784, 128)
        # Inputs to hidden layer 2 linear transformation
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flattens input tensor
        x = x.view(x.shape[0], -1)

        # Hidden layer 1 with ReLu activation
        x = self.dropout(F.relu(self.fc1(x)))
        # Hidden layer 2 with ReLu activation
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)

        return x
