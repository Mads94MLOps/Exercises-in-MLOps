import argparse
import sys

import torch
import numpy as np
from src.models.model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def predict():
    """Evaluates the given model on the given data set,
    printing the accuracy."""
    # parsing
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("load_model_from", default="")
    parser.add_argument("load_data_from", default="")
    args = parser.parse_args(sys.argv[1:])

    # model loading
    model = MyAwesomeModel()
    state_dict = torch.load(args.load_model_from)
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # data loading
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
    
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

    # prediction
    acc_list = []
    for images, labels in validation_loader:
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        acc_list.append(accuracy)
    model_accuracy = np.mean(acc_list)
    print(f"Accuracy: {model_accuracy.item() * 100}%")


if __name__ == "__main__":
    predict()