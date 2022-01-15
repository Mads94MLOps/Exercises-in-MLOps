import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.models.model import MyAwesomeModel


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
    dataset = torch.load("data/processed/data_set_processed.pt")

    # initializing parameters
    batch_size = 64
    train_split = 0.8
    random_seed=42

    # Defining split
    train_length=int(train_split* len(dataset))
    test_length=len(dataset)-train_length

    # Splitting
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,
            (train_length,test_length),
            generator=torch.Generator().manual_seed(random_seed)
            )

    validation_loader=torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=True)

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
