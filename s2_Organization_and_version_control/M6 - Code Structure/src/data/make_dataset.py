import logging
import os
from pathlib import Path
import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import TensorDataset
from torchvision import transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data_images = []
    data_labels = []
    for i, each_file in enumerate(os.listdir(input_filepath)):
        if each_file.endswith(".npz"):
            # Loop that sorts the content of .npz files into images and labels
            #  and append them to a list
            data = np.load(os.path.join(input_filepath, each_file))
            images = data["images"]
            labels = data["labels"]
            
            data_images.append(images)
            data_labels.append(labels)

    data_images = np.concatenate(data_images, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)
    
    data_images = torch.from_numpy(data_images.swapaxes(1, 2).reshape(len(data_images),
                 1, 28, 28)).type(torch.FloatTensor)
    data_labels = torch.from_numpy(data_labels).type(torch.LongTensor)

    # Normalizing the images with mean=0 and std=1
    norm_data_images = transforms.Normalize(0, 1)(data_images)

    # Creates a dataset of two tensors
    data_set_tensor = TensorDataset(norm_data_images, data_labels)

    train_split = 0.8
    random_seed=42

    train_length=int(train_split* len(data_set_tensor))

    test_length=len(data_set_tensor)-train_length

    train_dataset,test_dataset=torch.utils.data.random_split(data_set_tensor,
            (train_length,test_length),
            generator=torch.Generator().manual_seed(random_seed)
            )


    torch.save({'Trainset':train_dataset,'Testset':test_dataset}, f"{output_filepath}/data_set_processed.pt")

    data_set = torch.load(f"{output_filepath}/data_set_processed.pt")

    print(data_set['Trainset'])
    



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
