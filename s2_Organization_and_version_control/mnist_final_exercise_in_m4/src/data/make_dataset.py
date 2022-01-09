# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_images = []
    data_labels = []
    for i,each_file in enumerate(os.listdir(input_filepath)):

        if each_file.endswith('.npz'):        
            # Find the path of the file
            data = np.load(os.path.join(input_filepath, each_file))
            images = data['images']
            labels = data['labels']
            data_images.append(images)
            data_labels.append(labels)
    
    data_images = np.concatenate(data_images,axis=0)
    data_labels = np.concatenate(data_labels,axis=0)

    data_images = torch.from_numpy(data_images).type(torch.FloatTensor)
    data_labels = torch.from_numpy(data_labels).type(torch.LongTensor)

    from torchvision import transforms

    norm_data_images = transforms.Normalize(0,1)(data_images)

    data_set_tensor = TensorDataset(norm_data_images, data_labels)
    torch.save(data_set_tensor, f'{output_filepath}/data_set_processed.pt')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
