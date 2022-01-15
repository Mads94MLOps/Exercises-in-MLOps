import torch
from torch import Tensor
from src.models import train_model
from src.models.model import MyAwesomeModel
import mock

from tests import _PATH_DATA

#from src.data.make_dataset import


#dataset_img = torch.load(f'{_PATH_DATA}/processed/train/imgs_train_0.pt')
#dataset_label = torch.load(f'{_PATH_DATA}/processed/train/labels_train.pt')
#dataset_label = torch.load('data/processed/train/labels_train.pt')

#print(dataset_label.size(dim=0))



@mock.patch('src.models.train_model.MyAwesomeModel')
def test_train(mock_MyAwesomeModel):
    try:
        train_model.train()
    except ValueError:
        pass
    #train_model.train()

    mock_MyAwesomeModel.assert_called_once()