import torch
from torch import Tensor
from src.models import train_model
#from src.models.model import MyAwesomeModel
import mock

from tests import _PATH_DATA

@mock.patch('src.models.train_model.MyAwesomeModel')
def test_train(mock_MyAwesomeModel):
    try:
        train_model.train()
    except ValueError:
        pass
    #train_model.train()

    mock_MyAwesomeModel.assert_called_once()