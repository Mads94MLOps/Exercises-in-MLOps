import torch
from torch import Tensor
from src.models import train_model
import mock

from tests import _PATH_DATA

@mock.patch('src.models.train_model.MyAwesomeModel')
def test_model_is_called_once(mock_MyAwesomeModel):
    try:
        train_model.train()
    except ValueError: # Otherwise it requires a optmizer
        pass
    #train_model.train()

    mock_MyAwesomeModel.assert_called_once() , "Training script does not call training model"

