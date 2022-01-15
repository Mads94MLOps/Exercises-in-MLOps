from turtle import shape
import torch
from src.models.model import MyAwesomeModel

def test_model_output():
    shape_tensor = torch.randn(64,784)
    model = MyAwesomeModel()
    assert list(model(shape_tensor).shape) == [64, 10] , "Model didn't output the right shape"
