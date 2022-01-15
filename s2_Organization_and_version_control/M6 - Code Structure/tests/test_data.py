import torch
from tests import _PATH_DATA
from torch.utils.data import DataLoader
from mock import patch

#dataset_img = torch.load(f'{_PATH_DATA}/processed/data_set_processed.pt')
dataset = torch.load('data/processed/data_set_processed.pt')
print(len(dataset['Trainset']))
print(len(dataset['Testset']))
# initializing parameters
batch_size = 64

trainloader=DataLoader(dataset['Trainset'], batch_size=batch_size, shuffle=True)
testloader=DataLoader(dataset['Testset'], batch_size=batch_size, shuffle=True)

trainiter = iter(trainloader)
testiter = iter(testloader)

train_images, train_labels = next(trainiter)
test_images, test_labels = next(testiter)

#@patch('src.data.make_dataset.train_length')
def test_datalength():
    assert len(dataset['Trainset']) == 0.8 * (len(dataset['Trainset']+dataset['Testset']))
    assert len(dataset['Testset']) == 0.2 * (len(dataset['Trainset']+dataset['Testset']))

def test_format():
    assert list(train_images[0].shape) == [1, 28, 28]
    assert list(test_images[0].shape) == [1, 28, 28]

def test_label_representation():
    assert list(torch.unique(train_labels)) == [0,1,2,3,4,5,6,7,8,9]
    assert list(torch.unique(test_labels)) == [0,1,2,3,4,5,6,7,8,9]

