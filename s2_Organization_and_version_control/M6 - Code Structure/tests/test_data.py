import torch
from tests import _PATH_DATA
from torch.utils.data import DataLoader

#dataset_img = torch.load(f'{_PATH_DATA}/processed/data_set_processed.pt')
dataset = DataLoader(torch.load('data/processed/data_set_processed.pt'))

print(dataset[0])

dataiter = iter(dataset)
images, labels = dataiter.next()

print(dataiter)
print(type(images))
print(images.shape)
print(labels.shape)

#dataset[1].size()

#def test_cropped_images():
#    assert list(dataset_img[1].size()) == [50,50,3] 