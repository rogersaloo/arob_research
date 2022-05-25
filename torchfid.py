import os
import logging
import ignite
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import sys
ignite.utils.manual_seed(999)
ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)


#Set the transformation for the dataset
image_size = 64
batch_size = 8

real_dataset_path= "fid/real/"
synth_dataset_path= "fid/synthetic/"

data_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

real_dataset = torchvision.datasets.ImageFolder(root=real_dataset_path, transform=data_transform)
synth_dataset = torchvision.datasets.ImageFolder(root=synth_dataset_path, transform=data_transform)

#show the transformed dataset
def show_transformed_dataset(dataset):
    loader = DataLoader(dataset, batch_size, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow =3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid,(1,20)))
    print('labels', labels)

show_transformed_dataset(real_dataset)
sys.exit()

