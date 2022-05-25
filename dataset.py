from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class NormalPneumonia(Dataset):
    def __init__(self, root_normal, root_pneumonia, transform=None):
        self.root_normal = root_normal
        self.root_pneumonia = root_pneumonia
        self.transform = transform

        self.normal_images = os.listdir(root_normal)
        self.pneumonia_images = os.listdir(root_pneumonia)
        self.length_dataset = max(len(self.normal_images), len(self.pneumonia_images))
        self.normal_len = len(self.normal_images)
        self.pneumonia_len = len(self.pneumonia_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        normal_img = self.normal_images[index % self.normal_len]
        pneumonia_img = self.pneumonia_images[index % self.pneumonia_len]

        normal_path = os.path.join(self.root_normal, normal_img)
        pneumonia_path = os.path.join(self.root_pneumonia, pneumonia_img)

        normal_img = np.array(Image.open(normal_path).convert("RGB"))
        pneumonia_img = np.array(Image.open(pneumonia_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=normal_img, image0=pneumonia_img)
            normal_img = augmentations["image"]
            pneumonia_img = augmentations["image0"]

        return normal_img, pneumonia_img

# path = "C:/Users/Aloo/Documents/Research/arob/"
# new_path = os.path.join(path,fid_config.NORMAL_PATH)
# print(new_path)
# sys.exit()
# import config
# lr = config.LEARNING_RATE
# batch = config.BATCH_SIZE
# epoch = config.NUM_EPOCHS
#
# #Batch 1-ep_100 lr1e-2
# NORMAL_PATH = f"fid/batch_{batch}-epoch_{epoch}-lr_{lr}/real"
# SYNTHESIS_PATH = f"fid/{batch}-{epoch}-{lr}/synthetic/"
# NORMAL_IMAGE_LABEL=f"norm-batch_{batch}-epoch_{epoch}-lr_{lr}"
# PNEUMONIA_IMAGE_LABEL=f"pneu-batch_{batch}-epoch_{epoch}-lr_{lr}"