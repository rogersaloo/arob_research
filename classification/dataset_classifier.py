import os
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import (Dataset, DataLoader, )

transforming = transforms.Compose([ transforms.ToTensor()])

class PneuAndNormalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=transforming, target_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # image = io.imread(img_path)
        image = np.array(Image.open(img_path).convert("L").resize([256, 256]))
        # y_label = torch.tensor(int(self.annotations.iloc[index, 11]))
        y_label = self.annotations.iloc[index, 6]
        if self.transform:
            image = self.transform(image)
        return image, y_label


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# sys.exit()
# x= torch.randn(64,1,28,28).to(device)
# print(model(x).shape)
# exit()

# Display image and label.
# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

