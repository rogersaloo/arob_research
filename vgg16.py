import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1

# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from torch.utils.data import (Dataset, DataLoader, )
import sys
from PIL import Image

transforming = transforms.Compose([
    transforms.ToTensor()])


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
        y_label = self.annotations.iloc[index, 11]
        if self.transform:
            image = self.transform(image)

        return image, y_label


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 2
learning_rate = 1e-3
batch_size = 4
num_epochs = 10

# Load Data
dataset = PneuAndNormalDataset(
    csv_file="train_metadata.csv",
    root_dir="alldata/train/real_train_images",
)

train_set, test_set = torch.utils.data.random_split(dataset, [38846, 9700])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")

print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Model
model = torchvision.models.vgg16(pretrained=True)
model.classifier = nn.Linear(32768, 2)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
model.to(device)
print(model)

# sys.exit()
# x= torch.randn(64,1,28,28).to(device)
# print(model(x).shape)
# exit()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores_test = model(x)
            _, predictions = scores_test.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
