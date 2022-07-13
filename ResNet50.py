import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import (Dataset, DataLoader, )
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import seaborn as sns

#Variables
# csv_file = "train_metadata.csv",
from tqdm import tqdm

csv_file = 'sample_data/sample_data_combine.csv'
# root_dir = "alldata/train/real_train_images",
root_dir = 'sample_data/sample_images_combined'

transforming = transforms.Compose([
    transforms.ToTensor()])

writer_loss = SummaryWriter('runsclassification/loss')
writer_accuracy = SummaryWriter('runsclassification/accuracy')
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


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 2
learning_rate = 1e-3
batch_size = 16
num_epochs = 3

# Load Data
dataset = PneuAndNormalDataset(
    csv_file=csv_file,
    root_dir=root_dir,
)

# train_set, test_set = torch.utils.data.random_split(dataset, [38846, 9700])
train_set, test_set = torch.utils.data.random_split(dataset, [700, 212])
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

# create grid of images
tensorboard_images = iter(train_loader)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Model
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
# model.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
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
step = 0
for epoch in range(num_epochs):
    losses = []
    accuracies = []

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

        #calculate 'running' training accurcy
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct)/float(data.shape[0])

        writer_loss.add_scalar('Training Loss', loss,global_step=step)
        writer_accuracy.add_scalar('Training Loss', running_train_acc, global_step=step)
        step += 1



    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_wrong = 0
    num_samples = 0
    model.eval()
    CM=0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores_test = model(x)
            _, predictions = scores_test.max(1)

            CM += confusion_matrix(y.cpu(), predictions.cpu(), labels=[0, 1])
            tn = CM[0][0]
            tp = CM[1][1]
            fp = CM[0][1]
            fn = CM[1][0]
            acc = np.sum(np.diag(CM) / np.sum(CM))
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp)

            print('\nTestset Accuracy(mean): %f %%' % (100 * acc))

            num_correct += (predictions == y).sum()
            num_wrong += (predictions != y).sum()
            num_samples += predictions.size(0)
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)
        print('- Precision: ', (tp / (tp + fp)) * 100)
        print('- NPV: ', (tn / (tn + fn)) * 100)
        print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)

# y_true = []
# y_pred = []
#
# #Confusion Matrix
# for data in tqdm(test_loader):
#     images, labels = data[0].to(device), data[1]
#     y_true.extend(labels.numpy())
#
#     outputs = model(images)
#
#     _, predicted = torch.max(outputs, 1)
#     y_pred.extend(predicted.cpu().numpy())
#
# cf_matrix = confusion_matrix(y_true, y_pred)
#
# class_names = ('pneumonia','normal',)
#
# # Create pandas dataframe
# dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
# print(dataframe)
#
# plt.figure(figsize=(6, 4))
#
# # Create heatmap
# sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
#
# plt.title("Confusion Matrix"), plt.tight_layout()
#
# plt.ylabel("True Class"),
# plt.xlabel("Predicted Class")
# plt.show()

