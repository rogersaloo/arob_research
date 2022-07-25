import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import (Dataset, DataLoader, )
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from model_classifiers import resnet_50
from config_classifier import csv_file, root_dir, loss_tensorboard, accuracy_tensorboard
from config_classifier import learning_rate, batch_size, num_epochs
from dataset_classifier import PneuAndNormalDataset

# Tensor board writer
writer_loss = SummaryWriter(loss_tensorboard)
writer_accuracy = SummaryWriter(accuracy_tensorboard)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
dataset = PneuAndNormalDataset(csv_file=csv_file, root_dir=root_dir, )
# train_set, test_set = torch.utils.data.random_split(dataset, [38846, 9700])
train_set, test_set = torch.utils.data.random_split(dataset, [700, 212])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = resnet_50()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
step = 0
for epoch in range(num_epochs):
    losses = []
    accuracies = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
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

        # calculate 'running' training accurcy
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])

        writer_loss.add_scalar('Training Loss', loss, global_step=step)
        writer_accuracy.add_scalar('Training Loss', running_train_acc, global_step=step)
        step += 1

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    CM = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores_test = model(x)
            _, predictions = scores_test.max(1)

            CM += confusion_matrix(y.cpu(), predictions.cpu(), labels=[0, 1])
            tn, tp, fp, fn = CM[0][0], CM[1][1], CM[0][1], CM[1][0]
            acc = np.sum(np.diag(CM) / np.sum(CM))
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp)

            # print('\nTestset Accuracy(mean): %f %%' % (100 * acc))

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)
        print('- Precision: ', (tp / (tp + fp)) * 100)
        print('- NPV: ', (tn / (tn + fn)) * 100)
        # print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
