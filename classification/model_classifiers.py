import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

def resnet_50():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    # print(model)
    return model

def densnet_121():
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, 2)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    # print(model)
    return model

def vgg_16():
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier = nn.Linear(32768, 2)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
    # print(model)
    return model
