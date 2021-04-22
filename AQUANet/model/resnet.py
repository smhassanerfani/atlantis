#a
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torchvision
import torchvision.models as models

def ResNet18(num_classes=21):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
    return model


def ResNet34(num_classes=21):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
    return model


def ResNet50(num_classes=21):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
    return model


def ResNet101(num_classes=21):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(2048 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
    return model


def ResNet152(num_classes=21):
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(2048 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
    return model

