#a
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):
    def __init__(self, class_num=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = class_num
        self.size_average = size_average
        self.num_classes = class_num
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight


    def forward(self, input, target, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        target_onehot = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        if self.weights != None:
            probs = (self.weights * probs * target_onehot).sum(1)
        else:
            probs = (probs * target_onehot).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLossUncert(nn.Module):
    def __init__(self, class_num=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(FocalLossUncert, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = class_num
        self.size_average = size_average
        self.num_classes = class_num
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight


    def forward(self, input, target, uncet, eps=1e-5):

        EPS = 1e-7
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        uncet = uncet.permute(0, 2, 3, 1).contiguous().view(-1)  # B * H * W, C = P, C

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            uncet = uncet[valid]

        target_onehot = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        if self.weights != None:
            probs = (self.weights * probs * target_onehot).sum(1)
        else:
            probs = (probs * target_onehot).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p / (uncet + EPS)+ (uncet + EPS).log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss