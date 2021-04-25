#a
import os
import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from torch.nn import functional as F
from model.resnet import *

from AtlantisLoader import AWODataSet
import joint_transforms as joint_transforms

INPUT_SIZE = '640'
PADDING_SIZE = '768'
MODEL = 'ResNet152'
NUM_CLASSES = 19
SNAPSHOT_DIR = './snapshots/wo'
DATA_DIRECTORY = './data/awo'
BATCH_SIZE = 8
NUM_WORKERS = 8
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 30
POWER = 0.9
RESTORE_FROM = './model/resnet18-imagenet.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="PSPNet Network")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--padding-size", type=int, default=PADDING_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : PSPNet")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    return parser.parse_args()

args = get_arguments()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, total_steps):
    lr = lr_poly(args.learning_rate, i_iter, total_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    input_size = args.input_size

    if args.model == 'ResNet18':
        model = ResNet18(num_classes=args.num_classes)
    if args.model == 'ResNet34':
        model = ResNet34(num_classes=args.num_classes)
    if args.model == 'ResNet50':
        model = ResNet50(num_classes=args.num_classes)
    if args.model == 'ResNet101':
        model = ResNet101(num_classes=args.num_classes)
    if args.model == 'ResNet152':
        model = ResNet152(num_classes=args.num_classes)



    model.train()
    model.cuda()
    # torch.backends.cudnn.enabled = False
    cudnn.enabled = True
    cudnn.benchmark = True

    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.input_size,
                                           False,
                                           pre_size=None,
                                           scale_min=0.9,
                                           scale_max=1.1,
                                           ignore_index=0),
        joint_transforms.Resize(args.input_size),
        joint_transforms.RandomHorizontallyFlip()]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)
    trainloader = data.DataLoader(
            AWODataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)


    i_iter = 0
    print (len(trainloader))
    for epoch in range(args.num_epochs):

        model.train()
        for images, labels, _, _, _ in trainloader:
            i_iter+=args.batch_size
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, args.num_epochs*len(trainloader)*args.batch_size)
            images = images.cuda()
            labels = labels.long().cuda()

            pred = model(images)
            loss = seg_loss(pred, labels)
            loss.backward()
            optimizer.step()

            # print('epoch = {0:4d}, iter = {1:8d}/{2:8d}, loss_seg = {3:.3f}'.format(epoch, i_iter, args.num_epochs*len(trainloader), loss))

        model.eval()
        correct = 0
        totalpixel = 0

        valloader = data.DataLoader(
            AWODataSet(args.data_dir, split='val', padding_size=args.padding_size),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        for images, labels, name, width, height in valloader:
            images = images.cuda()
            images = F.upsample(images, [args.input_size, args.input_size], mode='bilinear')
            with torch.no_grad():
                pred = model(images)

            pred = pred.cpu().data[0].numpy()
            pred = np.asarray(np.argmax(pred, axis=0), dtype=np.uint8)
            labels = np.asarray(labels.squeeze(0), dtype=np.uint8)
            totalpixel += 1
            if pred == labels:
                correct += 1
        acc = correct / totalpixel
        print('epoch = {0:4d}, acc = {1:8f}'.format(epoch, acc))

        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    main()
