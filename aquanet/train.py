from dataloader.AtlantisLoader import AtlantisDataSet
import dataloader.joint_transforms as joint_transforms
from network import build_model
import os
import argparse
import torch
import torch.nn as nn
from utils import *
from torch.utils import data

import torch.backends.cudnn as cudnn
import torch.optim as optim

def get_arguments(
    INPUT_SIZE = '640',
    MODEL = 'AquaNet',
    NUM_CLASSES = 56,
    SNAPSHOT_DIR = './snapshots/aquanet',
    DATA_DIRECTORY = '../atlantis/',
    BATCH_SIZE = 2,
    NUM_WORKERS = 4,
    LEARNING_RATE = 2.5e-4,
    MOMENTUM = 0.9,
    WEIGHT_DECAY = 0.0001,
    NUM_EPOCHS = 30,
    POWER = 0.9,
    RESTORE_FROM = './network/resnet101-imagenet.pth'
    ):

    parser = argparse.ArgumentParser(description="All Networks")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
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

def main():
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Data augmentation for training
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.input_size,
                                           False,
                                           pre_size=None,
                                           scale_min=0.5,
                                           scale_max=2.0,
                                           ignore_index=0),
        joint_transforms.Resize(args.input_size),
        joint_transforms.RandomHorizontallyFlip()]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # Build the data loader
    trainloader = data.DataLoader(
            AtlantisDataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Load model
    model = build_model(args)

    model.train()
    model.cuda()
    cudnn.enabled = True
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)

    i_iter = 0
    for epoch in range(args.num_epochs):
        for images, labels, _, _, _ in trainloader:
            i_iter += args.batch_size
            optimizer.zero_grad()
            adjust_learning_rate(args, optimizer, i_iter, args.num_epochs*len(trainloader)*args.batch_size)
            images = images.cuda()
            labels = labels.long().cuda()

            pred_aux, pred = model(images)
            pred = interp(pred)
            pred_aux = interp(pred_aux)
            loss = seg_loss(pred, labels) + 0.4 * seg_loss(pred_aux, labels)
            loss.backward()
            optimizer.step()

            print('epoch = {0:4d}, iter = {1:8d}/{2:8d}, loss_seg = {3:.3f}'.format(epoch, i_iter, args.num_epochs*len(trainloader), loss))
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    main()