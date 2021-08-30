#a
import os
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from model.psp import PSPNet
from model.sp import SPNet
from model.ft import FTNet
from model.fer import FERNet
from model.ema import EMANet
from model.ocr import OCRNet
from model.oc import OCNet
from model.deeplabv3 import DeepLabV3
from model.cc import CCNet
from model.acf import ACFNet
from model.da import DANet
from model.ann import ANNet

from model.loss import FocalLoss

from AtlantisLoader import AtlantisDataSet
from AtlantisLoader import Atlantis36DataSet
from AtlantisLoader import Atlantis56DataSet
import joint_transforms as joint_transforms

INPUT_SIZE = '640'
MODEL = 'PSPNet'
NUM_CLASSES = 36
SNAPSHOT_DIR = './snapshots/psp'
DATA_DIRECTORY = './data/atlantis'
BATCH_SIZE = 2
NUM_WORKERS = 4
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 30
POWER = 0.9
RESTORE_FROM = './model/resnet101-imagenet.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="PSPNet Network")
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

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'PSP2Net':
        model = PSP2Net(num_classes=args.num_classes)
    if args.model == 'PSP3Net':
        model = PSP3Net(num_classes=args.num_classes)
    if args.model == 'PSP4Net':
        model = PSP4Net(num_classes=args.num_classes)
    if args.model == 'PSP5Net':
        model = PSP5Net(num_classes=args.num_classes)
    if args.model == 'SPNet':
        model = SPNet(num_classes=args.num_classes)
    if args.model == 'SP2Net':
        model = SP2Net(num_classes=args.num_classes)
    if args.model == 'SP3Net':
        model = SP3Net(num_classes=args.num_classes)
    if args.model == 'SP4Net':
        model = SP4Net(num_classes=args.num_classes)
    if args.model == 'SP5Net':
        model = SP5Net(num_classes=args.num_classes)
    if args.model == 'DANet':
        model = DANet(num_classes=args.num_classes)
    if args.model == 'ANNet':
        model = ANNet(num_classes=args.num_classes)
    if args.model == 'SETR':
        model = SETRModel(num_classes=args.num_classes)
    if args.model == 'EMANet':
        model = EMANet(num_classes=args.num_classes)
    if args.model == 'ACFNet':
        model = ACFNet(num_classes=args.num_classes)
    if args.model == 'FERNet':
        model = FERNet(num_classes=args.num_classes)
    if args.model == 'FTNet':
        model = FTNet(num_classes=args.num_classes)
    if args.model == 'CCNet':
        model = CCNet(num_classes=args.num_classes)
    if args.model == 'DeepLabV3':
        model = DeepLabV3(num_classes=args.num_classes)
    if args.model == 'OCRNet':
        model = OCRNet(num_classes=args.num_classes)
    if args.model == 'TPSPNet':
        model = TPSPNet(num_classes=args.num_classes)
    if args.model == 'OCNet':
        model = OCNet(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)


    model.train()
    model.cuda()
    # torch.backends.cudnn.enabled = False
    cudnn.enabled = True
    cudnn.benchmark = True

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
    if args.num_classes == 35:
        trainloader = data.DataLoader(
            AtlantisDataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.num_classes == 36:
        trainloader = data.DataLoader(
            Atlantis36DataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.num_classes == 56:
        trainloader = data.DataLoader(
            Atlantis56DataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)

    i_iter = 0
    print (len(trainloader))
    for epoch in range(args.num_epochs):
        for images, labels, _, _, _ in trainloader:
            i_iter+=args.batch_size
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, args.num_epochs*len(trainloader)*args.batch_size)
            images = images.cuda()
            labels = labels.long().cuda()

            pred_aux, pred = model(images)
            pred = interp(pred)
            pred_aux = interp(pred_aux)
            loss = seg_loss(pred, labels) + 0.4 * seg_loss(pred_aux, labels)
            loss.backward()
            optimizer.step()

            print('epoch = {0:4d}, iter = {1:8d}/{2:8d}, loss_seg = {3:.3f}'.format(epoch, i_iter, args.num_epochs*len(trainloader), loss))
        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    main()
