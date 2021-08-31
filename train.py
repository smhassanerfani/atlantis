import os
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from models.pspnet import PSPNet

from models.loss import FocalLoss

from AtlantisLoader import AtlantisDataSet
import joint_transforms as joint_transforms

INPUT_SIZE = '480'
MODEL = 'PSPNet'
NUM_CLASSES = 56
SNAPSHOT_DIR = './snapshots/psp'
DATA_DIRECTORY = './atlantis'
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 30
POWER = 0.9
RESTORE_FROM = './models/resnet101_imagenet.pth'


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


# def lr_poly(base_lr, iter, max_iter, power):
#     return base_lr * ((1 - float(iter) / max_iter) ** (power))


# def adjust_learning_rate(optimizer, i_iter, total_steps):
#     lr = lr_poly(args.learning_rate, i_iter, total_steps, args.power)
#     optimizer.param_groups[0]['lr'] = lr
#     if len(optimizer.param_groups) > 1:
#         optimizer.param_groups[1]['lr'] = lr * 10


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    input_size = args.input_size

    if args.model == 'PSPNet':
        model = PSPNet(img_channel=3, num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    # saved_state_dict = saved_state_dict["model_state"]
    new_params = model.state_dict().copy()

    for key, value in saved_state_dict.items():
        if (key.split(".")[0] not in ["head", "dsn", "fc"]):
            # print(key)
            new_params[key] = value

    model.load_state_dict(new_params, strict=False)
    # print(model)

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

    train_joint_transform = joint_transforms.Compose(
        train_joint_transform_list)

    trainloader = data.DataLoader(AtlantisDataSet(args.data_dir, split='train', joint_transform=train_joint_transform),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size, input_size),
                         mode='bilinear', align_corners=True)

    i_iter = 0
    print(len(trainloader))
    for epoch in range(args.num_epochs):
        for images, labels, _, _, _ in trainloader:

            i_iter += args.batch_size
            optimizer.zero_grad()

            lr = lr_poly(args.learning_rate, i_iter, args.num_epochs *
                         len(trainloader) * args.batch_size, args.power)
            adjust_learning_rate(optimizer, lr)

            images = images.cuda()
            labels = labels.long().cuda()

            aux, pred = model(images)
            pred = interp(pred)
            aux = interp(aux)

            loss = seg_loss(pred, labels) + 0.4 * seg_loss(aux, labels)
            loss.backward()
            optimizer.step()

            print(
                f'epoch = {epoch:4d}, iter = {i_iter:6d}/{args.num_epochs * len(trainloader):6d}, loss_seg = {loss:.3f}, lr = {lr:.6f}')
        torch.save(model.state_dict(), osp.join(
            args.snapshot_dir, 'epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    main()
