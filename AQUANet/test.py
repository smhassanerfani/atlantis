
import os
import argparse
import numpy as np
from PIL import Image
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from model.psp import PSPNet
from model.psp2 import PSP2Net
from model.psp3 import PSP3Net
from model.psp4 import PSP4Net
from model.psp5 import PSP5Net
from model.sp import SPNet
from model.sp2 import SP2Net
from model.sp3 import SP3Net
from model.sp4 import SP4Net
from model.sp5 import SP5Net
from model.ft import FTNet
from model.fer import FERNet
from model.tpsp import TPSPNet
from model.ema import EMANet
from model.ocr import OCRNet
from model.oc import OCNet
from model.deeplabv3 import DeepLabV3
from model.cc import CCNet
from model.acf import ACFNet
from model.da import DANet
from model.ann import ANNet
from model.setr import SETRModel

from AtlantisLoader import AtlantisDataSet
from AtlantisLoader import Atlantis36DataSet
from AtlantisLoader import Atlantis56DataSet



palette = [0,0,0,128,0,0,0,128,0,128,128,0,0,0,128,128,0,128,0,128,128,128,128,128,64,0,0,192,0,0,64,128,0,192,128,0,
           64,0,128,192,0,128,64,128,128,192,128,128,0,64,0,128,64,0,0,192,0,128,192,0,0,64,128,128,64,128,0,192,128,
           128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,128,192,192,128,0,0,64,128,0,
           64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,
           192,192,0,192,64,128,192,192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
           128,192,192,64,64,64]

id_to_colorid = {3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 42: 28, 43: 29,
                44: 30, 45: 31, 53: 32, 54: 33, 55: 34, 56: 35,  1: 36,  2: 37,  5: 38,  6: 39,
                 8: 40, 14: 41, 15: 42, 25: 43, 27: 44, 28: 45, 31: 46, 37: 47, 41: 48, 46: 49,
                47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55}

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask, num_classes):
    mask_copy = np.zeros_like(mask)
    if num_classes==56:
        for k, v in id_to_colorid.items():
            mask_copy[mask == (k-1)] = v
    else:
        mask_copy = mask
    new_mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

MODEL = 'OCNet'
NAME = 'oc'
SPLIT = 'val'
NUM_CLASSES = 36
BATCH_SIZE = 1
NUM_WORKERS = 1
PADDING_SIZE = '768'
DATA_DIRECTORY = './data/atlantis'
SAVE_PATH = './result/'+str(NUM_CLASSES)+'/'+SPLIT+'/'+NAME
RESTORE_FROM = './snapshots/'+str(NUM_CLASSES)+'/'+NAME+'/epoch29.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--padding-size", type=int, default=PADDING_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : PSPNet")
    parser.add_argument("--split", type=str, default=SPLIT,
                        help="test or val")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Path to save result.")
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

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
    if args.model == 'EMANet':
        model = EMANet(num_classes=args.num_classes)
    if args.model == 'FENet':
        model = FENet(num_classes=args.num_classes)
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



    model.eval()
    model.cuda()

    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    cudnn.enabled = True
    cudnn.benchmark = True


    if args.num_classes == 35:
        testloader = data.DataLoader(
            AtlantisDataSet(args.data_dir, split=args.split, padding_size = args.padding_size),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif args.num_classes == 36:
        testloader = data.DataLoader(
            Atlantis36DataSet(args.data_dir, split=args.split, padding_size = args.padding_size),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        testloader = data.DataLoader(
            Atlantis56DataSet(args.data_dir, split=args.split, padding_size = args.padding_size),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp = nn.Upsample(size=(args.padding_size, args.padding_size), mode='bilinear', align_corners=True)


    for images, labels, name, width, height in testloader:
        images = images.cuda()
        images = F.upsample(images, [640, 640], mode='bilinear')
        with torch.no_grad():
            _, pred = model(images)
        pred = interp(pred).cpu().data[0].numpy().transpose(1,2,0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)


        labels = np.asarray(labels.squeeze(0), dtype=np.uint8)

        top_pad = args.padding_size - height
        right_pad = args.padding_size - width
        pred = pred[top_pad:, :-right_pad]


        pred_col = colorize_mask(pred, args.num_classes)
        label_col = colorize_mask(labels, args.num_classes)
        imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)
        if args.split == 'val':
            pred_col.save('%s/%s_color.png' % (args.save_path, name[0][:-4]))
            label_col.save('%s/%s_gt.png' % (args.save_path, name[0][:-4]))


if __name__ == '__main__':
    main()
    os.system('python compute_iou.py --split ' + SPLIT + ' --pred_dir ' + SAVE_PATH + ' --num-classes ' + str(NUM_CLASSES))
