
import os
import argparse
import numpy as np
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from utils.palette import colorize_mask
from models.pspnet import PSPNet
from AtlantisLoader import AtlantisDataSet


MODEL = 'PSPNet'
NAME = 'psp56'
SPLIT = 'val'
NUM_CLASSES = 56
BATCH_SIZE = 1
NUM_WORKERS = 1
PADDING_SIZE = '768'
DATA_DIRECTORY = './atlantis'
SAVE_PATH = './snapshots/csi_atex'
RESTORE_FROM = "/home/serfani/Documents/atlantis/snapshots/pspnet_state_dict/epoch29_atex.pth"


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

def main():

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)

    model.eval()
    model.cuda()

    saved_state_dict = torch.load(args.restore_from)
    # model_dict = model.state_dict()
    # saved_state_dict = {k: v for k,
    #                     v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    cudnn.enabled = True
    cudnn.benchmark = True

    testloader = data.DataLoader(
        AtlantisDataSet(args.data_dir, split=args.split,
                        padding_size=args.padding_size),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp = nn.Upsample(
        size=(args.padding_size, args.padding_size), mode='bilinear', align_corners=True)

    for images, labels, name, width, height in tqdm(testloader):
        images = images.cuda()
        images = F.upsample(images, [640, 640], mode='bilinear')
        with torch.no_grad():
            _, pred = model(images)
        pred = interp(pred).cpu().data[0].numpy().transpose(1, 2, 0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)

        labels = np.asarray(labels.squeeze(0), dtype=np.uint8)

        top_pad = args.padding_size - height
        right_pad = args.padding_size - width
        pred = pred[top_pad:, :-right_pad]
        # print(np.unique(pred))
        # exit()
        pred_col = colorize_mask(pred, args.num_classes)
        label_col = colorize_mask(labels, args.num_classes)
        imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)
        if args.split != 'test':
            pred_col.save('%s/%s_color.png' % (args.save_path, name[0][:-4]))
            label_col.save('%s/%s_gt.png' % (args.save_path, name[0][:-4]))

    print("finish")


if __name__ == '__main__':
    main()
    # os.system('python3 compute_iou.py --split ' + SPLIT +
    #           ' --pred_dir ' + SAVE_PATH + ' --num-classes ' + str(NUM_CLASSES))
