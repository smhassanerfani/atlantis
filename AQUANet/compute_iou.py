import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os

id_to_trainid35 = {3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 43: 28, 44: 29,
                45: 30, 53: 31, 54: 32, 55: 33, 56: 34}
id_to_trainid36 = { 3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                    18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                    32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 42: 28, 43: 29,
                    44: 30, 45: 31, 53: 32, 54: 33, 55: 34, 56: 35}

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def compute_mIoU(gt_dir, pred_dir, split, devkit_dir, num_classes):
    # get the gt and pred
    gt_imgs = []
    pred_imgs = []
    masks_base = os.path.join(gt_dir, "masks", split)
    for root, dirs, files in os.walk(masks_base, topdown=True):
        for name in files:
            if name.endswith(".png"):
                gt_imgs.append(os.path.join(root, name))
                pred_imgs.append(os.path.join(pred_dir, name))

    # get the num_classes and class name
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    if num_classes == 35:
        name_classes = np.array(info['label_w35'], dtype=np.str)
    if num_classes == 36:
        name_classes = np.array(info['label_w36'], dtype=np.str)
    if num_classes == 56:
        name_classes = np.array(info['label'], dtype=np.str)

    hist = np.zeros((num_classes, num_classes))
    correct = 0
    totalpixel = 0

    for ind in range(len(gt_imgs)):
        try:
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))
        except:
            print ("We don't have the prediction of ", gt_imgs[ind].split('/')[-1] )
            continue
        if num_classes == 35:
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
            for k, v in id_to_trainid35.items():
                label_copy[label == k] = v
        if num_classes == 36:
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
            for k, v in id_to_trainid36.items():
                label_copy[label == k] = v
        if num_classes == 56:
            label_copy = label - 1
            label_copy[label_copy == -1] = 255
        hist += fast_hist(label_copy.flatten(), pred.flatten(), num_classes)
        totalpixel += (label_copy != 255).sum().item()
        correct += (label_copy==pred).sum().item()
    acc = correct/totalpixel
    mIoUs = per_class_iu(hist)

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))

    index_NaNs = np.isnan(mIoUs)
    mIoUs[index_NaNs] = 0.0
    print('===> mIoU: ' + str(round(np.mean(mIoUs) * 100, 2)))
    print('===> acc: ' + str(round(acc, 4)))

    for ind_class in range(num_classes):
        print('& ' + str(int(round(mIoUs[ind_class] * 100, 0)))+' ',end='')
    print ('&')
    return mIoUs

def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.split, args.devkit_dir, args.num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='./data/atlantis', type=str, help='directory of gt')
    parser.add_argument('--pred_dir', default='./result/56/test/sp', type=str, help='directory of pred')
    parser.add_argument('--split', default='test', type=str, help='val or test')
    parser.add_argument('--num-classes', default=56, type=int, help='number of classes')
    parser.add_argument('--devkit_dir', default='./', help='base information')
    args = parser.parse_args()
    main(args)

