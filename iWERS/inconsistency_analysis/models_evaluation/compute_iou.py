import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def iIoU(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def iACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(axis=1)


def mACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist).sum() / hist.sum()


def compute_mIoU(gt_dir, pred_dir, split, devkit_dir):
    # get the gt and pred
    gt_imgs = []
    pred_imgs = []
    masks_base = os.path.join(gt_dir, split, "images_list.txt")
    # masks_base = os.path.join(gt_dir, "ground_truth")

    with open(masks_base, 'r') as txtfile:
        images_list = [line.rstrip('\n') for line in txtfile]

    for root, dirs, files in os.walk(pred_dir, topdown=True):
        # for root, dirs, files in os.walk(gt_dir, topdown=True):
        for name in files:
            if name.endswith(".png"):
                if name in images_list:
                    gt_imgs.append(os.path.join(gt_dir, split, "masks", name))
                    pred_imgs.append(os.path.join(pred_dir, name))

    # get the num_classes and class name
    with open(join(devkit_dir, 'labels_ID.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = len(info)
    name_classes = [key for key in info.keys()]

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        try:
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))
        except:
            print("We don't have the prediction of ",
                  gt_imgs[ind].split('/')[-1])
            continue

        # label = label - 1
        # label[label == -1] = 255

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    iIoUs = iIoU(hist)
    iACCs = iACC(hist)

    for ind_class in range(num_classes):
        print(name_classes[ind_class] +
              '\t' + str(round(iIoUs[ind_class] * 100, 2)) +
              '\t' + str(round(iACCs[ind_class] * 100, 2)))

    print('mIoU' + '\t' + str(round(np.nanmean(iIoUs) * 100, 2)))
    print('mACC' + '\t' + str(round(mACC(hist) * 100, 2)))
    return iIoUs


def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, args.split, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='./data/atlantis',
                        type=str, help='directory of gt')
    parser.add_argument('--pred_dir', default='./models/predictions',
                        type=str, help='directory of pred')
    parser.add_argument('--split', default='val', type=str, help='val or test')
    parser.add_argument('--devkit_dir', default='./',
                        type=str, help='base information')
    args = parser.parse_args()
    main(args)
