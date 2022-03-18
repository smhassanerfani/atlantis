import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os


def _fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def _iIoU(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def _iACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(axis=1)


def _mACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist).sum() / hist.sum()


def compute_mIoU(gt_dir, pred_dir, json_dir="./"):
    # get the gt and pred
    gt_imgs = []
    pred_imgs = []

    masks_base = gt_dir
    preds_base = pred_dir

    for root, dirs, files in os.walk(masks_base, topdown=True):
        for file in files:
            if file.endswith(".png"):
                gt_imgs.append(os.path.join(root, file))
                pred_imgs.append(os.path.join(preds_base, file))

    with open(join(json_dir, 'labels_info.json'), 'r') as jf:
        info = json.load(jf)

    num_classes = len(info)
    name_classes = [entity["name"] for entity in info]

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        try:
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))

            label = label - 1
            label[label == -1] = 255

        except:
            print("We don't have the prediction of ",
                  gt_imgs[ind].split('/')[-1])
            continue

        hist += _fast_hist(label.flatten(), pred.flatten(), num_classes)

    iIoUs = _iIoU(hist)
    iACCs = _iACC(hist)

    for ind_class in range(num_classes):
        print(name_classes[ind_class] +
              '\t' + str(round(iIoUs[ind_class] * 100, 2)) +
              '\t' + str(round(iACCs[ind_class] * 100, 2)))

    print('mIoU' + '\t' + str(round(np.nanmean(iIoUs) * 100, 2)))
    print('mACC' + '\t' + str(round(_mACC(hist) * 100, 2)))


def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, json_dir=args.json_dir)


def get_arguments(
    gt_dir="../atlantis/masks/test",
    pred_dir="../snapshots/test_review_results_epoch30",
    json_dir="./"
):
    parser = argparse.ArgumentParser(description="Calculation of mIoU")
    parser.add_argument('-gt', '--gt-dir', default=gt_dir,
                        type=str, help='directory of inconsistency analysis data')
    parser.add_argument('-pd', '--pred-dir', default=pred_dir,
                        type=str, help='directory of annotator')
    parser.add_argument('-j', '--json-dir', default=json_dir,
                        type=str, help='labels\' ID information')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
