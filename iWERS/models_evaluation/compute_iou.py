import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def compute_mIoU(gt_dir, pred_dir, split, devkit_dir):
    # get the gt and pred
    gt_imgs = []
    pred_imgs = []
    masks_base = os.path.join(gt_dir, "masks", split)
    for root, dirs, files in os.walk(masks_base, topdown=True):
        for name in files:
            if name.endswith(".png"):
                gt_imgs.append(os.path.join(root, name))
                pred_imgs.append(os.path.join(pred_dir, name.replace(".png", "_pred.png")))
            
	
    # get the num_classes and class name
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        try:
           pred = np.array(Image.open(pred_imgs[ind]))
           label = np.array(Image.open(gt_imgs[ind]))
        except:
           print ("We don't have the prediction of ", gt_imgs[ind].split('/')[-1] )
           continue
        label[label==0] = 256
        label = label - 1

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    mIoUs = per_class_iu(hist)

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))

    index_NaNs = np.isnan(mIoUs)
    mIoUs[index_NaNs] = 0.0
    print('===> mIoU: ' + str(round(np.mean(mIoUs) * 100, 2)))
    return mIoUs

def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.split, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='./data/atlantis', type=str, help='directory of gt')
    parser.add_argument('--pred_dir', default='./models/deeplabv3_resnet50_ndata/predictions', type=str, help='directory of pred')
    parser.add_argument('--split', default='val', type=str, help='val or test')
    parser.add_argument('--devkit_dir', default='./', help='base information')
    args = parser.parse_args()
    main(args)

