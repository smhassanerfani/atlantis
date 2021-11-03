#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

gt="/home/serfani/Documents/atlantis/atlantis/masks/val";
pd1="/home/serfani/Documents/atlantis/snapshots/val_predictions/pspnet_imagenet/";
pd2="/home/serfani/Documents/atlantis/snapshots/val_predictions/pspnet_atex/";

python compute_iou.py -gt ${gt} -pd ${pd1} -j "./" > val_imagenet_.out
python compute_iou.py -gt ${gt} -pd ${pd2} -j "./" > val_atex_.out


