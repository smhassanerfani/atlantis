#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

gt="/home/serfani/Documents/atlantis/atlantis/masks/val";
pd="/home/serfani/Documents/atlantis/snapshots/pspnet480_ep29_imgnet";

python compute_iou.py -gt ${gt} -pd ${pd} -j "./"


