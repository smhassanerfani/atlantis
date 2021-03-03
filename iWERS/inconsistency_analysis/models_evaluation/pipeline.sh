#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

dir="/home/serfani/Desktop/inconsistency_analysis";


python compute_iou.py --gt_dir ${dir}/first_edition/ --pred_dir ${dir}/second_edition/tripp/SegmentationID --split tripp --devkit_dir ./ #> ${dir}/itripp.txt
# python compute_iou.py --gt_dir ${dir}/first_edition/ --pred_dir ${dir}/second_edition/ammar/SegmentationID --split ammar --devkit_dir ./ > ${dir}/iammar.txt
# python compute_iou.py --gt_dir ${dir}/first_edition/ --pred_dir ${dir}/second_edition/ashlin/SegmentationID --split ashlin --devkit_dir ./ > ${dir}/iashlin.txt