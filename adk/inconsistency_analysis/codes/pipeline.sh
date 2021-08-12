#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

dir="/home/serfani/Documents/atlantis/adk/inconsistency_analysis/data";

python compute_iou.py -d ${dir} -a annotator1 -j "./" > ann1_total.csv
python compute_iou.py -d ${dir} -a annotator1 -i -j "./" > ann1_indiv.csv

python compute_iou.py -d ${dir} -a annotator2 -j "./" > ann2_total.csv
python compute_iou.py -d ${dir} -a annotator2 -i -j "./" > ann2_indiv.csv

python compute_iou.py -d ${dir} -a annotator3 -j "./" > ann3_total.csv
python compute_iou.py -d ${dir} -a annotator3 -i -j "./" > ann3_indiv.csv
