#!/usr/bin/bash

venv="./venv/bin/activate";
source ${venv};

data_root="./"
api_key="000000000000000000000000000000";
labels_list="dam spillway"
image_number=5

python images_downloader.py -k $api_key -i $data_root -l $labels_list -n $image_number