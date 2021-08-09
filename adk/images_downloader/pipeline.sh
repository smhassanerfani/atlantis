#!/usr/bin/bash

venv="./venv/bin/activate";
source ${venv};

data_root="./"
api_key="da03d4d2c9d70753c5b918009711b9ea";
label_list="dam spillway"
image_number=5

python images_downloader.py -k $api_key -i $data_root -l $label_list -n $image_number