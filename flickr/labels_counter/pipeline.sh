#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

dir="/home/serfani/Downloads/appendix/raw";
sdirs=$(echo $(ls ${dir}));

for sdir in ${sdirs}; do
	python color2id.py ${dir}/${sdir};
	python labels_counter.py ${dir}/${sdir};

done
