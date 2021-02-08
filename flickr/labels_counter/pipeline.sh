#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

dir="/home/serfani/Downloads/atlantis/natural";
sdirs=$(echo $(ls ${dir}));

for sdir in ${sdirs}; do
	python class_count.py ${dir}/${sdir};
done
