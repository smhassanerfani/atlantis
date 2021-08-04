#!/usr/bin/bash

venv="/home/serfani/Documents/venv/bin/activate";
source ${venv};

dir="/home/serfani/Downloads/atlantis_analysis";
python spatial_analysis.py ${dir}/pipeline;

# sdirs=$(echo $(ls ${dir}));

# for sdir in ${sdirs}; do
# 	python spatial_analysis.py ${dir}/${sdir};
	
# done
