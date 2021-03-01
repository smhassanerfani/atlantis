#!/usr/bin/bash

# venv="/home/serfani/Documents/venv/bin/activate";
# source ${venv};

dir="/home/serfani/Desktop/atlantis_analysis";
sdirs="canal ditch fjord flood glaciers hot_spring lake puddle rapids reservoir river river_delta sea snow spillway swimming_pool waterfall wetland"

for sdir in ${sdirs}; do
	cp -r ${dir}/${sdir}/waterbody /home/serfani/Desktop/atlantis_texture/${sdir}
	# python pyscript.py ${dir}/${sdir};
	
done