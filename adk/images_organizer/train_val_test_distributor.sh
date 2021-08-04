#! /bin/bash

dir=/home/serfani/Downloads/atex_analysis;
subdirs=$(echo $(ls ${dir}));

ddir=/home/serfani/Downloads/atex;

# for sd in ${subdirs}; do
# 	paste <(echo ${sd}) <(echo $(ls ${sd}) | wc)
# done;

sdirs=("delta" "estuary" "flood" "glaciers" "hot_spring" "lake" "pool" "puddle" "rapids" "river" "sea" "snow" "swamp" "waterfall" "wetland")
ntrains=(1155 439 989 844 701 536 277 588 768 256 379 501 658 337 325)
nvals=(165 63 141 121 100 77 40 84 110 37 54 72 94 48 46)
ntests=(330 125 283 240 201 153 79 168 219 73 108 142 188 96 93)

for i in "${!sdirs[@]}"; do 
	echo $i
	sdir=${sdirs[$i]}; ntrain=${ntrains[$i]}; nval=${nvals[$i]}; ntest=${ntests[$i]};
	find ${dir}/${sdir}/*.jpg |cut -d "/" -f 7 | shuf > ${dir}/${sdir}_images.txt
	
	cat ${dir}/${sdir}_images.txt | head -n ${ntrain} > ${dir}/${sdir}_train_images.txt
	cat ${dir}/${sdir}_images.txt | tail -n ${ntest} > ${dir}/${sdir}_test_images.txt
	cat ${dir}/${sdir}_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/${sdir}_val_images.txt

	rsync --files-from=${dir}/${sdir}_train_images.txt ${dir}/${sdir} ${ddir}/train/${sdir}
	rsync --files-from=${dir}/${sdir}_test_images.txt ${dir}/${sdir} ${ddir}/test/${sdir}
	rsync --files-from=${dir}/${sdir}_val_images.txt ${dir}/${sdir} ${ddir}/val/${sdir}



