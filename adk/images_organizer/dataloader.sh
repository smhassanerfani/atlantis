#! /bin/bash

dir=/home/serfani/Downloads/atex_analysis;
subdirs=$(echo $(ls ${dir}));

ddir=/home/serfani/Downloads/atex;

# for sd in ${subdirs}; do
# 	paste <(echo ${sd}) <(echo $(ls ${sd}) | wc)
# done;

sdirs=("delta" "estuary" "flood" "glaciers")
ntrains=(1155 439 989 844)
nvals=(165 63 141 121)
ntests=(330 125 283 240)

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



