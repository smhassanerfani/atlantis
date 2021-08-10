#! /bin/bash

# origin directory & sub directories
odir=/home/serfani/Downloads/atlantis/s1n;
subdirs=$(echo $(ls ${odir}));

# destination directory
ddir=/home/serfani/Downloads/atlantis_new;


sdirs=("delta" "estuary" "flood" "glaciers")
ntrains=(1155 439 989 844)
nvals=(165 63 141 121)
ntests=(330 125 283 240)

for i in "${!sdirs[@]}"; do 
	echo $i
	sdir=${sdirs[$i]}; ntrain=${ntrains[$i]}; nval=${nvals[$i]}; ntest=${ntests[$i]};
	find ${odir}/${sdir}/*.jpg | cut -d "/" -f 7 | shuf > ${odir}/${sdir}_images.txt
	
	cat ${odir}/${sdir}_images.txt | head -n ${ntrain} > ${odir}/${sdir}_train_images.txt
	cat ${odir}/${sdir}_images.txt | tail -n ${ntest} > ${odir}/${sdir}_test_images.txt
	cat ${odir}/${sdir}_images.txt | head -n -${ntest}| tail -n ${nval} > ${odir}/${sdir}_val_images.txt

	rsync --files-from=${odir}/${sdir}_train_images.txt ${odir}/${sdir} ${ddir}/train/${sdir}
	rsync --files-from=${odir}/${sdir}_test_images.txt ${odir}/${sdir} ${ddir}/test/${sdir}
	rsync --files-from=${odir}/${sdir}_val_images.txt ${odir}/${sdir} ${ddir}/val/${sdir}



