#! /bin/bash

# origin directory & sub directories
odir=/home/serfani/Downloads/atlantis_raw/s1n;

# destination directory
ddir=/home/serfani/Downloads/atlantis_new;


sdirs=("cliff" "cypress_tree" "fjord" "flood");
ntrains=(28 28 62 81);
nvals=(5 4 9 12);
ntests=(8 8 17 23);

for i in "${!sdirs[@]}"; do 
	echo $i;
	sdir=${sdirs[$i]}; ntrain=${ntrains[$i]}; nval=${nvals[$i]}; ntest=${ntests[$i]};
	find ${odir}/${sdir}/images/*.jpg | cut -d "/" -f 9 | shuf > ${odir}/${sdir}/images_list.txt;
	cat ${odir}/${sdir}/images_list.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${odir}/${sdir}/masks_list.txt;
	
	
	cat ${odir}/${sdir}/images_list.txt | head -n ${ntrain} > ${odir}/${sdir}/images_train_list.txt; 
	cat ${odir}/${sdir}/images_list.txt | tail -n ${ntest} > ${odir}/${sdir}/images_test_list.txt; 
	cat ${odir}/${sdir}/images_list.txt | head -n -${ntest} | tail -n ${nval} > ${odir}/${sdir}/images_val_list.txt;

	cat ${odir}/${sdir}/masks_list.txt | head -n ${ntrain} > ${odir}/${sdir}/masks_train_list.txt; 
	cat ${odir}/${sdir}/masks_list.txt | tail -n ${ntest} > ${odir}/${sdir}/masks_test_list.txt;
	cat ${odir}/${sdir}/masks_list.txt | head -n -${ntest} | tail -n ${nval} > ${odir}/${sdir}/masks_val_list.txt;
	

	rsync --files-from=${odir}/${sdir}/images_train_list.txt ${odir}/${sdir}/images ${ddir}/images/train/${sdir};
	rsync --files-from=${odir}/${sdir}/images_test_list.txt ${odir}/${sdir}/images ${ddir}/images/test/${sdir};
	rsync --files-from=${odir}/${sdir}/images_val_list.txt ${odir}/${sdir}/images ${ddir}/images/val/${sdir};
	
	rsync --files-from=${odir}/${sdir}/masks_train_list.txt ${odir}/${sdir}/masks ${ddir}/masks/train/${sdir};
	rsync --files-from=${odir}/${sdir}/masks_test_list.txt ${odir}/${sdir}/masks ${ddir}/masks/test/${sdir};
	rsync --files-from=${odir}/${sdir}/masks_val_list.txt ${odir}/${sdir}/masks ${ddir}/masks/val/${sdir};
done

