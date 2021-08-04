dir=/home/serfani/Downloads/atlantis/natural;
sdirs=$(echo $(ls ${dir}));

dir2=/home/serfani/Downloads/dataset;

for ssdir in ${sdirs}; do
	# echo ${ssdir}
	cp -r ${dir}/${ssdir}/train_images ${dir2}/images/train/${ssdir};
	cp -r ${dir}/${ssdir}/train_masks ${dir2}/masks/train/${ssdir};

	cp -r ${dir}/${ssdir}/test_images ${dir2}/images/test/${ssdir};
	cp -r ${dir}/${ssdir}/test_masks ${dir2}/masks/test/${ssdir};

	cp -r ${dir}/${ssdir}/val_images ${dir2}/images/val/${ssdir};
	cp -r ${dir}/${ssdir}/val_masks ${dir2}/masks/val/${ssdir};
done