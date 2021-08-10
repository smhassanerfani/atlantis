#! /usr/bin/bash

# old directory
odir=/home/serfani/Downloads/atlantis/images;

# raw directory
rdir=/home/serfani/Downloads/appendix/raw;

# new directory
ndir=/home/serfani/Downloads/appendix/final;

# list of sub directories
subdirs=$(echo $(ls ${rdir}));

for sdir in ${subdirs}; do
    # copying the list of images in train test and val dirs to the correspoing text files
    ls ${odir}/test/${sdir} > ${rdir}/${sdir}/test_images.txt
    ls ${odir}/train/${sdir} > ${rdir}/${sdir}/train_images.txt
    ls ${odir}/val/${sdir} > ${rdir}/${sdir}/val_images.txt

    # merge the list of images in train test val to get the total list of images
    paste -s -d "\n" ${rdir}/${sdir}/train_images.txt ${rdir}/${sdir}/val_images.txt ${rdir}/${sdir}/test_images.txt > ${rdir}/${sdir}/shuf_images.txt

    # remove the redundant images from images directory based on the total list
    cd ${rdir}/${sdir}
    comm -2 -3 <(ls images/) <(sort shuf_images.txt) | sed 's/^/.\/images\//' | xargs rm

    # remove the redundant masks based on the total number if images existing in image direcotry
    comm -1 -3 <(ls images/ | cut -d "." -f 1) <(ls masks/ | cut -d "." -f 1) | sed 's/$/.png/' | sed 's/^/.\/masks\//' | xargs rm
    diff -u <(ls images/ | cut -d "." -f 1) <(ls masks/ | cut -d "." -f 1)

    # creating list of train test val for masks
    cat ${rdir}/${sdir}/train_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${rdir}/${sdir}/train_masks.txt;
    cat ${rdir}/${sdir}/test_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${rdir}/${sdir}/test_masks.txt;
    cat ${rdir}/${sdir}/val_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${rdir}/${sdir}/val_masks.txt;

    # copy images and masks based on the lists from destination to origin
    rsync --files-from=${rdir}/${sdir}/train_images.txt ${rdir}/${sdir}/images ${ndir}/images/train/${sdir};
    rsync --files-from=${rdir}/${sdir}/test_images.txt ${rdir}/${sdir}/images ${ndir}/images/test/${sdir};
    rsync --files-from=${rdir}/${sdir}/val_images.txt ${rdir}/${sdir}/images ${ndir}/images/val/${sdir};
    rsync --files-from=${rdir}/${sdir}/train_masks.txt ${rdir}/${sdir}/SegmentationID ${ndir}/masks/train/${sdir};
    rsync --files-from=${rdir}/${sdir}/test_masks.txt ${rdir}/${sdir}/SegmentationID ${ndir}/masks/test/${sdir};
    rsync --files-from=${rdir}/${sdir}/val_masks.txt ${rdir}/${sdir}/SegmentationID ${ndir}/masks/val/${sdir};

done
