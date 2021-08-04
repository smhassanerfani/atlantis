#! /usr/bin/bash

ndir=/home/serfani/Downloads/appendix/raw;
odir=/home/serfani/Documents/Microsoft_project/iWERS/data/atlantis/images;
dir2=/home/serfani/Downloads/appendix/final;

subdirs=$(echo $(ls ${ndir}));
for sd in ${subdirs}; do
    # copying the list of images in train test and val dirs to the correspoing text files
    ls ${odir}/test/${sd} > ${ndir}/${sd}/test_images.txt
    ls ${odir}/train/${sd} > ${ndir}/${sd}/train_images.txt
    ls ${odir}/val/${sd} > ${ndir}/${sd}/val_images.txt

    # merge the list of images in train test val to get the total list of images
    paste -s -d "\n" ${ndir}/${sd}/train_images.txt ${ndir}/${sd}/val_images.txt ${ndir}/${sd}/test_images.txt > ${ndir}/${sd}/shuf_images.txt

    # remove the redundant images from images directory based on the total list
    cd ${ndir}/${sd}
    comm -2 -3 <(ls images/) <(sort shuf_images.txt) | sed 's/^/.\/images\//' | xargs rm

    # remove the redundant masks based on the total number if images existing in image direcotry
    comm <(ls images/ | cut -d "." -f 1) <(ls masks/ | cut -d "." -f 1)
    comm -1 -3 <(ls images/ | cut -d "." -f 1) <(ls masks/ | cut -d "." -f 1) | sed 's/$/.png/' | sed 's/^/.\/masks\//' | xargs rm
    diff -u <(ls images/ | cut -d "." -f 1) <(ls masks/ | cut -d "." -f 1)

    # creating list of train test val for masks
    cat ${ndir}/${sd}/train_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${ndir}/${sd}/train_masks.txt;
    cat ${ndir}/${sd}/test_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${ndir}/${sd}/test_masks.txt;
    cat ${ndir}/${sd}/val_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${ndir}/${sd}/val_masks.txt;

    rsync --files-from=${ndir}/${sd}/train_images.txt ${ndir}/${sd}/images ${dir2}/images/train/${sd};
    rsync --files-from=${ndir}/${sd}/test_images.txt ${ndir}/${sd}/images ${dir2}/images/test/${sd};
    rsync --files-from=${ndir}/${sd}/val_images.txt ${ndir}/${sd}/images ${dir2}/images/val/${sd};
    rsync --files-from=${ndir}/${sd}/train_masks.txt ${ndir}/${sd}/SegmentationID ${dir2}/masks/train/${sd};
    rsync --files-from=${ndir}/${sd}/test_masks.txt ${ndir}/${sd}/SegmentationID ${dir2}/masks/test/${sd};
    rsync --files-from=${ndir}/${sd}/val_masks.txt ${ndir}/${sd}/SegmentationID ${dir2}/masks/val/${sd};

done
