#! /bin/bash

dir=/home/serfani/Downloads/atlantis/natural;

find ${dir}/cliff/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/cliff/shuf_images.txt
# cat ${dir}/cliff/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${dir}/cliff/shuf_masks.txt

ntrain=28; nval=5; ntest=8; 

cat ${dir}/cliff/shuf_images.txt | head -n ${ntrain} > ${dir}/cliff/train_images.txt
cat ${dir}/cliff/shuf_images.txt | tail -n ${ntest} > ${dir}/cliff/test_images.txt
cat ${dir}/cliff/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/cliff/val_images.txt
cat ${dir}/cliff/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/cliff/train_masks.txt
cat ${dir}/cliff/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/cliff/test_masks.txt
cat ${dir}/cliff/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/cliff/val_masks.txt

rsync --files-from=${dir}/cliff/train_images.txt ${dir}/cliff/images ${dir}/cliff/train_images
rsync --files-from=${dir}/cliff/test_images.txt ${dir}/cliff/images ${dir}/cliff/test_images
rsync --files-from=${dir}/cliff/val_images.txt ${dir}/cliff/images ${dir}/cliff/val_images
rsync --files-from=${dir}/cliff/train_masks.txt ${dir}/cliff/masks ${dir}/cliff/train_masks
rsync --files-from=${dir}/cliff/test_masks.txt ${dir}/cliff/masks ${dir}/cliff/test_masks
rsync --files-from=${dir}/cliff/val_masks.txt ${dir}/cliff/masks ${dir}/cliff/val_masks

find ${dir}/cypress_swamp/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/cypress_swamp/shuf_images.txt

ntrain=28; nval=4; ntest=8; 

cat ${dir}/cypress_swamp/shuf_images.txt | head -n ${ntrain} > ${dir}/cypress_swamp/train_images.txt
cat ${dir}/cypress_swamp/shuf_images.txt | tail -n ${ntest} > ${dir}/cypress_swamp/test_images.txt
cat ${dir}/cypress_swamp/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/cypress_swamp/val_images.txt
cat ${dir}/cypress_swamp/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/cypress_swamp/train_masks.txt
cat ${dir}/cypress_swamp/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/cypress_swamp/test_masks.txt
cat ${dir}/cypress_swamp/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/cypress_swamp/val_masks.txt

rsync --files-from=${dir}/cypress_swamp/train_images.txt ${dir}/cypress_swamp/images ${dir}/cypress_swamp/train_images
rsync --files-from=${dir}/cypress_swamp/test_images.txt ${dir}/cypress_swamp/images ${dir}/cypress_swamp/test_images
rsync --files-from=${dir}/cypress_swamp/val_images.txt ${dir}/cypress_swamp/images ${dir}/cypress_swamp/val_images
rsync --files-from=${dir}/cypress_swamp/train_masks.txt ${dir}/cypress_swamp/masks ${dir}/cypress_swamp/train_masks
rsync --files-from=${dir}/cypress_swamp/test_masks.txt ${dir}/cypress_swamp/masks ${dir}/cypress_swamp/test_masks
rsync --files-from=${dir}/cypress_swamp/val_masks.txt ${dir}/cypress_swamp/masks ${dir}/cypress_swamp/val_masks

find ${dir}/fjord/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/fjord/shuf_images.txt

ntrain=62; nval=9; ntest=17;

cat ${dir}/fjord/shuf_images.txt | head -n ${ntrain} > ${dir}/fjord/train_images.txt
cat ${dir}/fjord/shuf_images.txt | tail -n ${ntest} > ${dir}/fjord/test_images.txt
cat ${dir}/fjord/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/fjord/val_images.txt
cat ${dir}/fjord/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/fjord/train_masks.txt
cat ${dir}/fjord/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/fjord/test_masks.txt
cat ${dir}/fjord/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/fjord/val_masks.txt

rsync --files-from=${dir}/fjord/train_images.txt ${dir}/fjord/images ${dir}/fjord/train_images
rsync --files-from=${dir}/fjord/test_images.txt ${dir}/fjord/images ${dir}/fjord/test_images
rsync --files-from=${dir}/fjord/val_images.txt ${dir}/fjord/images ${dir}/fjord/val_images
rsync --files-from=${dir}/fjord/train_masks.txt ${dir}/fjord/masks ${dir}/fjord/train_masks
rsync --files-from=${dir}/fjord/test_masks.txt ${dir}/fjord/masks ${dir}/fjord/test_masks
rsync --files-from=${dir}/fjord/val_masks.txt ${dir}/fjord/masks ${dir}/fjord/val_masks

find ${dir}/flood/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/flood/shuf_images.txt

ntrain=81; nval=12; ntest=23;

cat ${dir}/flood/shuf_images.txt | head -n ${ntrain} > ${dir}/flood/train_images.txt
cat ${dir}/flood/shuf_images.txt | tail -n ${ntest} > ${dir}/flood/test_images.txt
cat ${dir}/flood/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/flood/val_images.txt
cat ${dir}/flood/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/flood/train_masks.txt
cat ${dir}/flood/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/flood/test_masks.txt
cat ${dir}/flood/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/flood/val_masks.txt

rsync --files-from=${dir}/flood/train_images.txt ${dir}/flood/images ${dir}/flood/train_images
rsync --files-from=${dir}/flood/test_images.txt ${dir}/flood/images ${dir}/flood/test_images
rsync --files-from=${dir}/flood/val_images.txt ${dir}/flood/images ${dir}/flood/val_images
rsync --files-from=${dir}/flood/train_masks.txt ${dir}/flood/masks ${dir}/flood/train_masks
rsync --files-from=${dir}/flood/test_masks.txt ${dir}/flood/masks ${dir}/flood/test_masks
rsync --files-from=${dir}/flood/val_masks.txt ${dir}/flood/masks ${dir}/flood/val_masks

find ${dir}/glaciers/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/glaciers/shuf_images.txt

ntrain=57; nval=9; ntest=16;

cat ${dir}/glaciers/shuf_images.txt | head -n ${ntrain} > ${dir}/glaciers/train_images.txt
cat ${dir}/glaciers/shuf_images.txt | tail -n ${ntest} > ${dir}/glaciers/test_images.txt
cat ${dir}/glaciers/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/glaciers/val_images.txt
cat ${dir}/glaciers/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/glaciers/train_masks.txt
cat ${dir}/glaciers/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/glaciers/test_masks.txt
cat ${dir}/glaciers/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/glaciers/val_masks.txt

rsync --files-from=${dir}/glaciers/train_images.txt ${dir}/glaciers/images ${dir}/glaciers/train_images
rsync --files-from=${dir}/glaciers/test_images.txt ${dir}/glaciers/images ${dir}/glaciers/test_images
rsync --files-from=${dir}/glaciers/val_images.txt ${dir}/glaciers/images ${dir}/glaciers/val_images
rsync --files-from=${dir}/glaciers/train_masks.txt ${dir}/glaciers/masks ${dir}/glaciers/train_masks
rsync --files-from=${dir}/glaciers/test_masks.txt ${dir}/glaciers/masks ${dir}/glaciers/test_masks
rsync --files-from=${dir}/glaciers/val_masks.txt ${dir}/glaciers/masks ${dir}/glaciers/val_masks

find ${dir}/lake/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/lake/shuf_images.txt

ntrain=84; nval=12; ntest=24;

cat ${dir}/lake/shuf_images.txt | head -n ${ntrain} > ${dir}/lake/train_images.txt
cat ${dir}/lake/shuf_images.txt | tail -n ${ntest} > ${dir}/lake/test_images.txt
cat ${dir}/lake/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/lake/val_images.txt
cat ${dir}/lake/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/lake/train_masks.txt
cat ${dir}/lake/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/lake/test_masks.txt
cat ${dir}/lake/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/lake/val_masks.txt

rsync --files-from=${dir}/lake/train_images.txt ${dir}/lake/images ${dir}/lake/train_images
rsync --files-from=${dir}/lake/test_images.txt ${dir}/lake/images ${dir}/lake/test_images
rsync --files-from=${dir}/lake/val_images.txt ${dir}/lake/images ${dir}/lake/val_images
rsync --files-from=${dir}/lake/train_masks.txt ${dir}/lake/masks ${dir}/lake/train_masks
rsync --files-from=${dir}/lake/test_masks.txt ${dir}/lake/masks ${dir}/lake/test_masks
rsync --files-from=${dir}/lake/val_masks.txt ${dir}/lake/masks ${dir}/lake/val_masks

find ${dir}/mangrove/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/mangrove/shuf_images.txt

ntrain=63; nval=9; ntest=18;

cat ${dir}/mangrove/shuf_images.txt | head -n ${ntrain} > ${dir}/mangrove/train_images.txt
cat ${dir}/mangrove/shuf_images.txt | tail -n ${ntest} > ${dir}/mangrove/test_images.txt
cat ${dir}/mangrove/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/mangrove/val_images.txt
cat ${dir}/mangrove/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/mangrove/train_masks.txt
cat ${dir}/mangrove/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/mangrove/test_masks.txt
cat ${dir}/mangrove/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/mangrove/val_masks.txt

rsync --files-from=${dir}/mangrove/train_images.txt ${dir}/mangrove/images ${dir}/mangrove/train_images
rsync --files-from=${dir}/mangrove/test_images.txt ${dir}/mangrove/images ${dir}/mangrove/test_images
rsync --files-from=${dir}/mangrove/val_images.txt ${dir}/mangrove/images ${dir}/mangrove/val_images
rsync --files-from=${dir}/mangrove/train_masks.txt ${dir}/mangrove/masks ${dir}/mangrove/train_masks
rsync --files-from=${dir}/mangrove/test_masks.txt ${dir}/mangrove/masks ${dir}/mangrove/test_masks
rsync --files-from=${dir}/mangrove/val_masks.txt ${dir}/mangrove/masks ${dir}/mangrove/val_masks

find ${dir}/marsh/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/marsh/shuf_images.txt

ntrain=49; nval=7; ntest=14;

cat ${dir}/marsh/shuf_images.txt | head -n ${ntrain} > ${dir}/marsh/train_images.txt
cat ${dir}/marsh/shuf_images.txt | tail -n ${ntest} > ${dir}/marsh/test_images.txt
cat ${dir}/marsh/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/marsh/val_images.txt
cat ${dir}/marsh/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/marsh/train_masks.txt
cat ${dir}/marsh/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/marsh/test_masks.txt
cat ${dir}/marsh/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/marsh/val_masks.txt

rsync --files-from=${dir}/marsh/train_images.txt ${dir}/marsh/images ${dir}/marsh/train_images
rsync --files-from=${dir}/marsh/test_images.txt ${dir}/marsh/images ${dir}/marsh/test_images
rsync --files-from=${dir}/marsh/val_images.txt ${dir}/marsh/images ${dir}/marsh/val_images
rsync --files-from=${dir}/marsh/train_masks.txt ${dir}/marsh/masks ${dir}/marsh/train_masks
rsync --files-from=${dir}/marsh/test_masks.txt ${dir}/marsh/masks ${dir}/marsh/test_masks
rsync --files-from=${dir}/marsh/val_masks.txt ${dir}/marsh/masks ${dir}/marsh/val_masks

find ${dir}/puddle/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/puddle/shuf_images.txt

ntrain=43; nval=7; ntest=12;

cat ${dir}/puddle/shuf_images.txt | head -n ${ntrain} > ${dir}/puddle/train_images.txt
cat ${dir}/puddle/shuf_images.txt | tail -n ${ntest} > ${dir}/puddle/test_images.txt
cat ${dir}/puddle/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/puddle/val_images.txt
cat ${dir}/puddle/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/puddle/train_masks.txt
cat ${dir}/puddle/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/puddle/test_masks.txt
cat ${dir}/puddle/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/puddle/val_masks.txt

rsync --files-from=${dir}/puddle/train_images.txt ${dir}/puddle/images ${dir}/puddle/train_images
rsync --files-from=${dir}/puddle/test_images.txt ${dir}/puddle/images ${dir}/puddle/test_images
rsync --files-from=${dir}/puddle/val_images.txt ${dir}/puddle/images ${dir}/puddle/val_images
rsync --files-from=${dir}/puddle/train_masks.txt ${dir}/puddle/masks ${dir}/puddle/train_masks
rsync --files-from=${dir}/puddle/test_masks.txt ${dir}/puddle/masks ${dir}/puddle/test_masks
rsync --files-from=${dir}/puddle/val_masks.txt ${dir}/puddle/masks ${dir}/puddle/val_masks

find ${dir}/rapids/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/rapids/shuf_images.txt

ntrain=64; nval=10; ntest=18;

cat ${dir}/rapids/shuf_images.txt | head -n ${ntrain} > ${dir}/rapids/train_images.txt
cat ${dir}/rapids/shuf_images.txt | tail -n ${ntest} > ${dir}/rapids/test_images.txt
cat ${dir}/rapids/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/rapids/val_images.txt
cat ${dir}/rapids/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/rapids/train_masks.txt
cat ${dir}/rapids/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/rapids/test_masks.txt
cat ${dir}/rapids/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/rapids/val_masks.txt

rsync --files-from=${dir}/rapids/train_images.txt ${dir}/rapids/images ${dir}/rapids/train_images
rsync --files-from=${dir}/rapids/test_images.txt ${dir}/rapids/images ${dir}/rapids/test_images
rsync --files-from=${dir}/rapids/val_images.txt ${dir}/rapids/images ${dir}/rapids/val_images
rsync --files-from=${dir}/rapids/train_masks.txt ${dir}/rapids/masks ${dir}/rapids/train_masks
rsync --files-from=${dir}/rapids/test_masks.txt ${dir}/rapids/masks ${dir}/rapids/test_masks
rsync --files-from=${dir}/rapids/val_masks.txt ${dir}/rapids/masks ${dir}/rapids/val_masks

find ${dir}/river/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/river/shuf_images.txt

ntrain=89; nval=13; ntest=25;

cat ${dir}/river/shuf_images.txt | head -n ${ntrain} > ${dir}/river/train_images.txt
cat ${dir}/river/shuf_images.txt | tail -n ${ntest} > ${dir}/river/test_images.txt
cat ${dir}/river/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/river/val_images.txt
cat ${dir}/river/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/river/train_masks.txt
cat ${dir}/river/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/river/test_masks.txt
cat ${dir}/river/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/river/val_masks.txt

rsync --files-from=${dir}/river/train_images.txt ${dir}/river/images ${dir}/river/train_images
rsync --files-from=${dir}/river/test_images.txt ${dir}/river/images ${dir}/river/test_images
rsync --files-from=${dir}/river/val_images.txt ${dir}/river/images ${dir}/river/val_images
rsync --files-from=${dir}/river/train_masks.txt ${dir}/river/masks ${dir}/river/train_masks
rsync --files-from=${dir}/river/test_masks.txt ${dir}/river/masks ${dir}/river/test_masks
rsync --files-from=${dir}/river/val_masks.txt ${dir}/river/masks ${dir}/river/val_masks

find ${dir}/river_delta/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/river_delta/shuf_images.txt

ntrain=16; nval=4; ntest=4;

cat ${dir}/river_delta/shuf_images.txt | head -n ${ntrain} > ${dir}/river_delta/train_images.txt
cat ${dir}/river_delta/shuf_images.txt | tail -n ${ntest} > ${dir}/river_delta/test_images.txt
cat ${dir}/river_delta/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/river_delta/val_images.txt
cat ${dir}/river_delta/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/river_delta/train_masks.txt
cat ${dir}/river_delta/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/river_delta/test_masks.txt
cat ${dir}/river_delta/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/river_delta/val_masks.txt

rsync --files-from=${dir}/river_delta/train_images.txt ${dir}/river_delta/images ${dir}/river_delta/train_images
rsync --files-from=${dir}/river_delta/test_images.txt ${dir}/river_delta/images ${dir}/river_delta/test_images
rsync --files-from=${dir}/river_delta/val_images.txt ${dir}/river_delta/images ${dir}/river_delta/val_images
rsync --files-from=${dir}/river_delta/train_masks.txt ${dir}/river_delta/masks ${dir}/river_delta/train_masks
rsync --files-from=${dir}/river_delta/test_masks.txt ${dir}/river_delta/masks ${dir}/river_delta/test_masks
rsync --files-from=${dir}/river_delta/val_masks.txt ${dir}/river_delta/masks ${dir}/river_delta/val_masks

find ${dir}/sea/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/sea/shuf_images.txt

ntrain=142; nval=22; ntest=40;

cat ${dir}/sea/shuf_images.txt | head -n ${ntrain} > ${dir}/sea/train_images.txt
cat ${dir}/sea/shuf_images.txt | tail -n ${ntest} > ${dir}/sea/test_images.txt
cat ${dir}/sea/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/sea/val_images.txt
cat ${dir}/sea/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/sea/train_masks.txt
cat ${dir}/sea/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/sea/test_masks.txt
cat ${dir}/sea/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/sea/val_masks.txt

rsync --files-from=${dir}/sea/train_images.txt ${dir}/sea/images ${dir}/sea/train_images
rsync --files-from=${dir}/sea/test_images.txt ${dir}/sea/images ${dir}/sea/test_images
rsync --files-from=${dir}/sea/val_images.txt ${dir}/sea/images ${dir}/sea/val_images
rsync --files-from=${dir}/sea/train_masks.txt ${dir}/sea/masks ${dir}/sea/train_masks
rsync --files-from=${dir}/sea/test_masks.txt ${dir}/sea/masks ${dir}/sea/test_masks
rsync --files-from=${dir}/sea/val_masks.txt ${dir}/sea/masks ${dir}/sea/val_masks

find ${dir}/shoreline/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/shoreline/shuf_images.txt

ntrain=48; nval=8; ntest=13;

cat ${dir}/shoreline/shuf_images.txt | head -n ${ntrain} > ${dir}/shoreline/train_images.txt
cat ${dir}/shoreline/shuf_images.txt | tail -n ${ntest} > ${dir}/shoreline/test_images.txt
cat ${dir}/shoreline/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/shoreline/val_images.txt
cat ${dir}/shoreline/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/shoreline/train_masks.txt
cat ${dir}/shoreline/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/shoreline/test_masks.txt
cat ${dir}/shoreline/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/shoreline/val_masks.txt

rsync --files-from=${dir}/shoreline/train_images.txt ${dir}/shoreline/images ${dir}/shoreline/train_images
rsync --files-from=${dir}/shoreline/test_images.txt ${dir}/shoreline/images ${dir}/shoreline/test_images
rsync --files-from=${dir}/shoreline/val_images.txt ${dir}/shoreline/images ${dir}/shoreline/val_images
rsync --files-from=${dir}/shoreline/train_masks.txt ${dir}/shoreline/masks ${dir}/shoreline/train_masks
rsync --files-from=${dir}/shoreline/test_masks.txt ${dir}/shoreline/masks ${dir}/shoreline/test_masks
rsync --files-from=${dir}/shoreline/val_masks.txt ${dir}/shoreline/masks ${dir}/shoreline/val_masks

find ${dir}/snow/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/snow/shuf_images.txt

ntrain=168; nval=24; ntest=48;

cat ${dir}/snow/shuf_images.txt | head -n ${ntrain} > ${dir}/snow/train_images.txt
cat ${dir}/snow/shuf_images.txt | tail -n ${ntest} > ${dir}/snow/test_images.txt
cat ${dir}/snow/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/snow/val_images.txt
cat ${dir}/snow/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/snow/train_masks.txt
cat ${dir}/snow/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/snow/test_masks.txt
cat ${dir}/snow/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/snow/val_masks.txt

rsync --files-from=${dir}/snow/train_images.txt ${dir}/snow/images ${dir}/snow/train_images
rsync --files-from=${dir}/snow/test_images.txt ${dir}/snow/images ${dir}/snow/test_images
rsync --files-from=${dir}/snow/val_images.txt ${dir}/snow/images ${dir}/snow/val_images
rsync --files-from=${dir}/snow/train_masks.txt ${dir}/snow/masks ${dir}/snow/train_masks
rsync --files-from=${dir}/snow/test_masks.txt ${dir}/snow/masks ${dir}/snow/test_masks
rsync --files-from=${dir}/snow/val_masks.txt ${dir}/snow/masks ${dir}/snow/val_masks

find ${dir}/wetland/images/ -type f |cut -d "/" -f 9 | shuf > ${dir}/wetland/shuf_images.txt

ntrain=56; nval=8; ntest=16;

cat ${dir}/wetland/shuf_images.txt | head -n ${ntrain} > ${dir}/wetland/train_images.txt
cat ${dir}/wetland/shuf_images.txt | tail -n ${ntest} > ${dir}/wetland/test_images.txt
cat ${dir}/wetland/shuf_images.txt | head -n -${ntest}| tail -n ${nval} > ${dir}/wetland/val_images.txt
cat ${dir}/wetland/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n ${ntrain} > ${dir}/wetland/train_masks.txt
cat ${dir}/wetland/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | tail -n ${ntest} > ${dir}/wetland/test_masks.txt
cat ${dir}/wetland/shuf_images.txt | cut -d "." -f 1 | sed 's/$/.png/' | head -n -${ntest} | tail -n ${nval} > ${dir}/wetland/val_masks.txt

rsync --files-from=${dir}/wetland/train_images.txt ${dir}/wetland/images ${dir}/wetland/train_images
rsync --files-from=${dir}/wetland/test_images.txt ${dir}/wetland/images ${dir}/wetland/test_images
rsync --files-from=${dir}/wetland/val_images.txt ${dir}/wetland/images ${dir}/wetland/val_images
rsync --files-from=${dir}/wetland/train_masks.txt ${dir}/wetland/masks ${dir}/wetland/train_masks
rsync --files-from=${dir}/wetland/test_masks.txt ${dir}/wetland/masks ${dir}/wetland/test_masks
rsync --files-from=${dir}/wetland/val_masks.txt ${dir}/wetland/masks ${dir}/wetland/val_masks
