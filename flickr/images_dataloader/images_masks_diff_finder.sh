dir=/home/serfani/Documents/Atlantis;
subdirs=$(echo $(ls ${dir}));
for sd in ${subdirs}; do
    subsubdir=$(echo $(ls ${sd}));
    for sd2 in ${subsubdir}; do
        #echo ${sd},${sd2};
        if [[ ${sd2} == "images" ]]; then
           ls ${dir}/${sd}/${sd2} | cut -d "." -f 1 > ${dir}/${sd}/images.txt;
        else 
            ls ${dir}/${sd}/${sd2} | cut -d "." -f 1 > ${dir}/${sd}/segid.txt;  
        fi
    done
    diff -u ${dir}/${sd}/images.txt ${dir}/${sd}/segid.txt > ${dir}/${sd}/diff.txt
done