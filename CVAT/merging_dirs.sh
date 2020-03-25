dirs=$(echo $(ls /home/serfani/goharian/projects/microsoft/Images/images_pn1/artificial));
for d in ${dirs}; do
    subdirs=$(echo $(ls /home/serfani/goharian/projects/microsoft/Images/images_pn1/artificial/${d}));
    mkdir -p /home/serfani/goharian/projects/microsoft/Images/images_pn1/artificial/${d}/new_${d};
    for sd in ${subdirs}; do
        files=$(echo $(ls /home/serfani/goharian/projects/microsoft/Images/images_pn1/artificial/${d}/${sd}));
        for file in ${files}; do
            cp ${d}/${sd}/${file} ${d}/new_${d}/
        done
    done
done

