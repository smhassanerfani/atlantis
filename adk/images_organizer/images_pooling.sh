dir="/home/serfani/Downloads/atlantis/s1a"
sdirs=$(echo $(ls /home/serfani/Downloads/atlantis/s1a));

for sdir in ${sdirs}; do
    licenses=$(echo $(ls ${dir}/${sdir}));
    mkdir -p ${dir}/${sdir}/images_pool;
    for license in ${licenses}; do
        files=$(echo $(ls ${dir}/${sdir}/{license}));
        for file in ${files}; do
            cp ${dir}/${sdir}/{license}/${file} ${dir}/${sdir}/images_pool
        done
    done
    rm -rf ${dir}/${sdir}/images_pool/*.json
done