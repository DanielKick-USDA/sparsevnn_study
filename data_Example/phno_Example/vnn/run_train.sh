phno='phno_Example'
species='ExampleKEGG'
shared_data='../../shared_data/'
gff_path="$shared_data"'Example.gff'
hmp_path="$shared_data"'Example.hmp.txt'
phno_path="$shared_data""$phno"'.csv'
kmeans_path="$shared_data"'KMeans_taxa_holdouts.json'

train_max_epoch=64

sparsevnn_path='../../../containers/sparsevnn.sif'

echo 'start at'
date

singularity exec --nv $sparsevnn_path python editholdouts.py \
    --inp_fp $kmeans_path \
    --out_fp ./params_data.json \
    --action clear \
    --attribute holdout_taxa_ignore

for i in {0..4};
do
        # echo '-------------------------------------------------------------------------------'
    echo 'Beginning fold '$i
    date
        # echo '-------------------------------------------------------------------------------'
        # edit settings files to remove previous fold
    singularity exec --nv $sparsevnn_path python editholdouts.py \
        --inp_fp $kmeans_path \
        --out_fp ./params_data.json \
        --action clear \
        --attribute holdout_taxa

    singularity exec --nv $sparsevnn_path python editholdouts.py \
        --inp_fp $kmeans_path \
        --out_fp ./params_data.json \
        --action add \
        --attribute holdout_taxa \
        --fold $i

        # echo ''
        # echo 'Training'
        # date
    singularity exec --nv $sparsevnn_path python vnn.py \
        --run_mode train \
        --train_from_ax True \
        --max_epoch $train_max_epoch
done

echo 'done at'
date
