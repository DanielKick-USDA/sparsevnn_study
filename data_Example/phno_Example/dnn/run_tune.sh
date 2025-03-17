phno='phno_Example'
species='ExampleKEGG'
shared_data='../../shared_data/'
gff_path="$shared_data"'Example.gff'
hmp_path="$shared_data"'Example.hmp.txt'
phno_path="$shared_data""$phno"'.csv'
kmeans_path="$shared_data"'KMeans_taxa_holdouts.json'

tune_trials=4
tune_max=4
tune_max_epoch=4

sparsevnn_path='../../../containers/sparsevnn.sif'

echo 'start at'
date

# results in params_data.json  params.json  params_list.json  params_run.json
singularity exec --nv $sparsevnn_path python dnn.py \
        --run_mode setup                                              \
        --species "$species"                                          \
        --gff_path "$gff_path"                                        \
        --hmp_path "$hmp_path"                                        \
        --phno_path "$phno_path"                                      \
        --graph_cache_path "$shared_data"

# edit in the holdouts
# Set holdout taxa ignore to the top 20% of smallest clusters
singularity exec --nv $sparsevnn_path python editholdouts.py \
    --inp_fp $kmeans_path \
    --out_fp ./params_data.json \
    --action clear \
    --attribute holdout_taxa_ignore

singularity exec --nv $sparsevnn_path python editholdouts.py \
    --inp_fp $kmeans_path \
    --out_fp ./params_data.json \
    --action add \
    --attribute holdout_taxa_ignore

singularity exec --nv $sparsevnn_path python dnn.py \
    --run_mode tune         \
    --tune_trials $tune_trials \
    --tune_max $tune_max    \
    --max_epoch $tune_max_epoch  

echo 'end at'
date
