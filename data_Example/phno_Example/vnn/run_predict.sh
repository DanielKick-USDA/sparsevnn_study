phno='phno_Example'
species='ExampleKEGG'
shared_data='../../shared_data/'
gff_path="$shared_data"'Example.gff'
hmp_path="$shared_data"'Example.hmp.txt'
phno_path="$shared_data""$phno"'.csv'
kmeans_path="$shared_data"'KMeans_taxa_holdouts.json'

sparsevnn_path='../../../containers/sparsevnn.sif'

for i in {0..4};
do
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
        # echo 'Prediction'
        # date
    singularity exec --nv $sparsevnn_path python vnn.py \
        --run_mode predict \
        --model_path './models/vnn/version_'$i'/version_'$i'.pt' \
        --dataloader_shuffle_train False \
        --dataloader_shuffle_valid False

        # echo ''
        # echo 'Evaluation'
        # date
    singularity exec --nv $sparsevnn_path python vnn.py \
        --run_mode 'eval' \
        --model_path './models/vnn/version_'$i'/version_'$i'.pt' \
        --dataloader_shuffle_train False \
        --dataloader_shuffle_valid False \
        --eval saliency_inp saliency_wab rho_out
        # echo 'Finished fold '$i
        # echo ''
done
