import os, re
# find a file with each hash
o  = [e for e in os.listdir('../dnn/vnn_cache/') if re.match('.+_acgt_shape.parquet', e)]
# get the ctimes for each. The setup run will create a file we can ignore. 
oc = [os.path.getctime(f'../dnn/vnn_cache/{e}') for e in o]
# ignore the first file.
o  = [o[i] for i in range(len(o)) if oc[i] > min(oc)]
# get only hashes
o = [e.split('_')[0] for e in o]
# convert into commands
o = [' '.join([
'singularity exec ../../../containers/bWGR.sif Rscript bWGR_cli.R',
f"--hash {e}",
'--inp_dir ../dnn/vnn_cache',
'--out_dir ./models/blup/',
'--max_inps 1000'
]) for e in o]

with open('./tmp_bWGR.sh', 'w') as f:
     f.writelines('\n'.join(o))
