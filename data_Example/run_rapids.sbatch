#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name='Clustering with Rapids'
#SBATCH --account=usr_acct
#SBATCH --mail-user=usr_email
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apptainer
singularity exec --nv ../containers/rapids.sif python rapidscluster.py --hmp_path ./shared_data/Example.hmp.txt --max_snps 10000
