#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name='Clustering with Rapids'
#SBATCH --account=rover_pgru_wash
#SBATCH --mail-user=Daniel.Kick@usda.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48

module load apptainer
singularity exec --nv ../containers/rapids.sif python rapidscluster.py --hmp_path ./shared_data/SoySNP50K_genotyping_Gillman_GWAS.hmp.txt --max_snps 10000
