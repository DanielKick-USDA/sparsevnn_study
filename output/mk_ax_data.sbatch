#!/bin/bash
#SBATCH --partition=atlas

#SBATCH --job-name='Collecting ax records'
#SBATCH --account=rover_pgru_wash
#SBATCH --mail-user=Daniel.Kick@usda.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module load apptainer
apptainer exec ../containers/sparsevnn.sif python ./mk_ax_data.py
