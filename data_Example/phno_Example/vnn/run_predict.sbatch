#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name='Predict VNN phno_Example'
#SBATCH --account=usr_acct
#SBATCH --mail-user=usr_email
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module load apptainer
bash ./run_predict.sh
