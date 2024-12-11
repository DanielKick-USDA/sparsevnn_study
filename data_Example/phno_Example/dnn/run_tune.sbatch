#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name='Tune DNN phno_Example'
#SBATCH --account=usr_acct
#SBATCH --mail-user=usr_email
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module load apptainer
bash ./run_tune.sh
