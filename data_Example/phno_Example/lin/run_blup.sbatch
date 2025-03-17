#!/bin/bash
#SBATCH --partition=atlas

#SBATCH --job-name='Fit blup phno_Example'
#SBATCH --account=usr_acct
#SBATCH --mail-user=usr_email
#SBATCH --mail-type=END,FAIL
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module load python
module load apptainer
bash ./run_blup.sh
