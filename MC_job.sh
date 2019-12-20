#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=gpujob

module load CUDA

cd $SLURM_SUBMIT_DIR
python MC_train.py
