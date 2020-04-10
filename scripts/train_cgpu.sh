#!/bin/bash
#SBATCH -J train-cgpu
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --ntasks-per-node 8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 8:00:00
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
. scripts/setup_cgpu.sh

# Multi-GPU training
srun -u -l python train.py --rank-gpu -d ddp-file $@
