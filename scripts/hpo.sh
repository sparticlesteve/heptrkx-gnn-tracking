#!/bin/bash
#SBATCH -J hpo
#SBATCH -C gpu -N 1 -c 10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out

. scripts/setup_cgpu.sh

python -u hpo.py $@
