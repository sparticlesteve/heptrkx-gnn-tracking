#!/bin/bash

# Submit primary training jobs

module purge
module load esslurm

sbatch -J train-agnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/agnn.yaml
sbatch -J train-agnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/agnn.yaml --resume
sbatch -J train-agnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/agnn.yaml --resume

sbatch -J train-mpnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/mpnn.yaml
sbatch -J train-mpnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/mpnn.yaml --resume
sbatch -J train-mpnn-cgpu -t 8:00:00 scripts/train_cgpu.sh configs/mpnn.yaml --resume
