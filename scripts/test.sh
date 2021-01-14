#!/bin/bash

# Quick test script to make sure things are working

module purge
module load esslurm
. scripts/setup_cgpu.sh

set -e

srun -C gpu -G 8 -c 10 --ntasks-per-node 8 --exclusive -t 10 -u -l \
    python train.py --rank-gpu -d ddp-file \
    --n-train 32 --n-valid 32 --n-epochs 2 \
    --n-data-buckets 4 \
    --output-dir $SCRATCH/heptrkx/results/test_agnn \
    configs/agnn.yaml

#srun -C gpu -G 8 -c 10 --ntasks-per-node 8 --exclusive -t 10 -u -l \
#    python train.py --rank-gpu -d ddp-file \
#    --n-train 32 --n-valid 32 --n-epochs 2 \
#    --output-dir $SCRATCH/heptrkx/results/test_mpnn \
#    configs/mpnn.yaml
