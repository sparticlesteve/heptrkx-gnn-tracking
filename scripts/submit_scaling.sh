#!/bin/bash

# Cori CPU scaling

sbatch -N 1 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 32 --n-valid 32 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n1

sbatch -N 2 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 64 --n-valid 64 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n2

sbatch -N 4 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 128 --n-valid 128 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n4

sbatch -N 8 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 256 --n-valid 256 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n8

sbatch -N 16 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 512 --n-valid 512 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n16

sbatch -N 32 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 1024 --n-valid 1024 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n32

sbatch -N 64 -J scaling-cori -d singleton scripts/train_cori.sh configs/agnn.yaml \
    --n-train 2048 --n-valid 2048 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n64

sbatch -N 128 -J scaling-cori -d singleton -q regular scripts/train_cori.sh configs/agnn.yaml \
    --n-train 4096 --n-valid 4096 --n-epochs 4 \
    --n-data-buckets 32 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_agnn_n128

# Cori GPU scaling

module purge
module load esslurm

sbatch -n 1 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 256 --n-valid 256 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n1

sbatch -n 2 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 512 --n-valid 512 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n2

sbatch -n 4 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 1024 --n-valid 1024 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n4

sbatch -n 8 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 2048 --n-valid 2048 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n8

sbatch -n 16 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 4096 --n-valid 4096 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n16

sbatch -n 32 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 8192 --n-valid 8192 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n32

sbatch -n 64 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 16384 --n-valid 16384 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n64

sbatch -n 128 -J scaling-cgpu -d singleton -t 30 scripts/train_cgpu.sh configs/agnn.yaml \
    --n-train 32768 --n-valid 32768 --n-epochs 4 \
    --n-data-buckets 64 \
    --output-dir $SCRATCH/heptrkx/results_balanced/scaling_cgpu_agnn_n128
