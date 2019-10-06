# Software environment setup for Cori GPU
module load gcc/7.3.0
module load cuda/10.1.168
module load openmpi/4.0.1-ucx-1.6
module load pytorch/v1.2.0-gpu

# Libary path fix
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH

# Setup Cray AI HPO
export PYTHONPATH=/global/cscratch1/sd/sfarrell/cray-hpo/software
