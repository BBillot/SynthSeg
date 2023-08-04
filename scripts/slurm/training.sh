#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synthseg
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=3GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=01:00:00

module load anaconda/3/2021.11 scikit-learn/1.1.1 tensorflow/gpu-cuda-11.6/2.11.0 keras/2.11.0 tensorboard/2.11.0
source /raven/u/dcfidalgo/venvs/synthseg/bin/activate

export PYTHONPATH=/raven/u/dcfidalgo/projects/cbs/SynthSeg:$PYTHONPATH

# in case WANDB is used set WANDB_BASE_URL and WANDB_API_KEY
export WANDB_BASE_URL="http://10.186.1.194:80"
# export WANDB_API_KEY=...
export WANDB_PROJECT="SynthSeg"
export WANDB_NAME="test"

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64/:${LD_LIBRARY_PATH}

# Avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

## XLA
# cuda aware
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# enable autoclustering for CPU and GPU
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# ---

# TODO: David, why does Python know where `training.py` is when we only set the PYTHONPATH to the top-level
# project directory? IMO, the could _should_ be something like
# srun python scripts/slurm/training.sh --config_file data/cbs/original_training_config/training.json
srun python training.py --dice_epochs=3 --wandb
