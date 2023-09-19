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
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=00:15:00

cfg_file=/ptmp/dcfidalgo/projects/cbs/segmentation/experiments/reproduce/train.yml

module load anaconda/3/2021.11 scikit-learn/1.1.1 tensorflow/gpu-cuda-11.6/2.11.0 keras/2.11.0 tensorboard/2.11.0
source /raven/u/dcfidalgo/venvs/synthseg/bin/activate

export PYTHONPATH=/raven/u/dcfidalgo/projects/cbs/segmentation/SynthSeg:$PYTHONPATH

# in case WANDB is used set WANDB_BASE_URL and WANDB_API_KEY
export WANDB_BASE_URL="http://10.186.1.194:80"
export WANDB_API_KEY="local-3277de3bee22b1c6c3e074cfc1d73b7b169f5d41"
export WANDB_PROJECT="synthseg_test"
export WANDB_NAME="sunet_4gpu_bs16_pr2"

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

## XLA
# cuda aware
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# enable autoclustering for CPU and GPU
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 2 > nvidia_smi_monitoring.csv &
NVIDIASMI_PID=$!

srun python training.py --cfg_file=$cfg_file

kill $NVIDIASMI_PID
