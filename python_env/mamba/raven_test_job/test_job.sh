#!/bin/bash -l
#SBATCH -J testjob
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#
#SBATCH --time=12:00:00
module purge
module load cuda/11.6

source "${HOME}/mambaforge/etc/profile.d/conda.sh"
source "${HOME}/mambaforge/etc/profile.d/mamba.sh"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"

mamba activate synth_seg_py39


# Run the program:
srun python tf_example.py &> testjob_run.log
