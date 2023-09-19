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
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=00:20:00


output_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/data/reproduce
config_file=${output_dir}/config.yml

module load anaconda/3/2021.11 scikit-learn/1.1.1 tensorflow/gpu-cuda-11.6/2.11.0 keras/2.11.0 tensorboard/2.11.0
source /raven/u/dcfidalgo/venvs/synthseg/bin/activate

export PYTHONPATH=/raven/u/dcfidalgo/projects/cbs/segmentation/SynthSeg:$PYTHONPATH


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
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 2 > nvidia_smi_monitoring.csv &
NVIDIASMI_PID=$!

srun python generation.py \
--output_dir=${output_dir} \
--config_file=${config_file} \
--tfrecord \
--count=50 \
--start_int=$(($SLURM_ARRAY_TASK_ID*50)) \
2>&1 | tee -a generation${SLURM_ARRAY_TASK_ID}.log

kill $NVIDIASMI_PID
