# Python environment using Mamba

Mamba is basically conda, only faster, especially during the resolution of dependencies.
Most conda commands can also be used in Mamba, but Mamba has additional nice features like
`mamba repoquery search`.

To install Mamba, create the environment or update the environment you can use

```bash
bash bootstrap_mamba.sh
```

It will use the `environment.yml` file as the definition for the Python environment and do the following things

1. Check if Mamba is installed in `$HOME/mambaforge`. If not, it will download and install it.
2. Check if Mamba has already an environment with the same name as specified in `environment.yml`
   - If yes, it will call `mamba env update -f "environment.yml"` which checks if any of the package versions in the file have changed and update the environment accordingly.
   - If no, it will call `mamba env create -f "environment.yml"` to create a new environment.

## Test job for Raven

The `raven_test_job` directory contains a small example that uses tensorflow to train a network. It can be used
to check if the Python environment indeed works and makes use of the GPU.
You can submit this job on the Raven cluster using

```bash
sbatch test_job.sh
```

and check its status using `squery -u yourLogin`.
Once it is finished you will find the log-files of the run inside the directory.

Note that this test-job currently assumes that the mamba environment is called `synth_seg_py39` and that you
indeed have installed mamba in `$HOME/mambaforge`.
Additionally, the line

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"
```

inside `test_job.sh` ensures that tensorflow can find CUDA that was loaded with `module load cuda/11.6`.

Bootstrapping Mamba, creating a fresh environment, and running the job doesn't take more than 10min.