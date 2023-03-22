#!/bin/bash

# This is to prevent conda/mamba to run into situations
# where it can't solve the environment because TF has a weird
# way of specifying it's CUDA dependency.
export CONDA_OVERRIDE_CUDA="11.8"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CONDA_INIT="${HOME}/mambaforge/etc/profile.d/conda.sh"
MAMBA_INIT="${HOME}/mambaforge/etc/profile.d/mamba.sh"

# First check if the environment.yml file does exist in the script directory
ENV_FILE="${SCRIPT_DIR}/environment.yml"
if [ ! -f "${ENV_FILE}" ]; then
  echo "Could not find environment.yml file. This isn't going to work out between us my friend."
  exit 1
fi

# Extract the name of the environment
ENV_NAME=$(sed -n '1s/name: //p' "${ENV_FILE}")

if [ -f "$MAMBA_INIT" ]; then
  echo "Mamba seems to be already installed."
else
  echo "Couldn't find mamba. Installing it now..."
  wget -O Mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
  bash Mambaforge.sh -b -p "${HOME}/mambaforge"
fi

source "${CONDA_INIT}"
source "${MAMBA_INIT}"

if mamba env list | grep -q "^${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Running an update according to ${ENV_FILE}"
  mamba env update -f "${SCRIPT_DIR}/environment.yml"
else
  echo "Environment ${ENV_NAME} does not exist. Creating it according to ${ENV_FILE}"
  mamba env create -f "${SCRIPT_DIR}/environment.yml"
fi
