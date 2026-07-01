#!/bin/bash
#SBATCH --job-name=hmc-gpu
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.out
#SBATCH --error=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.err
#
# Runs benchmarks/hmc/run-gpu.R inside the anvl CUDA benchmark enroot image.
# Uses the pyxis SPANK plugin (srun --container-*) instead of a manual
# `enroot create` / `enroot start`: pyxis mounts the .sqsh directly, so there
# is no separate import/create step. run-gpu.R deletes an existing registry
# automatically when run non-interactively.

set -eo pipefail

IMAGE=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/r-xla+anvl-cuda-bench+latest.sqsh
RXLA=/dss/dsshome1/lxc0C/ru48nas2/r-xla
PROJECT="$RXLA/benchmarks"

# The container provides R + torch (CUDA) + anvl. We bind-mount the project at
# the same path inside the container so here() resolves and the registry /
# result-gpu.rds are written back to the host.
srun \
  --container-image="$IMAGE" \
  --container-mounts="$RXLA:$RXLA" \
  --container-workdir="$PROJECT" \
  Rscript benchmarks/hmc/run-gpu.R
