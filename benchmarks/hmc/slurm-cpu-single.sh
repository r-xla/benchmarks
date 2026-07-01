#!/bin/bash
#SBATCH --job-name=hmc-cpu-single
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.out
#SBATCH --error=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.err
#
# Launches benchmarks/hmc/run-cpu-single.R (single-threaded; anvl + torch + stan).

set -eo pipefail

# Activate the conda env that provides R + torch (matches the interactive setup).
source /dss/dsshome1/lxc0C/ru48nas2/miniforge3/etc/profile.d/conda.sh
conda activate base

# Keep the run single-threaded.
export OMP_NUM_THREADS=1

PROJECT=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks
HMC="$PROJECT/benchmarks/hmc"
cd "$PROJECT"

# run-cpu-single.R deletes an existing registry automatically when run non-interactively.
Rscript "$HMC/run-cpu-single.R"
