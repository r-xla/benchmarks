#!/bin/bash
#SBATCH --job-name=hmc-cpu-multi
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.out
#SBATCH --error=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks/benchmarks/hmc/slurm-%x-%j.err
#
# Launches benchmarks/hmc/run-cpu-multi.R (multi-threaded; anvl + torch).
# The script labels its registry / result with the number of allocated cores
# (parallel::mcaffinity()), so --cpus-per-task here sets that count.

set -eo pipefail

# Activate the conda env that provides R + torch (matches the interactive setup).
source /dss/dsshome1/lxc0C/ru48nas2/miniforge3/etc/profile.d/conda.sh
conda activate base

# Let BLAS/OpenMP use all allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

PROJECT=/dss/dsshome1/lxc0C/ru48nas2/r-xla/benchmarks
HMC="$PROJECT/benchmarks/hmc"
cd "$PROJECT"

# run-cpu-multi.R deletes an existing registry automatically when run non-interactively.
Rscript "$HMC/run-cpu-multi.R"
