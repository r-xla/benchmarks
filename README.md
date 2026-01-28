# benchmarks

Benchmarks comparing R torch, anvil, and PyTorch.

## Benchmarks

### Matrix Multiplication (`benchmarks/matmul.R`)

Compares matrix multiplication performance between R torch and anvil.

### MLP Training (`benchmarks/mlp/`)

Compares MLP training performance between R torch, anvil, and Python PyTorch.

**Files:**
- `benchmark.R` - Setup function and experiment definitions (using batchtools)
- `time_rtorch.R` - R torch timing function
- `time_anvil.R` - anvil timing function
- `time_pytorch.py` - Python PyTorch timing function
- `run-cpu.R` - Run benchmark on CPU
- `run-gpu.R` - Run benchmark on GPU (CUDA)
- `summarize.R` - Summarize results

## Building the Docker Image

For CPU:

```bash
make env-cpu
```

## Downloading the Image

```bash
docker pull sebffischer/anvil-bench-cpu:latest
```

## Running the Image

Assuming `benchmarks` is in your home directory, you can start the container as follows:

```bash
docker run -it --rm -v ~/benchmarks:/mnt/data/benchmarks sebffischer/anvil-bench-cpu:latest
```

## Running the Benchmarks

### Matrix Multiplication

```bash
Rscript benchmarks/matmul.R
```

### MLP

```bash
Rscript benchmarks/mlp/run-cpu.R
```
