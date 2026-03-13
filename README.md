# benchmarks

Benchmarks for the {anvil} package.

## Benchmarks

* mlp: Trains an MLP in PyTorch, R torch and anvil.

## Controlling CPU Threads

Unfortunately, there is no easy way to control the number of CPU threads for XLA within R.
Therefore, we start the R processes with `taskset -c 0-{nthreads - 1} R` and then run the benchmark.
The child processes will inherit the number of threads from the parent process.

## Environments

For benchmarking, we use the `anvil-cpu-bench` and `anvil-cuda-bench` images as defined in https://github.com/r-xla/docker.

### CPU Benchmark

Start the Docker container (mounting the benchmarks repo):

```bash
docker run -it --rm -v $(pwd):/benchmarks -w /benchmarks sebffischer/anvil-cpu-bench
```

Then, inside the container, run the CPU benchmark with a specific number of threads (e.g. 1):

```bash
taskset -c 0 Rscript benchmarks/mlp/run-cpu.R
```

To use e.g. 4 threads:

```bash
taskset -c 0-3 Rscript benchmarks/mlp/run-cpu.R
```

### CUDA Benchmark

Start the Docker container with GPU access (mounting the benchmarks repo):

```bash
docker run -it --rm --gpus all -v $(pwd):/benchmarks -w /benchmarks sebffischer/anvil-cuda-bench
```

Then, inside the container, run the GPU benchmark:

```bash
Rscript benchmarks/mlp/run-gpu.R
```
