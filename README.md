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

There are two files with CPU benchmarks:
* `run-cpu-single.R`, which is intended to be run with a single core.
  It compares PyTorch vs rTorch vs anvil (compiled step vs compiled loop)
* `run-cpu-multi.R`, which is intended to be run with multiple threads.
  It compares anvil with compiled loop and compiled step.

Run as follows:

```bash
taskset -c 0 Rscript benchmarks/mlp/run-cpu-single.R
taskset -c 0-31 Rscript benchmarks/mlp/run-cpu-multi.R
```

When running PyTorch with multiple threads, the threads need to be carefully configured
(`MKL_THREADS` vs. `OPENMP_THREADS`), otherwise the comparison is not fair.
For anvil we don't have to take care of this, as XLA seems to be doing fine without any configuration.
