# benchmarks

Benchmarks for the {anvil} package.

## Benchmarks

* mlp: Trains an MLP in PyTorch, R torch and anvil.

## Controlling CPU Threads

Unfortunately, there is no easy way to control the number of CPU threads for XLA within R.
Therefore, we start the R processes with `taskset -c <nthreads> R` and then run the benchmark.
The child processes will inherit the number of threads from the parent process.
