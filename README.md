# benchmarks

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

## Running the Benchmark

```bash
Rscript benchmarks/matmul.R
```
