library(anvilbench)
library(batchtools)
library(here)

Sys.setenv(XLA_FLAGS = paste(c(
  "--xla_cpu_multi_thread_eigen=false",
  "intra_op_parallelism_threads=1"),
collapse = " "))

Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(NPROC = "1")

bench_matmul <- function(n_matmuls, matrix_size, device) {
  x <- diag(matrix_size)
  browser()

  x_torch <- torch_tensor(x, device = device)
  x_anvil <- nv_tensor(x, platform = device)

  f_torch <- function(x) {
    for (i in seq_len(n_matmuls)) {
      x <- torch_matmul(x, x)
    }
    x
  }

  f_anvil <- jit(function(x) {
    for (i in seq_len(n_matmuls)) {
      x <- x %*% x
    }
    x
  })

  bench::mark(
    f_torch = f_torch(x_torch),
    f_anvil = f_anvil(x_anvil),
    check = FALSE
  )
}



config <- expand.grid(
  n_matmuls = c(1, 10, 100, 1000),
  matrix_size = c(200, 400, 800, 1600, 3200),
  device = "cpu",
  stringsAsFactors = FALSE
)

if (dir.exists(here("results", "matmul"))) {
    unlink(here("results", "matmul"), recursive = TRUE)
  # ask user if they want to delete the results
  #if (readline("Do you want to delete the results? (y/n)") == "y") {
  #  unlink(here("results", "matmul"), recursive = TRUE)
  #} else {
  #  stop("Results already exist. Please delete them or run the benchmark with a different configuration.")
  #}
}

reg <- makeRegistry(file.dir = here("results", "matmul"), packages = "anvilbench")

batchMap(bench_matmul, n_matmuls = config$n_matmuls, matrix_size = config$matrix_size, device = config$device)
