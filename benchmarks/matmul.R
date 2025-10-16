library(batchtools)
library(here)
library(data.table)

n_matmuls <- 40
matrix_size <- 200
device <- "cpu"

bench_matmul <- function(n_matmuls, matrix_size, device) {
  x <- matrix(runif(matrix_size^2), nrow = matrix_size, ncol = matrix_size)

  x_torch <- torch_tensor(x, device = device)
  x_anvil <- nv_tensor(x, platform = device)

  torch::torch_set_num_threads(1)

  Sys.setenv(XLA_FLAGS = paste(c(
    "--xla_cpu_multi_thread_eigen=false",
    "intra_op_parallelism_threads=1",
    "inter_op_parallelism_threads=1"),
    collapse = " "))

  Sys.setenv(NPROC = "1")


  f_torch <- function(x) {
    for (i in seq_len(n_matmuls)) {
      x <- torch_matmul(x, x)
    }
    c(torch_mean(x)$item())
  }

  f_anvil <- jit(function(x) {
    y <- x
    for (i in seq_len(n_matmuls)) {
      x <- nv_matmul(x, x)
    }
    nv_reduce_sum(x, dim = 1:2)
  })
  
  f_anvil(x_anvil)

  bench::mark(
    f_torch = torch::as_array(f_torch(x_torch)),
    f_anvil = as_array(f_anvil(x_anvil)),
    check = FALSE,
    memory = FALSE
  )
}

config <- expand.grid(
  n_matmuls = c(10, 20, 40, 80, 160, 320, 640, 1280, 2560),
  matrix_size = c(100, 200, 400, 800, 1600),
  matrix_size = 100,
  device = "cpu",
  stringsAsFactors = FALSE
)

if (dir.exists(here("registries", "matmul"))) {
  unlink(here("registries", "matmul"), recursive = TRUE)
}

reg <- makeRegistry(file.dir = here("registries", "matmul"), packages = c("torch", "anvil"))

batchMap(bench_matmul, n_matmuls = config$n_matmuls, matrix_size = config$matrix_size, device = config$device)

submitJobs()

results <- reduceResultsList()

job_pars <- rbindlist(getJobTable()$job.pars)

times_min <- rbindlist(lapply(results, \(x) list(torch = x$min[1L], anvil = x$min[2L])))

tbl <- cbind(job_pars, times_min)

# wide to long, so there is one column "framework" that is torch or anvil
tbl_long <- melt(tbl, id.vars = c("n_matmuls", "matrix_size", "device"), variable.name = "framework", value.name = "time")

saveRDS(tbl_long, here("results", "matmul.rds"))
