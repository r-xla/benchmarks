# Run HMC benchmark on GPU (CUDA)
# Stan has no GPU sampler, so only the device-vectorised implementations
# (anvl, torch) are compared here, over a larger range of parallel chains.
library(here)

source(here("benchmarks", "hmc", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

REG_PATH <- here("benchmarks", "hmc", "registry-gpu")

if (dir.exists(REG_PATH)) {
  # Ask whether to delete the registry
  answer <- readline("Registry already exists. Delete it to run the benchmark again? (y/n)")
  if (answer == "y") {
    unlink(REG_PATH, recursive = TRUE)
  } else {
    stop("Registry already exists. Delete it to run the benchmark again.")
  }
}

setup(
  REG_PATH,
  here(),
  seed = SEED
)

problem_design <- expand.grid(
  list(
    n_chains = c(1L, 16L, 64L, 256L, 1024L, 4096L, 16384L),
    n_samples = 1000L,
    n_warmup = 500L,
    L = 80L,
    eps = 0.2,
    b = 0.01,
    device = "cuda"
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_sample = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(),
    anvl = data.frame(compile_loop = c(TRUE, FALSE))
  ),
  repls = 1L
)

tbl <- unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmarks", "hmc", "summarize.R"))
result <- summarize(tbl$job.id)
saveRDS(result, here("benchmarks", "hmc", "result-gpu.rds"))
