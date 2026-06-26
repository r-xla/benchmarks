# Run HMC benchmark on CPU (single thread)
library(here)

source(here("benchmarks", "hmc", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

REG_PATH <- here("benchmarks", "hmc", paste0("registry-cpu-single-", length(parallel::mcaffinity())))

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
    n_chains = c(1L, 2L, 4L, 8L, 16L, 32L, 64L),
    n_samples = 1000L,
    n_warmup = 500L,
    L = 80L,
    eps = 0.2,
    b = 0.01,
    device = "cpu"
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_sample = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(),
    stan = data.frame(),
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
saveRDS(result, here("benchmarks", "hmc", paste0("result-cpu-single-", length(parallel::mcaffinity()), ".rds")))
