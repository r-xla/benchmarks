# Run HMC benchmark on CPU (multi thread)
# Focuses on the implementations that vectorise the chains onto one device
# (anvl, torch); Stan does not, so it is excluded here.
library(here)

source(here("benchmarks", "hmc", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

REG_PATH <- here("benchmarks", "hmc", paste0("registry-cpu-multi-", length(parallel::mcaffinity())))

if (dir.exists(REG_PATH)) {
  if (interactive()) {
    # Ask whether to delete the registry
    answer <- readline("Registry already exists. Delete it to run the benchmark again? (y/n)")
    if (answer != "y") {
      stop("Registry already exists. Delete it to run the benchmark again.")
    }
  }
  # Interactive "y" or non-interactive (e.g. batch): delete and re-run.
  unlink(REG_PATH, recursive = TRUE)
}

setup(
  REG_PATH,
  here(),
  seed = SEED
)

problem_design <- expand.grid(
  list(
    n_chains = c(1L, 4L, 16L, 64L, 256L, 1024L),
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
    anvl = data.frame(compile_loop = c(TRUE, FALSE))
  ),
  repls = 2L
)

tbl <- unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmarks", "hmc", "summarize.R"))
result <- summarize(tbl$job.id)
saveRDS(result, here("benchmarks", "hmc", paste0("result-cpu-multi-", length(parallel::mcaffinity()), ".rds")))
