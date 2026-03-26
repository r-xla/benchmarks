# Run GP benchmark on GPU (CUDA)
library(here)

source(here("benchmarks", "gp", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

# Change this when not running in Docker
PYTHON_PATH <- Sys.which("python3")
if (PYTHON_PATH == "") {
  PYTHON_PATH <- Sys.which("python")
}

REG_PATH <- here("benchmarks", "gp", "registry-gpu")

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
  PYTHON_PATH,
  here(),
  seed = SEED
)

problem_design <- expand.grid(
  list(
    n = c(256L, 512L, 1024L, 2048L, 4096L),
    p = c(1L, 5L, 10L, 20L, 50L),
    n_steps = 100L,
    learning_rate = 0.01,
    device = "cuda"
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    gpytorch = data.frame(),
    anvil = data.frame(compile_loop = c(TRUE, FALSE))
  ),
  repls = 1L
)

tbl <- unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmarks", "gp", "summarize.R"))
result <- summarize(tbl$job.id)
saveRDS(result, here("benchmarks", "gp", "result-gpu.rds"))
