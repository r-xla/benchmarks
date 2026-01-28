# Run MLP benchmark on GPU (CUDA)
library(here)

source(here("benchmarks", "mlp", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

# Change this when not running in Docker
PYTHON_PATH <- "/usr/bin/python3"

REG_PATH <- here("benchmarks", "mlp", "registry-gpu")

if (dir.exists(REG_PATH)) {
  stop("Registry already exists. Delete it to run the benchmark again.")
}

setup(
  REG_PATH,
  PYTHON_PATH,
  here(),
  seed = SEED
)

problem_design <- expand.grid(
  list(
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    n_batches = N_BATCHES,
    p = P,
    device = "cuda",
    n_layers = c(0L, 4L, 8L, 16L),
    latent = c(1000L, 3000L, 9000L)
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(),
    anvil = data.frame(compile_loop = c(TRUE, FALSE)),
    pytorch = data.frame()
  ),
  repls = REPLS
)

tbl <- unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmarks", "mlp", "summarize.R"))
result <- summarize(tbl$job.id)
saveRDS(result, here("benchmarks", "mlp", "result-gpu.rds"))
