# Run MLP benchmark on CPU
library(here)

source(here("benchmarks", "mlp", "benchmark.R"))

# Configuration
SEED <- 42L
set.seed(SEED)

# Change this when not running in Docker
PYTHON_PATH <- Sys.which("python3")
if (PYTHON_PATH == "") {
  PYTHON_PATH <- Sys.which("python")
}

REG_PATH <- here("benchmarks", "mlp", "registry-cpu")

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
    device = "cpu",
    n_layers = c(0L, 4L),
    latent = c(10, 20, 40, 80)
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(jit = FALSE),
    #pytorch = data.frame(jit = FALSE),
    anvil = data.frame(placeholder = TRUE)
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
saveRDS(result, here("benchmarks", "mlp", "result-cpu.rds"))
