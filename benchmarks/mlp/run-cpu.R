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
    epochs = 10L,
    # batch_size must divide 1024 (n)
    batch_size = c(32L, 64L, 128L),
    #batch_size = 32L,
    n = 256 * 4L,
    p = 10L,
    device = "cpu",
    #n_layers = 0L,
    #latent = 10L
    n_layers = c(0L, 4L, 8L),
    latent = c(10L, 20L, 40L, 80L, 160L)
  ),
  stringsAsFactors = FALSE
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(),
    pytorch = data.frame(),
    anvil = data.frame(compile_loop = c(TRUE, FALSE))
  ),
  repls = 1L
)

tbl <- unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmarks", "mlp", "summarize.R"))
result <- summarize(tbl$job.id)
saveRDS(result, here("benchmarks", "mlp", "result-cpu.rds"))
