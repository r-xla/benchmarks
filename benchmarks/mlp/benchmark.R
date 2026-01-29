library(batchtools)

setup <- function(reg_path, python_path, work_dir, seed = 42L) {
  reg <- makeExperimentRegistry(
    file.dir = reg_path,
    work.dir = work_dir,
    packages = "checkmate",
    seed = seed
  )
  reg$cluster.functions <- makeClusterFunctionsInteractive()

  source(here::here("benchmarks", "mlp", "time_rtorch.R"))
  source(here::here("benchmarks", "mlp", "time_anvil.R"))

  batchExport(list(
    time_rtorch = time_rtorch,
    time_anvil = time_anvil
  ))

  addProblem(
    "runtime_train",
    data = NULL,
    fun = function(
      epochs,
      batch_size,
      n_batches,
      n_layers,
      latent,
      p,
      device,
      ...
    ) {
      problem <- list(
        epochs = assert_int(epochs),
        batch_size = assert_int(batch_size),
        n_batches = assert_int(n_batches),
        n_layers = assert_int(n_layers),
        latent = assert_int(latent),
        p = assert_int(p),
        device = assert_choice(device, c("cuda", "cpu", "mps"))
      )
      problem
    }
  )

  addAlgorithm("pytorch", fun = function(instance, job, ...) {
    f <- function(..., python_path) {
      library(reticulate)
      x <- try({
        reticulate::use_python(python_path, required = TRUE)
        reticulate::source_python(here::here("benchmarks", "mlp", "time_pytorch.py"))
        print(reticulate::py_config())
        time_pytorch(...)
      }, silent = TRUE)
      print(x)
    }
    args <- c(instance, list(seed = job$seed, python_path = python_path))
    callr::r(f, args = args)
  })

  addAlgorithm("rtorch", fun = function(instance, job, ...) {
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed)))
  })

  addAlgorithm("anvil", fun = function(instance, job, compile_loop, ...) {
    callr::r(time_anvil, args = c(instance, list(seed = job$seed, compile_loop = compile_loop)))
  })
}