library(batchtools)

setup <- function(reg_path, python_path, work_dir, seed = 42L) {
  reg <- makeExperimentRegistry(
    file.dir = reg_path,
    work.dir = work_dir,
    packages = "checkmate",
    seed = seed
  )
  reg$cluster.functions <- makeClusterFunctionsInteractive()

  source(here::here("benchmarks", "gp", "time_anvil.R"))

  batchExport(list(
    time_anvil = time_anvil
  ))

  addProblem(
    "runtime_train",
    data = NULL,
    fun = function(
      n,
      p,
      n_steps,
      learning_rate,
      device,
      ...
    ) {
      problem <- list(
        n = assert_int(n),
        p = assert_int(p),
        n_steps = assert_int(n_steps),
        learning_rate = assert_number(learning_rate),
        device = assert_choice(device, c("cuda", "cpu", "mps"))
      )
      problem
    }
  )

  addAlgorithm("gpytorch", fun = function(instance, job, ...) {
    f <- function(..., python_path) {
      library(reticulate)
      x <- try({
        reticulate::use_python(python_path, required = TRUE)
        reticulate::source_python(here::here("benchmarks", "gp", "time_gpytorch.py"))
        print(reticulate::py_config())
        time_gpytorch(...)
      }, silent = TRUE)
      print(x)
    }
    args <- c(instance, list(seed = job$seed, python_path = python_path))
    callr::r(f, args = args)
  })

  addAlgorithm("anvil", fun = function(instance, job, compile_loop, ...) {
    callr::r(time_anvil, args = c(instance, list(seed = job$seed, compile_loop = compile_loop)))
  })
}
