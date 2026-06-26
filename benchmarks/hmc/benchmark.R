library(batchtools)

setup <- function(reg_path, work_dir, seed = 42L) {
  reg <- makeExperimentRegistry(
    file.dir = reg_path,
    work.dir = work_dir,
    packages = "checkmate",
    seed = seed
  )
  reg$cluster.functions <- makeClusterFunctionsInteractive()

  source(here::here("benchmarks", "hmc", "time_rtorch.R"))
  source(here::here("benchmarks", "hmc", "time_anvl.R"))
  source(here::here("benchmarks", "hmc", "time_stan.R"))

  batchExport(list(
    time_rtorch = time_rtorch,
    time_anvl = time_anvl,
    time_stan = time_stan
  ))

  # Sampling the 2-D banana target with K = n_chains parallel chains.
  addProblem(
    "runtime_sample",
    data = NULL,
    fun = function(
      n_chains,
      n_samples,
      n_warmup,
      L,
      eps,
      b,
      device,
      ...
    ) {
      list(
        n_chains = assert_int(n_chains),
        n_samples = assert_int(n_samples),
        n_warmup = assert_int(n_warmup),
        L = assert_int(L),
        eps = assert_number(eps),
        b = assert_number(b),
        device = assert_choice(device, c("cuda", "cpu", "mps"))
      )
    }
  )

  addAlgorithm("rtorch", fun = function(instance, job, ...) {
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed)))
  })

  addAlgorithm("stan", fun = function(instance, job, ...) {
    callr::r(time_stan, args = c(instance, list(seed = job$seed)))
  })

  addAlgorithm("anvl", fun = function(instance, job, compile_loop, ...) {
    callr::r(time_anvl, args = c(instance, list(seed = job$seed, compile_loop = compile_loop)))
  })
}
