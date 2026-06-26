# Time Stan HMC sampling on the 2-D "banana" target.
#
# Stan's default sampler is NUTS (adaptive HMC). For an apples-to-apples
# comparison with the fixed-trajectory anvl / torch samplers we switch to
# algorithm = "HMC", disable adaptation, and match the step size, integration
# time (int_time / stepsize = L leapfrog steps) and identity mass matrix.
#
# Unlike anvl and torch, Stan does not vectorise the chains onto one device: it
# runs `n_chains` independent Markov chains. We pin cores = 1 so the comparison
# is single-threaded; the compiled C++ model is reused across jobs via rstan's
# auto_write cache, and its (one-time) compile cost is reported as compile_time.

time_stan <- function(n_chains, n_samples, n_warmup, L, eps, b, device, seed) {
  suppressMessages(library(rstan))
  source(here::here("benchmarks", "hmc", "diagnostics.R"))
  rstan::rstan_options(auto_write = TRUE)

  model_file <- here::here("benchmarks", "hmc", "banana.stan")

  # Compiling reuses the cached binary (next to the .stan file) when present, so
  # this is only expensive on the very first job.
  t0 <- Sys.time()
  model <- rstan::stan_model(file = model_file)
  compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  int_time <- eps * L

  sample_fit <- function(iter, warmup, chains) {
    suppressWarnings(rstan::sampling(
      model,
      data = list(b = b),
      chains = chains, iter = iter, warmup = warmup,
      cores = 1L,
      algorithm = "HMC",
      control = list(
        adapt_engaged = FALSE, stepsize = eps,
        int_time = int_time, metric = "unit_e"
      ),
      refresh = 0, seed = seed,
      save_warmup = FALSE, pars = "theta", include = TRUE
    ))
  }

  # One warm-up sampling call so the first timed call pays no first-time setup.
  invisible(sample_fit(iter = n_warmup + 10L, warmup = n_warmup, chains = n_chains))

  t0 <- Sys.time()
  fit <- sample_fit(iter = n_warmup + n_samples, warmup = n_warmup, chains = n_chains)
  time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # (n_samples, n_chains, 2), parameter axis last as hmc_diagnostics expects.
  samples <- rstan::extract(fit, pars = "theta", permuted = FALSE)
  dimnames(samples) <- NULL
  diag <- hmc_diagnostics(samples, b)

  n_cores <- length(parallel::mcaffinity())
  list(
    time = time,
    compile_time = compile_time,
    sd1 = diag$sd1,
    sd2 = diag$sd2,
    err = diag$err,
    n_cores = n_cores
  )
}

if (FALSE) {
  time_stan(
    n_chains = 4L,
    n_samples = 500L,
    n_warmup = 200L,
    L = 80L,
    eps = 0.2,
    b = 0.01,
    device = "cpu",
    seed = 42L
  )
}
