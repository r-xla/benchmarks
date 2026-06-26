# Time R torch HMC sampling on the 2-D "banana" target.
#
# Mirrors the anvl sampler: the kernel is batched over a leading chain axis of
# size K, so K chains evolve in parallel as a single (K, 2) tensor. The gradient
# of the potential is obtained with torch autograd, and the sampling loop runs
# in R. torch has no ahead-of-time compilation step here, so compile_time is 0.

time_rtorch <- function(n_chains, n_samples, n_warmup, L, eps, b, device, seed) {
  library(torch)
  torch_set_num_interop_threads(1L)
  torch_manual_seed(seed)
  source(here::here("benchmarks", "hmc", "diagnostics.R"))

  dtype <- torch_float64()

  # Potential energy: theta is (K, 2), returns a K-vector of per-chain energies.
  U <- function(theta) {
    theta1 <- theta[, 1]
    theta2 <- theta[, 2]
    theta1^2 / 200 + (theta2 - b * theta1^2 + 100 * b)^2 / 2
  }

  # Gradient of sum(U) wrt theta. As in the anvl version, the per-chain energies
  # are independent so row k of the gradient is exactly grad U at theta_k.
  grad_U <- function(theta) {
    theta <- theta$detach()$requires_grad_(TRUE)
    u <- U(theta)$sum()
    autograd_grad(u, theta)[[1L]]$detach()
  }

  leapfrog <- function(theta, p) {
    grad <- grad_U(theta)
    for (i in seq_len(L)) {
      p <- p - 0.5 * eps * grad
      theta <- theta + eps * p
      grad <- grad_U(theta)
      p <- p - 0.5 * eps * grad
    }
    list(theta = theta, p = p)
  }

  # A single HMC transition for all K chains at once.
  hmc_step <- function(theta) {
    p <- torch_randn(n_chains, 2L, dtype = dtype, device = device)
    H_current <- U(theta) + p$pow(2)$sum(dim = 2L) / 2
    lf <- leapfrog(theta, p)
    H_proposed <- U(lf$theta) + lf$p$pow(2)$sum(dim = 2L) / 2
    log_accept <- H_current - H_proposed

    u <- torch_rand(n_chains, dtype = dtype, device = device)
    accept <- (u$log() < log_accept)$unsqueeze(2L)
    torch_where(accept, lf$theta, theta)
  }

  # Draw n samples, writing each into a pre-allocated (n, K, 2) device buffer so
  # the timed region performs no host transfers.
  run <- function(theta, n) {
    samples <- torch_empty(n, n_chains, 2L, dtype = dtype, device = device)
    for (i in seq_len(n)) {
      theta <- hmc_step(theta)
      samples[i, , ] <- theta
    }
    list(theta = theta, samples = samples)
  }

  warmup <- function(theta, n) {
    for (i in seq_len(n)) {
      theta <- hmc_step(theta)
    }
    theta
  }

  theta0 <- torch_zeros(n_chains, 2L, dtype = dtype, device = device)

  theta_warm <- warmup(theta0, n_warmup)
  if (device == "cuda") cuda_synchronize()

  t0 <- Sys.time()
  out <- run(theta_warm, n_samples)
  if (device == "cuda") cuda_synchronize()
  time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  samples <- as.array(out$samples$to(device = "cpu")) # (n_samples, K, 2)
  diag <- hmc_diagnostics(samples, b)

  n_cores <- length(parallel::mcaffinity())
  list(
    time = time,
    compile_time = 0,
    sd1 = diag$sd1,
    sd2 = diag$sd2,
    err = diag$err,
    n_cores = n_cores
  )
}

if (FALSE) {
  time_rtorch(
    n_chains = 16L,
    n_samples = 500L,
    n_warmup = 200L,
    L = 80L,
    eps = 0.2,
    b = 0.01,
    device = "cpu",
    seed = 42L
  )
}
