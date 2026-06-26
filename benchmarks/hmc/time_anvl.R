# Time anvl HMC sampling on the 2-D "banana" target.
#
# The sampler is written in batched form: every array carries a leading chain
# axis of size K, and the K chains evolve independently. Running many chains in
# parallel needs no extra code, only a (K, 2) starting tensor.
#
# Supports two modes:
# - compile_loop = TRUE:  the whole sampling loop is lifted into a single
#   JIT-compiled graph (hmc_sample_many), so one executable call produces all
#   n_samples for all K chains with no per-iteration R <-> XLA dispatch.
# - compile_loop = FALSE: only a single HMC transition is JIT-compiled
#   (hmc_step), and the sampling loop runs in R.

time_anvl <- function(n_chains, n_samples, n_warmup, L, eps, b, device, seed, compile_loop = TRUE) {
  library(anvl)
  source(here::here("benchmarks", "hmc", "diagnostics.R"))

  # Recursively block until every AnvlArray leaf in a (possibly nested) list is
  # materialised on the device; non-array leaves are ignored. Used to make sure
  # asynchronous XLA work is finished both before and after the timed region.
  await_all <- function(x) {
    if (inherits(x, "AnvlArray")) {
      await(x)
    } else if (is.list(x)) {
      lapply(x, await_all)
    }
    invisible(NULL)
  }

  b_t <- nv_scalar(b, dtype = "f64", device = device)
  eps_t <- nv_scalar(eps, dtype = "f64", device = device)
  L_t <- nv_scalar(as.integer(L))

  # Potential energy U(theta) = -log p(theta), dropping additive constants.
  # theta is a (K, 2) tensor and U returns a K-vector of per-chain energies.
  U <- function(theta, b) {
    theta1 <- theta[, 1]
    theta2 <- theta[, 2]
    theta1^2 / 200 + (theta2 - b * theta1^2 + 100 * b)^2 / 2
  }

  # Differentiate the sum of the per-chain energies: because chain k's energy
  # depends only on theta_k, row k of the gradient is exactly the gradient of U
  # at theta_k, so the K chains' gradients are obtained in a single call.
  U_total <- function(theta, b) sum(U(theta, b))
  grad_U <- gradient(U_total, wrt = "theta")

  # Leapfrog (velocity Verlet) integrator: L steps of half-kick / drift /
  # half-kick. Every operation is element-wise on the leading chain axis. The
  # gradient is cached in the loop state so each iteration evaluates grad_U once.
  leapfrog <- function(theta, p, b, eps, L) {
    grad <- grad_U(theta, b)$theta
    out <- nv_while(
      list(theta = theta, p = p, grad = grad, i = 0L),
      \(theta, p, grad, i) i < L,
      \(theta, p, grad, i) {
        p <- p - 0.5 * eps * grad
        theta <- theta + eps * p
        grad <- grad_U(theta, b)$theta
        p <- p - 0.5 * eps * grad
        list(theta = theta, p = p, grad = grad, i = i + 1L)
      }
    )
    list(theta = out$theta, p = out$p)
  }

  # A single HMC transition for all K chains at once: draw momentum, integrate,
  # and make a per-chain Metropolis accept/reject decision.
  hmc_step <- function(theta, rng_state, b, eps, L) {
    rng_out <- nv_rnorm(shape(theta), rng_state, dtype = "f64")
    rng_state <- rng_out[[1L]]
    p <- rng_out[[2L]]

    H_current <- U(theta, b) + nv_reduce_sum(p^2, dims = 2L) / 2
    lf <- leapfrog(theta, p, b, eps, L)
    H_proposed <- U(lf$theta, b) + nv_reduce_sum(lf$p^2, dims = 2L) / 2
    log_accept <- H_current - H_proposed

    n_chains <- shape(theta)[1L]
    rng_out <- nv_runif(n_chains, rng_state, dtype = "f64")
    rng_state <- rng_out[[1L]]
    u <- rng_out[[2L]]

    accept <- log(u) < log_accept
    accept_2d <- nv_broadcast_to(nv_reshape(accept, c(n_chains, 1L)), shape(theta))
    new_theta <- nv_ifelse(accept_2d, lf$theta, theta)
    list(theta = new_theta, rng_state = rng_state)
  }

  theta0 <- nv_array(matrix(0, nrow = n_chains, ncol = 2L), dtype = "f64", device = device)
  rng_state0 <- nv_rng_state(seed, device = device)

  if (compile_loop) {
    # The whole sampling loop lifted into a single graph: draws are written into
    # a pre-allocated (n, K, 2) buffer. The sample count n is a static argument
    # because the buffer shape must be known at compile time.
    sample_many_r <- function(theta, rng_state, b, eps, L, n) {
      samples0 <- nv_fill(0, shape = c(n, shape(theta)[1L], 2L), dtype = "f64")
      out <- nv_while(
        list(theta = theta, rng_state = rng_state, samples = samples0, i = nv_scalar(0L)),
        \(theta, rng_state, samples, i) i < n,
        \(theta, rng_state, samples, i) {
          step <- hmc_step(theta, rng_state, b, eps, L)
          samples[i + 1L, , ] <- step$theta
          list(theta = step$theta, rng_state = step$rng_state, samples = samples, i = i + 1L)
        }
      )
      list(samples = out$samples, theta = out$theta, rng_state = out$rng_state)
    }
    hmc_sample_many <- jit(sample_many_r, static = "n")

    # Precompile (warms the JIT cache for both n_warmup and n_samples).
    w <- hmc_sample_many(theta0, rng_state0, b_t, eps_t, L_t, n = n_warmup)
    await_all(w)
    pre <- hmc_sample_many(w$theta, w$rng_state, b_t, eps_t, L_t, n = n_samples)
    await_all(pre)

    # Time compilation overhead (xla() compiles the raw function without
    # consuming the jit executable cache).
    t0 <- Sys.time()
    xla(sample_many_r, args = list(theta = theta0, rng_state = rng_state0, b = b_t, eps = eps_t, L = L_t, n = n_samples))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # Fresh warmup, awaited before the clock starts.
    warm <- hmc_sample_many(theta0, rng_state0, b_t, eps_t, L_t, n = n_warmup)
    await_all(warm)

    t0 <- Sys.time()
    out <- hmc_sample_many(warm$theta, warm$rng_state, b_t, eps_t, L_t, n = n_samples)
    await_all(out)
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    samples <- as_array(out$samples) # (n_samples, K, 2)
  } else {
    hmc_sample <- jit(hmc_step)

    sample_loop <- function(theta, rng_state, n) {
      thetas <- vector("list", n)
      for (i in seq_len(n)) {
        res <- hmc_sample(theta, rng_state, b_t, eps_t, L_t)
        theta <- res$theta
        rng_state <- res$rng_state
        thetas[[i]] <- theta
      }
      list(theta = theta, rng_state = rng_state, thetas = thetas)
    }

    # Time compilation overhead (xla() compiles a single step).
    t0 <- Sys.time()
    xla(hmc_step, args = list(theta = theta0, rng_state = rng_state0, b = b_t, eps = eps_t, L = L_t))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # Warmup (also warms the jit executable cache), awaited before the clock
    # starts.
    warm <- sample_loop(theta0, rng_state0, n_warmup)
    await_all(list(warm$theta, warm$rng_state))

    t0 <- Sys.time()
    out <- sample_loop(warm$theta, warm$rng_state, n_samples)
    await_all(out$theta)
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # (n_samples, K, 2)
    samples <- aperm(
      array(
        unlist(lapply(out$thetas, as_array)),
        dim = c(n_chains, 2L, n_samples)
      ),
      c(3L, 1L, 2L)
    )
  }

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
  source(here::here("benchmarks", "hmc", "diagnostics.R"))
  args <- list(
    n_chains = 16L,
    n_samples = 500L,
    n_warmup = 200L,
    L = 80L,
    eps = 0.2,
    b = 0.01,
    device = "cpu",
    seed = 42L
  )
  r1 <- do.call(time_anvl, c(args, list(compile_loop = TRUE)))
  r2 <- do.call(time_anvl, c(args, list(compile_loop = FALSE)))
  print(r1)
  print(r2)
}
