# Time anvil GP hyperparameter training
# Supports two modes:
# - compile_loop = TRUE: Whole training loop is JIT compiled using nv_while
# - compile_loop = FALSE: Step function is JIT compiled, loop is in R

time_anvil <- function(n, p, n_steps, learning_rate, device, seed, compile_loop = TRUE) {
  library(anvil)
  set.seed(seed)

  lr <- nv_scalar(learning_rate, "f32", device = device)

  # Generate data
  X <- matrix(rnorm(n * p), n, p)
  beta <- matrix(rnorm(p), p, 1L)
  Y <- X %*% beta + matrix(rnorm(n, sd = 0.1), n, 1L)

  # RBF kernel (from anvil GP vignette)
  rbf_kernel_matrix <- function(X1, X2, lengthscale, signal_var) {
    n1 <- shape(X1)[1L]
    m1 <- shape(X2)[1L]
    d <- shape(X1)[2L]
    diff <- nv_broadcast_arrays(
      nv_reshape(X1, c(n1, 1L, d)),
      nv_reshape(X2, c(1L, m1, d))
    )
    diff <- diff[[1L]] - diff[[2L]]
    sq_dist <- nv_reduce_sum(diff * diff, dims = 3L)
    signal_var * exp(-sq_dist / (2 * lengthscale^2))
  }

  neg_log_marginal_likelihood <- function(kernel, X, y, lengthscale, signal_var, noise_var) {
    n_obs <- shape(y)[1L]
    K <- kernel(X, X, lengthscale, signal_var)
    eye <- nv_eye(n_obs, dtype = dtype(K))
    K_y <- K + noise_var * eye
    L <- nv_cholesky(K_y, lower = TRUE)
    alpha <- nv_solve(K_y, y)
    data_fit <- 0.5 * nv_reduce_sum(y * alpha, dims = c(1L, 2L))
    diag_L <- nv_reduce_sum(L * eye, dims = 2L)
    log_det <- nv_reduce_sum(log(diag_L), dims = 1L)
    data_fit + log_det
  }

  grad_params <- c("log_lengthscale", "log_signal_var", "log_noise_var")
  nll_grad <- gradient(\(X, y, log_lengthscale, log_signal_var, log_noise_var) {
    neg_log_marginal_likelihood(
      rbf_kernel_matrix, X, y,
      exp(log_lengthscale), exp(log_signal_var), exp(log_noise_var)
    )
  }, wrt = grad_params)

  eval_nll <- jit(function(X, y, log_lengthscale, log_signal_var, log_noise_var) {
    neg_log_marginal_likelihood(
      rbf_kernel_matrix, X, y,
      exp(log_lengthscale), exp(log_signal_var), exp(log_noise_var)
    )
  })

  X_t <- nv_tensor(X, dtype = "f32", device = device)
  y_t <- nv_tensor(Y, dtype = "f32", device = device)

  if (compile_loop) {

    train_gp_r <- function(X, y, lengthscale, signal_var, noise_var, n_steps, learning_rate) {
      result <- nv_while(
        list(
          log_lengthscale = log(lengthscale),
          log_signal_var = log(signal_var),
          log_noise_var = log(noise_var),
          step = 0L
        ),
        \(log_lengthscale, log_signal_var, log_noise_var, step) step < n_steps,
        \(log_lengthscale, log_signal_var, log_noise_var, step) {
          grads <- nll_grad(X, y, log_lengthscale, log_signal_var, log_noise_var)
          list(
            log_lengthscale = log_lengthscale - learning_rate * grads$log_lengthscale,
            log_signal_var = log_signal_var - learning_rate * grads$log_signal_var,
            log_noise_var = log_noise_var - learning_rate * grads$log_noise_var,
            step = step + 1L
          )
        }
      )
      list(
        log_lengthscale = result$log_lengthscale,
        log_signal_var = result$log_signal_var,
        log_noise_var = result$log_noise_var
      )
    }

    train_gp <- jit(train_gp_r, device = device)

    init_ls <- nv_scalar(1, "f32", device = device)
    init_sv <- nv_scalar(1, "f32", device = device)
    init_nv <- nv_scalar(exp(-1), "f32", device = device)

    # precompile
    out <- train_gp(X_t, y_t, init_ls, init_sv, init_nv,
      n_steps = nv_scalar(1L, device = device), learning_rate = lr)

    # time compilation overhead
    t0 <- Sys.time()
    xla(train_gp_r, args = list(
      X = X_t, y = y_t, lengthscale = init_ls, signal_var = init_sv, noise_var = init_nv,
      n_steps = nv_scalar(1L), learning_rate = lr
    ))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    # Sync
    as_array(out$log_lengthscale)

  } else {
    step_gp_r <- function(X, y, log_lengthscale, log_signal_var, log_noise_var, learning_rate) {
      grads <- nll_grad(X, y, log_lengthscale, log_signal_var, log_noise_var)
      list(
        log_lengthscale = log_lengthscale - learning_rate * grads$log_lengthscale,
        log_signal_var = log_signal_var - learning_rate * grads$log_signal_var,
        log_noise_var = log_noise_var - learning_rate * grads$log_noise_var
      )
    }

    step_gp <- jit(step_gp_r)

    init_log_ls <- nv_scalar(0, "f32", device = device)
    init_log_sv <- nv_scalar(0, "f32", device = device)
    init_log_nv <- nv_scalar(-1, "f32", device = device)

    # precompile
    out <- step_gp(X_t, y_t, init_log_ls, init_log_sv, init_log_nv, lr)

    # time compilation overhead
    t0 <- Sys.time()
    xla(step_gp_r, args = list(
      X = X_t, y = y_t, log_lengthscale = init_log_ls, log_signal_var = init_log_sv,
      log_noise_var = init_log_nv, learning_rate = lr
    ))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    # Sync
    as_array(out$log_lengthscale)
  }

  # Timed run
  init_ls <- nv_scalar(1, "f32", device = device)
  init_sv <- nv_scalar(1, "f32", device = device)
  init_nv <- nv_scalar(exp(-1), "f32", device = device)

  t0 <- Sys.time()
  if (compile_loop) {
    result <- train_gp(X_t, y_t, init_ls, init_sv, init_nv,
      n_steps = nv_scalar(as.integer(n_steps), device = device), learning_rate = lr)
  } else {
    log_ls <- nv_scalar(0, "f32", device = device)
    log_sv <- nv_scalar(0, "f32", device = device)
    log_nv <- nv_scalar(-1, "f32", device = device)
    for (i in seq_len(n_steps)) {
      out <- step_gp(X_t, y_t, log_ls, log_sv, log_nv, lr)
      log_ls <- out$log_lengthscale
      log_sv <- out$log_signal_var
      log_nv <- out$log_noise_var
    }
    result <- out
  }
  # Sync
  final_val <- as_array(result$log_lengthscale)
  time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # Evaluate final loss
  eval_loss <- eval_nll(X_t, y_t, result$log_lengthscale, result$log_signal_var, result$log_noise_var)
  eval_loss <- as_array(eval_loss)

  n_cores <- length(parallel::mcaffinity())
  list(time = time, loss = eval_loss, n_params = 3L, n_cores = n_cores, compile_time = compile_time)
}

if (FALSE) {
  args <- list(
    n = 256L,
    p = 10L,
    n_steps = 50L,
    learning_rate = 0.01,
    device = "cpu",
    seed = 42L
  )

  r1 <- do.call(time_anvil, c(args, list(compile_loop = TRUE)))
  r2 <- do.call(time_anvil, c(args, list(compile_loop = FALSE)))
  print(r1)
  print(r2)
}
