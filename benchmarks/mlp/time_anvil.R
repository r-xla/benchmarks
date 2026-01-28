# Time anvil MLP training
# Supports two modes:
# - compile_loop = TRUE: Whole training loop is JIT compiled using nv_while
# - compile_loop = FALSE: Step function is JIT compiled, loop is in R

time_anvil <- function(epochs, batch_size, n_batches, n_layers, latent, p, device, seed, compile_loop = TRUE) {
  library(anvil)
  set.seed(seed)

  lr <- 0.0001
  n <- batch_size * n_batches

  # Create data
  X <- matrix(rnorm(n * p), n, p)
  beta <- matrix(rnorm(p * 1L), p, 1L)
  Y <- X %*% beta + matrix(rnorm(n * 1L, sd = 0.1), n, 1L)

  # Build hidden dimensions
  if (n_layers == 0L) {
    hidden_dims <- c(p, 1L)
  } else {
    hidden_dims <- c(p, rep(latent, n_layers), 1L)
  }

  # Linear layer
  linear <- function(x, params) {
    out <- x %*% params$W
    out + nv_broadcast_to(params$b, shape = shape(out))
  }

  init_linear_params <- function(nin, nout) {
    list(
      W = nv_tensor(matrix(rnorm(nin * nout) * sqrt(2.0 / nin), nin, nout), dtype = "f32"),
      b = nv_tensor(matrix(0, 1L, nout), dtype = "f32")
    )
  }

  relu <- function(x) {
    nv_max(x, 0)
  }

  model <- function(x, params) {
    for (p in params[-length(params)]) {
      x <- linear(x, p)
      x <- relu(x)
    }
    linear(x, params[[length(params)]])
  }

  init_model_params <- function(hidden) {
    params <- list()
    for (i in seq_along(hidden[-1L])) {
      params[[i]] <- init_linear_params(hidden[i], hidden[i + 1L])
    }
    params
  }

  loss_fn <- function(x, y, params) {
    pred <- model(x, params)
    diff <- pred - y
    nv_reduce_mean(diff * diff, dims = c(1L, 2L))
  }

  step_sgd <- function(x, y, params) {
    out <- value_and_gradient(loss_fn, wrt = "params")(x, y, params)
    l <- out[[1L]]
    grads <- out[[2L]][[1L]]
    params <- Map(function(p_layer, g_layer) {
      Map(\(p, g) p - nv_scalar(lr, "f32") * g, p_layer, g_layer)
    }, params, grads)
    list(l, params)
  }

  if (compile_loop) {
    # Mode 1: Whole training loop is JIT compiled using nv_while
    train_anvil <- function(X, y, params, n_epochs) {
      X_t <- nv_tensor(X, dtype = "f32")
      y_t <- nv_tensor(y, dtype = "f32")

      out <- nv_while(
        list(i = 1L, p = params, l = nv_scalar(Inf, "f32", ambiguous = FALSE)),
        \(i, p, l) i <= n_epochs,
        \(i, p, l) {
          out <- step_sgd(X_t, y_t, p)
          list(i = i + 1L, p = out[[2L]], l = out[[1L]])
        }
      )
      list(loss = out$l, params = out$p)
    }

    # JIT compile the training function
    train_anvil_jit <- jit(train_anvil, static = c("X", "y", "n_epochs"))

    # Initialize parameters and warmup
    params <- init_model_params(hidden_dims)
    train_anvil_jit(X, Y, params, n_epochs = 1L)

    # Reinitialize for actual run
    params <- init_model_params(hidden_dims)

    # Timed run
    t0 <- Sys.time()
    result <- train_anvil_jit(X, Y, params, n_epochs = epochs)
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    final_loss <- anvil::as_array(result$loss)
  } else {
    # Mode 2: Step function is JIT compiled, loop is in R
    step_anvil <- function(X, y, params) {
      X_t <- nv_tensor(X, dtype = "f32")
      y_t <- nv_tensor(y, dtype = "f32")
      step_sgd(X_t, y_t, params)
    }

    # JIT compile the step function
    step_anvil_jit <- jit(step_anvil, static = c("X", "y"))

    # Initialize parameters and warmup
    params <- init_model_params(hidden_dims)
    step_anvil_jit(X, Y, params)

    # Reinitialize for actual run
    params <- init_model_params(hidden_dims)

    # Timed run with R loop
    t0 <- Sys.time()
    for (i in seq_len(epochs)) {
      out <- step_anvil_jit(X, Y, params)
      params <- out[[2L]]
    }
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    final_loss <- anvil::as_array(out[[1L]])
  }

  list(time = time, loss = final_loss, cuda_memory = NA)
}
