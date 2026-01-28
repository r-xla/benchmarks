# Time anvil MLP training
# Supports two modes:
# - compile_loop = TRUE: Whole training loop is JIT compiled using nv_while
# - compile_loop = FALSE: Step function is JIT compiled, loop is in R

time_anvil <- function(epochs, batch_size, n_batches, n_layers, latent, p, device, seed, compile_loop = TRUE) {
  library(anvil)
  set.seed(seed)

  lr <- nv_scalar(0.0001, "f32")
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

  step_sgd <- function(x, y, params, lr) {
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
    # Uses mini-batch SGD with iota-based indexing
    train_anvil <- function(X, y, params, n_epochs, batch_size, n_batches) {
      X_t <- nv_tensor(X, dtype = "f32")
      y_t <- nv_tensor(y, dtype = "f32")

      # Create batch indices: iota of shape (n_batches, batch_size)
      indices <- nv_iota(1L, shape = n_batches * batch_size, dtype = "i32")
      indices <- nv_reshape(indices, shape = c(n_batches, batch_size))

      out <- nv_while(
        list(epoch = 1L, batch = 1L, p = params, l = nv_scalar(Inf, "f32", ambiguous = FALSE)),
        \(epoch, batch, p, l) epoch <= n_epochs,
        \(epoch, batch, p, l) {
          # Get indices for current batch and slice X, y
          batch_indices <- indices[batch, ]
          X_batch <- X_t[batch_indices, ]
          y_batch <- y_t[batch_indices, ]

          out <- step_sgd(X_batch, y_batch, p, lr)

          # Update batch and epoch counters
          next_batch <- nv_if(batch == n_batches, 1L, batch + 1L)
          next_epoch <- nv_if(batch == n_batches, epoch + 1L, epoch)

          list(epoch = next_epoch, batch = next_batch, p = out[[2L]], l = out[[1L]])
        }
      )
      list(loss = out$l, params = out$p)
    }

    # JIT compile the training function
    train_anvil_jit <- jit(train_anvil, static = c("X", "y", "n_epochs", "batch_size", "n_batches"))

    # Initialize parameters and warmup
    params <- init_model_params(hidden_dims)
    train_anvil_jit(X, Y, params, n_epochs = 1L, batch_size = batch_size, n_batches = n_batches)

    # Reinitialize for actual run
    params <- init_model_params(hidden_dims)

    # Timed run
    t0 <- Sys.time()
    result <- train_anvil_jit(X, Y, params, n_epochs = epochs, batch_size = batch_size, n_batches = n_batches)
    final_loss <- anvil::as_array(result$loss)
    # needed to sync
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  } else {
    # Mode 2: Step function is JIT compiled, loop is in R
    # Uses mini-batch SGD with R-level slicing
    step_anvil <- function(X_t, y_t, params, lr) {
      step_sgd(X_t, y_t, params, lr)
    }

    # JIT compile the step function
    step_anvil_jit <- jit(step_anvil, donate = c("X", "y", "params"))

    # Initialize parameters and warmup with first batch
    params <- init_model_params(hidden_dims)
    X_batch <- nv_tensor(X[1:batch_size, , drop = FALSE], dtype = "f32")
    Y_batch <- nv_tensor(Y[1:batch_size, , drop = FALSE], dtype = "f32")
    step_anvil_jit(X_batch, Y_batch, params, lr)

    # Reinitialize for actual run
    params <- init_model_params(hidden_dims)

    # Timed run with R loop - mini-batch SGD
    t0 <- Sys.time()
    for (epoch in seq_len(epochs)) {
      for (b in seq_len(n_batches)) {
        idx_start <- (b - 1L) * batch_size + 1L
        idx_end <- b * batch_size
        X_batch <- nv_tensor(X[idx_start:idx_end, , drop = FALSE], dtype = "f32")
        Y_batch <- nv_tensor(Y[idx_start:idx_end, , drop = FALSE], dtype = "f32")
        out <- step_anvil_jit(X_batch, Y_batch, params, lr)
        params <- out[[2L]]
      }
    }
    # needed to sync
    final_loss <- anvil::as_array(out[[1L]])
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  }

  list(time = time, loss = final_loss)
}
