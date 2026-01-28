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

  step_sgd <- function(x, y, params, lr) {
    out <- value_and_gradient(loss_fn, wrt = "params")(x, y, params)
    l <- out[[1L]]
    grads <- out[[2L]][[1L]]
    params <- Map(function(p_layer, g_layer) {
      Map(\(p, g) p - nv_scalar(lr, "f32") * g, p_layer, g_layer)
    }, params, grads)
    list(l, params)
  }

  mse <- function(pred, y) {
    diff <- pred - y
    mean(diff * diff)
  }

  if (compile_loop) {
    # Mode 1: Whole training loop is JIT compiled using nv_while
    # Uses mini-batch SGD with iota-based indexing
    train_anvil <- function(X, y, params, n_epochs, batch_size, n_batches) {
      X_t <- nv_tensor(X, dtype = "f32")
      y_t <- nv_tensor(y, dtype = "f32")

      # Create batch indices: iota of shape (n_batches, batch_size)
      indices <- nv_iota(c(n_batches, batch_size), 1L, dtype = "s32")

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

    eval_anvil <- function(X, y, params, batch_size, n_batches) {
      X_t <- nv_tensor(X, dtype = "f32")
      y_t <- nv_tensor(y, dtype = "f32")

      indices <- nv_iota(c(n_batches, batch_size), 1L, dtype = "s32")

      out <- nv_while(
        list(batch = 1L, total_loss = nv_scalar(0, "f32", ambiguous = FALSE)),
        \(batch, total_loss) batch <= n_batches,
        \(batch, total_loss) {
          batch_indices <- indices[batch, ]
          X_batch <- X_t[batch_indices, ]
          y_batch <- y_t[batch_indices, ]

          pred <- model(X_batch, params)
          loss <- mse(pred, y_batch)

          list(batch = batch + 1L, total_loss = total_loss + loss)
        }
      )
      out$total_loss / nv_scalar(n_batches, "f32")
    }

    # JIT compile the training function
    train_anvil_jit <- jit(train_anvil, static = c("X", "y", "n_epochs", "batch_size", "n_batches"))
    eval_anvil_jit <- jit(eval_anvil, static = c("X", "y", "batch_size", "n_batches"))

    # Initialize parameters and warmup
    params <- init_model_params(hidden_dims)
    train_anvil_jit(X, Y, params, n_epochs = 1L, batch_size = batch_size, n_batches = n_batches)

    # Reinitialize for actual run
    params <- init_model_params(hidden_dims)

    # Timed run
    t0 <- Sys.time()
    result <- train_anvil_jit(X, Y, params, n_epochs = epochs, batch_size = batch_size, n_batches = n_batches)
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    final_loss <- anvil::as_array(eval_anvil_jit(X, Y, result$params, batch_size = batch_size, n_batches = n_batches))
  } else {
    # Mode 2: Step function is JIT compiled, loop is in R
    # Uses mini-batch SGD with R-level slicing
    step_anvil <- function(X_t, y_t, params, lr) {
      step_sgd(X_t, y_t, params, lr)
    }

    # JIT compile the step function
    step_anvil_jit <- jit(step_anvil)

    eval_batch <- function(X_t, y_t, params) {
      pred <- model(X_t, params)
      mse(pred, y_t)
    }
    eval_batch_jit <- jit(eval_batch)

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
    time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # Evaluation loop
    total_loss <- 0
    for (b in seq_len(n_batches)) {
      idx_start <- (b - 1L) * batch_size + 1L
      idx_end <- b * batch_size
      X_batch <- nv_tensor(X[idx_start:idx_end, , drop = FALSE], dtype = "f32")
      Y_batch <- nv_tensor(Y[idx_start:idx_end, , drop = FALSE], dtype = "f32")
      total_loss <- total_loss + anvil::as_array(eval_batch_jit(X_batch, Y_batch, params))
    }
    final_loss <- total_loss / n_batches
  }

  list(time = time, loss = final_loss, cuda_memory = NA)
}
