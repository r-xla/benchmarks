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
  hidden_dims <- c(p, rep(latent, n_layers), 1L)

  # Linear layer
  linear <- function(x, params) {
    # matmul is batched
    out <- x %*% params$W
    # out has shape (n_batch, d_out)
    # bias has shape (1, d_out) -> broadcast 
    out + nv_broadcast_to(params$b, shape = shape(out))
  }

  init_linear_params <- function(nin, nout) {
    list(
      W = nv_tensor(matrix(rnorm(nin * nout) * sqrt(2.0 / nin), nin, nout), dtype = "f32"),
      b = nv_tensor(matrix(0, 1L, nout), dtype = "f32")
    )
  }
  
  batch_tensor <- function(batch_size, n_batches) {
    nv_iota(1L, c(n_batches, batch_size), dtype = "i32") |> 
      nv_reshape(shape = c(n_batches, batch_size))
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

  mse <- function(pred, y) {
    mean((pred - y)^2)
  }

  loss_fn <- function(x, y, params) {
    pred <- model(x, params)
    mse(pred, y)
  }

  step_sgd <- function(X_batch, Y_batch, params, lr) {
    out <- value_and_gradient(loss_fn, wrt = "params")(X_batch, Y_batch, params)
    l <- out[[1L]]
    grads <- out[[2L]][[1L]]
    params <- Map(function(p_layer, g_layer) {
      Map(\(p, g) p - lr * g, p_layer, g_layer)
    }, params, grads)
    list(l, params)
  }
  
  eval_anvil <- jit(function(X, Y, params, batch_size, n_batches) {
    indices <- batch_tensor(batch_size, n_batches)
    out <- nv_while(
      list(batch = 1L, total_loss = 0.0),
      \(batch, total_loss) batch <= n_batches,
      \(batch, total_loss) {
        batch_indices <- indices[batch, ]
        print(batch_indices)
        X_batch <- X[batch_indices, ]
        Y_batch <- Y[batch_indices, ]
        pred <- model(X_batch, params)
        loss <- mse(pred, Y_batch)
        list(batch = batch + 1L, total_loss = total_loss + loss)
      }
    )
    out$total_loss / n_batches
  }, static = c("n_batches", "batch_size"))
  
  X_anvil <- nv_tensor(X, dtype = "f32")
  Y_anvil <- nv_tensor(Y, dtype = "f32")

  if (compile_loop) {

    train_anvil <- jit(function(X, Y, params, n_epochs, batch_size, n_batches) {
      indices <- batch_tensor(batch_size, n_batches)

      out <- nv_while(
        list(epoch = 1L, batch = 1L, p = params, l = Inf),
        \(epoch, batch, p, l) epoch <= n_epochs,
        \(epoch, batch, p, l) {
          # Get indices for current batch and slice X, y
          batch_indices <- indices[batch, ]
          X_batch <- X[batch_indices, ]
          Y_batch <- Y[batch_indices, ]

          out <- step_sgd(X_batch, Y_batch, p, lr)

          # Update batch and epoch counters
          next_batch <- nv_if(batch == n_batches, 1L, batch + 1L)
          next_epoch <- nv_if(batch == n_batches, epoch + 1L, epoch)

          list(epoch = next_epoch, batch = next_batch, p = out[[2L]], l = out[[1L]])
        }
      )
      list(loss = out$l, params = out$p)
    }, static = c("n_epochs", "batch_size", "n_batches"))

    # JIT
    out <- train_anvil(X_anvil, Y_anvil, init_model_params(hidden_dims), n_epochs = 1L, batch_size = batch_size, n_batches = n_batches)
    # Sync
    as_array(out$loss)
  } else {
    step_sgd <- jit(step_sgd, donate = c("X_batch", "Y_batch", "params"))
    
    train_anvil <- function(X, Y, params, n_epochs, batch_size, n_batches) {

      for (epoch in seq_len(n_epochs)) {
        for (b in seq_len(n_batches)) {
          idx_start <- (b - 1L) * batch_size + 1L
          idx_end <- b * batch_size
          X_batch <- nv_tensor(X[idx_start:idx_end, , drop = FALSE], "f32")
          Y_batch <- nv_tensor(Y[idx_start:idx_end, , drop = FALSE], "f32")
          out <- step_sgd(X_batch, Y_batch, params, lr)
          params <- out[[2L]]
        }
      }
      list(loss = out[[1L]], params = params)
    }
    
    X_batch <- nv_tensor(X[seq_len(batch_size), , drop = FALSE], "f32")
    Y_batch <- nv_tensor(Y[seq_len(batch_size), , drop = FALSE], "f32")
    # JIT
    out <- step_sgd(X_batch, Y_batch, init_model_params(hidden_dims), lr)
    # Sync
    as_array(out[[1L]])
  }

  params <- init_model_params(hidden_dims)
  # Sync
  as_array(params[[1L]][[1L]])

  t0 <- Sys.time()
  result <- if (compile_loop) {
    train_anvil(X_anvil, Y_anvil, params, n_epochs = epochs, batch_size = batch_size, n_batches = n_batches)
  } else {
    train_anvil(X, Y, params, n_epochs = epochs, batch_size = batch_size, n_batches = n_batches)
  }
  # Need to sync once we have async XLA API
  # TODO: Does this sync the whole stream? We need an API like torch_cuda_synchronize() to do this properly I think.
  final_loss <- anvil::as_array(result$loss)
  time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  eval_loss <- eval_anvil(X_anvil, Y_anvil, result$params, batch_size = batch_size, n_batches = n_batches)

  list(time = time, loss = eval_loss, ncpus = length(parallel::mcaffinity()))
}
if (FALSE) {
  args <- list(
    epochs = 10L,
    batch_size = 320L,
    n_batches = 64L,
    n_layers = 0L,
    latent = 100L,
    p = 10L,
    device = "cpu",
    seed = 42L
  )

  r1 <- do.call(time_anvil, c(args, list(compile_loop = TRUE)))
  r2 <- do.call(time_anvil, c(args, list(compile_loop = FALSE)))
  print(r1)
  print(r2)
}
