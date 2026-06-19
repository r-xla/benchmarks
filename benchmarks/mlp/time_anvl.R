# Time anvl MLP training
# Supports two modes:
# - compile_loop = TRUE: Whole training loop is JIT compiled using nv_while
# - compile_loop = FALSE: Step function is JIT compiled, loop is in R

time_anvl <- function(epochs, batch_size, n, n_layers, latent, p, device, seed, compile_loop = TRUE) {
  library(anvl)

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

  set.seed(seed)
  if ((n %% batch_size) != 0L) {
    stop("n must be divisible by batch_size")
  }
  n_batches <- n / batch_size

  lr <- nv_scalar(0.0001, "f32", device = device)

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
      W = nv_array(matrix(rnorm(nin * nout) * sqrt(2.0 / nin), nin, nout), dtype = "f32", device = device),
      b = nv_array(matrix(0, 1L, nout), dtype = "f32", device = device)
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

  mse <- function(pred, y) {
    mean((pred - y)^2)
  }

  loss_fn <- function(x, y, params) {
    pred <- model(x, params)
    mse(pred, y)
  }

  step_sgd_r <- function(X_batch, Y_batch, params, lr) {
    out <- value_and_gradient(loss_fn, wrt = "params")(X_batch, Y_batch, params)
    l <- out[[1L]]
    grads <- out[[2L]]$params
    params <- Map(function(p_layer, g_layer) {
      Map(\(p, g) p - lr * g, p_layer, g_layer)
    }, params, grads)
    list(l, params)
  }

  # JIT-compile a single evaluation step (the MSE of one batch) and loop over
  # the batches in R, averaging the per-batch losses.
  eval_step <- jit(function(X_batch, Y_batch, params) {
    pred <- model(X_batch, params)
    mse(pred, Y_batch)
  })

  eval_anvl <- function(X, Y, params, n_batches) {
    total_loss <- 0
    for (batch in seq_len(n_batches)) {
      loss <- eval_step(X[batch, ], Y[batch, ], params)
      total_loss <- total_loss + as_array(loss)
    }
    total_loss / n_batches
  }

  X_anvl <- nv_reshape(nv_array(X, dtype = "f32", device = device), shape = c(n_batches, batch_size, shape(X)[-1L]))
  Y_anvl <- nv_reshape(nv_array(Y, dtype = "f32", device = device), shape = c(n_batches, batch_size, shape(Y)[-1L]))

  if (compile_loop) {

    train_anvl_r <- function(X, Y, params, n_epochs, batch_size, n_batches, lr) {
      out <- nv_while(
        list(epoch = 1L, batch = 1L, p = params, l = Inf),
        \(epoch, batch, p, l) epoch <= n_epochs,
        \(epoch, batch, p, l) {
          X_batch <- X[batch, ]
          Y_batch <- Y[batch, ]

          out <- step_sgd_r(X_batch, Y_batch, p, lr)

          # Update batch and epoch counters
          next_batch <- nv_if(batch == n_batches, \() 1L, \() batch + 1L)
          next_epoch <- nv_if(batch == n_batches, \() epoch + 1L, \() epoch)

          list(epoch = next_epoch, batch = next_batch, p = out[[2L]], l = out[[1L]])
        }
      )
      list(loss = out$l, params = out$p)
    }

    train_anvl <- jit(train_anvl_r, static = c("batch_size", "n_batches"), device = device)

    params_ <- init_model_params(hidden_dims)

    # precompile
    out <- train_anvl(X_anvl, Y_anvl, params_, n_epochs = nv_scalar(1L, device = device), batch_size = batch_size, n_batches = n_batches, lr = lr)

    # time compilation overhead
    t0 <- Sys.time()
    xla(train_anvl_r, args = list(X = X_anvl, Y = Y_anvl, params = params_, n_epochs = nv_scalar(1L), batch_size = batch_size, n_batches = n_batches, lr = lr))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    # Sync
    as_array(out$loss)
  } else {
    step_sgd <- jit(step_sgd_r, donate = c("X_batch", "Y_batch", "params"))

    train_anvl <- function(X, Y, params, n_epochs, batch_size, n_batches, lr) {

      for (epoch in seq_len(n_epochs)) {
        for (b in seq_len(n_batches)) {
          idx_start <- (b - 1L) * batch_size + 1L
          idx_end <- b * batch_size
          X_batch <- nv_array(X[idx_start:idx_end, , drop = FALSE], "f32", device = device)
          Y_batch <- nv_array(Y[idx_start:idx_end, , drop = FALSE], "f32", device = device)
          out <- step_sgd(X_batch, Y_batch, params, lr)
          params <- out[[2L]]
        }
      }
      list(loss = out[[1L]], params = params)
    }

    X_batch <- nv_array(X[seq_len(batch_size), , drop = FALSE], "f32", device = device)
    Y_batch <- nv_array(Y[seq_len(batch_size), , drop = FALSE], "f32", device = device)
    params_ <- init_model_params(hidden_dims)
    # time compilation overhead (xla() compiles without consuming the buffers)
    t0 <- Sys.time()
    xla(step_sgd_r, args = list(X_batch = X_batch, Y_batch = Y_batch, params = params_, lr = lr))
    compile_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    # precompile: warm the donating jit cache (this consumes the buffers above)
    out <- step_sgd(X_batch, Y_batch, params_, lr)
    # Sync
    await_all(out)
  }

  params <- init_model_params(hidden_dims)

  # Await all inputs before starting the clock so that host-to-device transfers
  # and any pending compilation are not counted in the measured training time.
  if (compile_loop) {
    n_epochs_arg <- nv_scalar(epochs, device = device)
    await_all(list(X_anvl, Y_anvl, params, lr, n_epochs_arg))
  } else {
    await_all(list(params, lr))
  }

  await_all(list(X_anvl, Y_anvl))
  t0 <- Sys.time()
  result <- if (compile_loop) {
    train_anvl(X_anvl, Y_anvl, params, n_epochs = n_epochs_arg, batch_size = batch_size, n_batches = n_batches, lr = lr)
  } else {
    train_anvl(X, Y, params, n_epochs = epochs, batch_size = batch_size, n_batches = n_batches, lr = lr)
  }
  # Await all results (loss and updated params) before stopping the clock so the
  # timing covers the full asynchronous computation, not just the dispatch.
  await_all(result)
  time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  eval_loss <- eval_anvl(X_anvl, Y_anvl, result$params, n_batches = n_batches)

  n_params <- sum(sapply(flatten(result$params), function(p) prod(shape(p))))

  n_cores <- length(parallel::mcaffinity())
  list(time = time, loss = eval_loss, n_params = n_params, n_cores = n_cores, compile_time = compile_time)
}
if (FALSE) {
  args <- list(
    epochs = 10L,
    batch_size = 32L,
    n= 640,
    n_layers = 0L,
    latent = 100L,
    p = 10L,
    device = "cpu",
    seed = 42L
  )

  r1 <- do.call(time_anvl, c(args, list(compile_loop = TRUE)))
  r2 <- do.call(time_anvl, c(args, list(compile_loop = FALSE)))
  print(r1)
  print(r2)
}
