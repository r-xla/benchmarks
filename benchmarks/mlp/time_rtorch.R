# Time R torch MLP training

time_rtorch <- function(epochs, batch_size, n_batches, n_layers, latent, p, device, seed) {
  library(torch)
  torch_set_num_threads(1L)
  torch_manual_seed(seed)

  lr <- 0.0001
  n <- batch_size * n_batches

  make_network <- function(p, latent, n_layers) {
    if (n_layers == 0L) return(nn_linear(p, 1L))
    layers <- list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1L)) {
      layers <- c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers <- c(layers, list(nn_linear(latent, 1L)))
    net <- do.call(nn_sequential, args = layers)
    net
  }

  X <- torch_randn(n, p, device = device)
  beta <- torch_randn(p, 1L, device = device)
  Y <- X$matmul(beta) + torch_randn(n, 1L, device = device) * 0.1^2

  net <- make_network(p, latent, n_layers)
  loss_fn <- nn_mse_loss()
  net$to(device = device)

  dataset <- torch::dataset(
    initialize = function(X, Y) {
      self$X <- X
      self$Y <- Y
    },
    .getbatch = function(i) {
      list(x = self$X[i, drop = FALSE], y = self$Y[i, drop = FALSE])
    },
    .length = function() {
      nrow(self$X)
    }
  )(X, Y)

  train_run <- function(epochs) {
    opt <- optim_sgd(net$parameters, lr = lr)
    dataloader <- torch::dataloader(dataset, batch_size = batch_size, shuffle = FALSE)
    t0 <- Sys.time()
    for (epoch in seq(epochs)) {
      step <- 0L
      iter <- dataloader_make_iter(dataloader)
      while (step < length(dataloader)) {
        batch <- dataloader_next(iter)
        y_hat <- net$forward(batch$x)
        opt$zero_grad()
        loss <- loss_fn(y_hat, batch$y)
        loss$backward()
        opt$step()
        step <- step + 1L
      }
    }
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  }

  eval_run <- function() {
    mean_loss <- 0
    with_no_grad({
      dataloader <- torch::dataloader(dataset, batch_size = batch_size, shuffle = FALSE)
      coro::loop(for (batch in dataloader) {
        y_hat <- net$forward(batch[[1L]])
        loss <- loss_fn(y_hat, batch[[2L]])
        mean_loss <- mean_loss + loss$item()
      })
    })
    mean_loss / n_batches
  }

  # Warmup
  train_run(2L)

  if (device == "cuda") cuda_synchronize()
  list(time = time, loss = eval_run())
}
