# MLP Training Benchmark: Anvil vs Torch

library(anvil)
library(torch)
library(bench)

N_STEPS <- 1000L
N_SAMPLES <- 10000L
N_FEATURES <- 10L
N_OUTPUS <- 1L
BATCH_SIZE <- 64L
LEARNING_RATE <- 0.01
HIDDEN_DIMS <- c(N_FEATURES, rep(10, 10), N_OUTPUS)

set.seed(42)
X <- matrix(rnorm(N_SAMPLES * N_FEATURES), N_SAMPLES, N_FEATURES)
beta <- matrix(rnorm(N_FEATURES * N_OUTPUS), N_FEATURES, N_OUTPUS)
y <- X %*% beta + matrix(rnorm(N_SAMPLES * N_OUTPUS, sd = 0.1), N_SAMPLES, N_OUTPUS)

# Training hyperparameters

linear <- function(x, params) {
  out <- x %*% params$W
  out + nv_broadcast_to(params$b, shape = shape(out))
}

init_linear_params <- function(nin, nout) {
  list(
    W = nv_tensor(matrix(rnorm(nin * nout) * sqrt(2.0 / nin), nin, nout)),
    b = nv_tensor(matrix(0, 1, nout))
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
  for (i in seq_along(hidden[-1])) {
    params[[i]] <- init_linear_params(hidden[i], hidden[i + 1])
  }
  params
}

loss <- function(x, y, params) {
  pred <- model(x, params)
  diff <- pred - y
  nv_reduce_mean(diff * diff, dims = c(1, 2))
}

step <- function(x, y, params) {
  out <- value_and_gradient(loss, wrt = "params")(x, y, params)
  l <- out[[1L]]
  grads <- out[[2L]][[1L]]
  params <- Map(function(p_layer, g_layer) {
    Map(\(p, g) p - nv_scalar(LEARNING_RATE) * g, p_layer, g_layer)
  }, params, grads)
  list(l, params)
}

torch_mlp <- nn_module(
  initialize = function(hidden_dims) {
    self$layers <- nn_module_list()
    for (i in seq_along(hidden_dims[-1])) {
      self$layers$append(nn_linear(hidden_dims[i], hidden_dims[i + 1]))
    }
  },

  forward = function(x) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x)
      if (i < length(self$layers)) {
        x <- nnf_relu(x)
      }
    }
    x
  }
)

torch_loss <- function(pred, targets) {
  nnf_mse_loss(pred, targets)
}

train_anvil <- function(X, y, params, n_steps, batch_size, step_fn, seed = 123) {
  set.seed(seed)
  n_samples <- nrow(X)

  for (i in seq_len(n_steps)) {
    ids <- sample.int(n_samples, size = batch_size)
    bx <- nv_tensor(X[ids, , drop = FALSE])
    by <- nv_tensor(y[ids, , drop = FALSE])
    out <- step_fn(bx, by, params)
    l <- out[[1L]]
    params <- out[[2L]]
  }

  anvil::as_array(l)
}

train_torch <- function(X, y, model, optimizer, n_steps, batch_size, seed = 123) {
  set.seed(seed)
  torch_manual_seed(seed)
  n_samples <- nrow(X)

  for (i in seq_len(n_steps)) {
    ids <- sample.int(n_samples, size = batch_size)
    bx <- torch_tensor(X[ids, , drop = FALSE])
    by <- torch_tensor(y[ids, , drop = FALSE])

    optimizer$zero_grad()
    pred <- model(bx)
    l <- torch_loss(pred, by)
    l$backward()
    optimizer$step()
  }

  l$item()
}

step <- jit(step, donate = c("x", "y", "params"))

# Warmup Anvil (JIT compilation)
params_warmup <- init_model_params(HIDDEN_DIMS)
train_anvil(X, y, params_warmup, n_steps = 1, BATCH_SIZE, step)

# ============================================================================
# Run Benchmark
# ============================================================================

result <- bench::mark(
  anvil = {
    params <- init_model_params(HIDDEN_DIMS)
    train_anvil(X, y, params, N_STEPS, BATCH_SIZE, step)
  },
  torch = {
    model <- torch_mlp(HIDDEN_DIMS)
    optimizer <- optim_sgd(model$parameters, lr = LEARNING_RATE, momentum = 0, dampening = 0)
    train_torch(X, y, model, optimizer, N_STEPS, BATCH_SIZE)
  },
  iterations = 1,
  check = FALSE,
  memory = FALSE
)

print(result)
