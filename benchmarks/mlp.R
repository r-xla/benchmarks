# MLP Training Benchmark: Anvil vs Torch

library(anvil)
library(torch)
library(dotty)
library(bench)

# Benchmark configuration
N_STEPS <- 100L

# Configure torch to use all CPU threads
torch_set_num_threads(parallel::detectCores())
torch_set_num_interop_threads(parallel::detectCores())

# Generate random data: y = X * beta + eps
set.seed(42)
n_samples <- 10000L
n_features <- 100L
n_outputs <- 1L

X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)
beta <- matrix(rnorm(n_features * n_outputs), n_features, n_outputs)
y <- X %*% beta + matrix(rnorm(n_samples * n_outputs, sd = 0.1), n_samples, n_outputs)

# Training hyperparameters
batch_size <- 64L
learning_rate <- 0.01
hidden_dims <- c(n_features, 256, 128, n_outputs)

# ============================================================================
# Anvil Implementation
# ============================================================================

# Anvil helper functions
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
  nv_max(x, nv_scalar(0))
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

step <- jit(donate = c("x", "y", "params"), function(x, y, params) {
  .[l, .[grads]] <- value_and_gradient(loss, wrt = "params")(x, y, params)
  params <- Map(function(p_layer, g_layer) {
    Map(\(p, g) p - nv_scalar(learning_rate) * g, p_layer, g_layer)
  }, params, grads)
  list(l, params)
})

# ============================================================================
# Torch Implementation
# ============================================================================

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

# ============================================================================
# Benchmark Functions
# ============================================================================

train_anvil <- function(X, y, params, n_steps, batch_size, step_fn, seed = 123) {
  set.seed(seed)
  n_samples <- nrow(X)

  for (i in seq_len(n_steps)) {
    ids <- sample.int(n_samples, size = batch_size)
    bx <- nv_tensor(X[ids, , drop = FALSE])
    by <- nv_tensor(y[ids, , drop = FALSE])
    .[l, params] <- step_fn(bx, by, params)
  }

  invisible(NULL)
}

train_torch <- function(X, y, model, optimizer, n_steps, batch_size, seed = 123) {
  set.seed(seed)
  torch_manual_seed(seed)
  n_samples <- nrow(X)

  x_torch <- torch_tensor(X, dtype = torch_float32())
  y_torch <- torch_tensor(y, dtype = torch_float32())

  for (i in seq_len(n_steps)) {
    ids <- sample.int(n_samples, size = batch_size)
    bx <- x_torch[ids, , drop = FALSE]
    by <- y_torch[ids, , drop = FALSE]

    optimizer$zero_grad()
    pred <- model(bx)
    l <- torch_loss(pred, by)
    l$backward()
    optimizer$step()
  }

  invisible(NULL)
}

# Warmup Anvil (JIT compilation)
params_warmup <- init_model_params(hidden_dims)
train_anvil(X, y, params_warmup, n_steps = 1, batch_size, step)

# ============================================================================
# Run Benchmark
# ============================================================================

result <- bench::mark(
  anvil = {
    params <- init_model_params(hidden_dims)
    train_anvil(X, y, params, N_STEPS, batch_size, step)
  },
  torch = {
    model <- torch_mlp(hidden_dims)
    optimizer <- optim_sgd(model$parameters, lr = learning_rate, momentum = 0, dampening = 0)
    train_torch(X, y, model, optimizer, N_STEPS, batch_size)
  },
  iterations = 10,
  check = FALSE
)

print(result)
