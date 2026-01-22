# MLP Training Benchmark: Anvil vs Torch

library(anvil)
library(torch)
library(bench)

N_STEPS <- 1000L
N_FEATURES <- 10L
N_OUTPUS <- 1L
BATCH_SIZE <- 64L
LEARNING_RATE <- 0.01
HIDDEN_DIMS <- c(N_FEATURES, rep(100, 10), N_OUTPUS)

set.seed(42)
X <- matrix(rnorm(BATCH_SIZE * N_FEATURES), BATCH_SIZE, N_FEATURES)
beta <- matrix(rnorm(N_FEATURES * N_OUTPUS), N_FEATURES, N_OUTPUS)
y <- X %*% beta + matrix(rnorm(BATCH_SIZE * N_OUTPUS, sd = 0.1), BATCH_SIZE, N_OUTPUS)

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

train_anvil <- function(X, y, params, n_steps) {
  X <- nv_tensor(X, dtype = "f32")
  y <- nv_tensor(y, dtype = "f32")
  out <- nv_while(list(i = 1L, p = params, l = nv_scalar(Inf, "f32")), \(i, p, l) i <= n_steps, \(i, p, l) {
    out <- step(X, y, p)
    l <- out[[1L]]
    print(l)
    params <- out[[2L]]
    list(
      i = i + 1L,
      p = params,
      l = l
    )
  })
  l <- out[[1L]]
  params <- out[[2L]]
  l
}

train_torch <- function(X, y, model, optimizer, n_steps) {
  X <- torch_tensor(X, dtype = torch_float32())
  y <- torch_tensor(y, dtype = torch_float32())
  for (i in seq_len(n_steps)) {
    optimizer$zero_grad()
    pred <- model(X)
    l <- torch_loss(pred, y)
    l$backward()
    optimizer$step()
  }

  l$item()
}

# Warmup Anvil (JIT compilation)
params_warmup <- init_model_params(HIDDEN_DIMS)
train_anvil_jit <- jit(train_anvil, static = c("X", "y", "n_steps"))
train_anvil_jit(X, y, params_warmup, n_steps = 1)

# ============================================================================
# Run Benchmark
# ============================================================================

result <- bench::mark(
  anvil = {
    params <- init_model_params(HIDDEN_DIMS)
    loss <- train_anvil_jit(X, y, params, N_STEPS)
    loss
  },
  torch = {
    model <- torch_mlp(HIDDEN_DIMS)
    optimizer <- optim_sgd(model$parameters, lr = LEARNING_RATE, momentum = 0, dampening = 0)
    loss <- train_torch(X, y, model, optimizer, N_STEPS)
    loss
  },
  iterations = 1,
  check = FALSE,
  memory = FALSE
)

print(result)
