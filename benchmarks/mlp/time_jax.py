import jax
import jax.numpy as jnp
import time
import os
from functools import partial


def time_jax(epochs, batch_size, n, n_layers, latent, p, device, seed):
    n = int(n)
    latent = int(latent)
    n_layers = int(n_layers)
    p = int(p)
    epochs = int(epochs)
    batch_size = int(batch_size)

    if n % batch_size != 0:
        raise ValueError("n must be divisible by batch_size")
    n_batches = int(n / batch_size)

    if device == "cuda":
        jax_device = jax.devices("gpu")[0]
    elif device == "cpu":
        jax_device = jax.devices("cpu")[0]
    else:
        raise ValueError(f"Unsupported device: {device}")

    key = jax.random.PRNGKey(int(seed))

    # Initialize network parameters
    def init_params(key, p, latent, n_layers):
        params = []
        if n_layers == 0:
            k1, k2 = jax.random.split(key)
            w = jax.random.normal(k1, (p, 1)) / jnp.sqrt(p)
            b = jnp.zeros((1,))
            params.append((w, b))
        else:
            keys = jax.random.split(key, n_layers + 1)
            # First layer: p -> latent
            w = jax.random.normal(keys[0], (p, latent)) / jnp.sqrt(p)
            b = jnp.zeros((latent,))
            params.append((w, b))
            # Hidden layers: latent -> latent
            for i in range(1, n_layers):
                w = jax.random.normal(keys[i], (latent, latent)) / jnp.sqrt(latent)
                b = jnp.zeros((latent,))
                params.append((w, b))
            # Output layer: latent -> 1
            w = jax.random.normal(keys[n_layers], (latent, 1)) / jnp.sqrt(latent)
            b = jnp.zeros((1,))
            params.append((w, b))
        return params

    def forward(params, x):
        for w, b in params[:-1]:
            x = jnp.maximum(x @ w + b, 0)  # ReLU
        w, b = params[-1]
        return x @ w + b

    def loss_fn(params, x, y):
        y_hat = forward(params, x)
        return jnp.mean((y_hat - y) ** 2)

    @jax.jit
    def train_step(params, x, y, lr):
        grads = jax.grad(loss_fn)(params, x, y)
        return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

    key, k1, k2, k3 = jax.random.split(key, 4)
    params = init_params(k1, p, latent, n_layers)

    X = jax.random.normal(k2, (n, p))
    beta = jax.random.normal(k3, (p, 1))
    Y = X @ beta + jax.random.normal(key, (n, 1)) * 0.01

    # Place on device
    params = jax.device_put(params, jax_device)
    X = jax.device_put(X, jax_device)
    Y = jax.device_put(Y, jax_device)

    lr = 0.0001

    def train_run(params, epochs):
        for _ in range(epochs):
            for i in range(n_batches):
                x_batch = jax.lax.dynamic_slice(X, (i * batch_size, 0), (batch_size, p))
                y_batch = jax.lax.dynamic_slice(Y, (i * batch_size, 0), (batch_size, 1))
                params = train_step(params, x_batch, y_batch, lr)
        return params

    # Warmup
    params = train_run(params, epochs=2)
    jax.block_until_ready(params)

    t0 = time.time()
    params = train_run(params, epochs=epochs)
    jax.block_until_ready(params)
    t = time.time() - t0

    # Evaluate training loss
    mean_loss = 0.0
    for i in range(n_batches):
        x_batch = jax.lax.dynamic_slice(X, (i * batch_size, 0), (batch_size, p))
        y_batch = jax.lax.dynamic_slice(Y, (i * batch_size, 0), (batch_size, 1))
        mean_loss += float(loss_fn(params, x_batch, y_batch))
    mean_loss /= n_batches

    affinity = os.sched_getaffinity(0)
    ncores = len(affinity)
    n_params = sum(w.size + b.size for w, b in params)

    return {'time': t, 'loss': mean_loss, 'n_params': n_params, 'ncores': ncores, 'affinity': affinity, 'compile_time': 0}


if False:
    print(time_jax(
        epochs=1, batch_size=32, n=64, n_layers=4, latent=100,
        p=1000, device='cpu', seed=42
    ))
