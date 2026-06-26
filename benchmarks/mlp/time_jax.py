import os
import time

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit


def time_jax(epochs, batch_size, n, n_layers, latent, p, device, seed):
    n = int(n)
    latent = int(latent)
    n_layers = int(n_layers)
    p = int(p)
    epochs = int(epochs)
    batch_size = int(batch_size)
    if n % batch_size != 0:
        raise ValueError("n must be divisible by batch_size")
    n_batches = n // batch_size

    # Resolve the device. JAX exposes the CPU backend even when a GPU is the
    # default, so placing the inputs on `dev` is enough to force execution there.
    if device == "cuda":
        dev = jax.devices("gpu")[0]
    elif device == "cpu":
        dev = jax.devices("cpu")[0]
    else:
        dev = jax.devices()[0]

    lr = 0.0001
    key = random.PRNGKey(int(seed))

    # ----- data -----
    key, kx, kb, ke = random.split(key, 4)
    X = random.normal(kx, (n, p))
    beta = random.normal(kb, (p, 1))
    Y = X @ beta + random.normal(ke, (n, 1)) * 0.1

    # ----- model -----
    def init_params(key):
        dims = [p] + [latent] * n_layers + [1]
        params = []
        for nin, nout in zip(dims[:-1], dims[1:]):
            key, k = random.split(key)
            W = random.normal(k, (nin, nout)) * jnp.sqrt(2.0 / nin)
            b = jnp.zeros((1, nout))
            params.append((W, b))
        return params

    def model(x, params):
        for (W, b) in params[:-1]:
            x = jnp.maximum(x @ W + b, 0.0)
        W, b = params[-1]
        return x @ W + b

    def loss_fn(params, x, y):
        pred = model(x, params)
        return jnp.mean((pred - y) ** 2)

    # JIT-compile a single SGD step; the batch loop is driven from Python and
    # relies on JAX's asynchronous dispatch (analogous to the eager frameworks).
    @jit
    def step(params, x, y):
        l, grads = value_and_grad(loss_fn)(params, x, y)
        params = [
            (W - lr * gW, b - lr * gb)
            for (W, b), (gW, gb) in zip(params, grads)
        ]
        return params, l

    Xb = jax.device_put(X.reshape(n_batches, batch_size, p), dev)
    Yb = jax.device_put(Y.reshape(n_batches, batch_size, 1), dev)
    params = jax.device_put(init_params(key), dev)

    def train_run(params, n_epochs):
        loss = None
        for _ in range(n_epochs):
            for b in range(n_batches):
                params, loss = step(params, Xb[b], Yb[b])
        return params, loss

    # ----- compile time (first step triggers tracing + XLA compilation) -----
    t0 = time.time()
    p_c, l_c = step(params, Xb[0], Yb[0])
    jax.block_until_ready((p_c, l_c))
    compile_time = time.time() - t0

    # ----- warmup -----
    params, _ = train_run(params, 2)
    jax.block_until_ready(params)

    # ----- timed training -----
    t0 = time.time()
    params, _ = train_run(params, epochs)
    jax.block_until_ready(params)
    t = time.time() - t0

    # ----- evaluate training loss -----
    mean_loss = 0.0
    for b in range(n_batches):
        mean_loss += float(loss_fn(params, Xb[b], Yb[b]))
    mean_loss /= n_batches

    affinity = os.sched_getaffinity(0)
    ncores = len(affinity)
    n_params = int(sum(W.size + b.size for (W, b) in params))

    return {
        'time': t,
        'loss': mean_loss,
        'n_params': n_params,
        'ncores': ncores,
        'affinity': affinity,
        'compile_time': compile_time,
    }


if False:
    print(time_jax(
        epochs=1, batch_size=32, n=64, n_layers=4, latent=100,
        p=1000, device='cpu', seed=42
    ))
