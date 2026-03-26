import gpytorch
import torch
import time
import os


def time_gpytorch(n, p, n_steps, learning_rate, device, seed):
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)

    n = int(n)
    p = int(p)
    n_steps = int(n_steps)

    device = torch.device(device)

    # Generate data (same structure as anvil)
    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, 1, device=device)
    Y = (X @ beta + torch.randn(n, 1, device=device) * 0.1).squeeze(-1)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, Y, likelihood)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Set initial hyperparameters to match anvil
    # anvil starts: lengthscale=1, signal_var=1, noise_var=exp(-1)
    model.covar_module.base_kernel.lengthscale = 1.0
    model.covar_module.outputscale = 1.0
    likelihood.noise = torch.tensor([torch.exp(torch.tensor(-1.0)).item()])

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_run(steps):
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, Y)
            loss.backward()
            optimizer.step()

    # Warmup
    train_run(2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Ensure Cholesky is used (not CG) regardless of n
    with gpytorch.settings.max_cholesky_size(n + 1):
        t0 = time.time()
        train_run(n_steps)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t = time.time() - t0

        # Evaluate final loss
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(X)
            final_loss = -mll(output, Y).item()

    n_params = sum(param.numel() for param in model.parameters())

    try:
        ncores = len(os.sched_getaffinity(0))
    except AttributeError:
        ncores = os.cpu_count()

    return {
        'time': float(t),
        'loss': float(final_loss),
        'n_params': int(n_params),
        'n_cores': int(ncores),
        'compile_time': 0.0
    }


if False:
    print(time_gpytorch(
        n=256, p=10, n_steps=50,
        learning_rate=0.01, device='cpu', seed=42
    ))
