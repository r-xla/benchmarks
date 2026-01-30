import torch
import time
import math
from torch import nn


def time_pytorch(epochs, batch_size, n, n_layers, latent, p, device, seed):
    torch.manual_seed(seed)

    n = int(n)
    latent = int(latent)
    n_layers = int(n_layers)
    p = int(p)
    if n % batch_size != 0:
        raise ValueError("n must be divisible by batch_size")
    n_batches = int(n / batch_size)

    epochs = int(epochs)
    batch_size = int(batch_size)

    def make_network(p, latent, n_layers):
        if n_layers == 0:
            return nn.Linear(p, 1)
        layers = [nn.Linear(p, latent), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent, latent))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent, 1))
        return nn.Sequential(*layers)

    device = torch.device(device)

    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, 1, device=device)
    Y = X.matmul(beta) + torch.randn(n, 1, device=device) * 0.01

    net = make_network(p, latent, n_layers)
    net.to(device)

    lr = 0.0001
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def train_run(epochs):
        for _ in range(epochs):
            for (x, y) in dataloader:
                opt.zero_grad()
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                opt.step()

    # Warmup
    train_run(epochs=2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    train_run(epochs=epochs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t = time.time() - t0

    net.eval()

    # Evaluate training loss
    mean_loss = 0
    with torch.no_grad():
        for (x, y) in dataloader:
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            mean_loss += loss.item()
    mean_loss /= n_batches
    
    return {'time': t, 'loss': mean_loss}


if False:
    print(time_pytorch(
        epochs=1, batch_size=32, n=64, n_layers=4, latent=100,
        p=1000, device='cpu', seed=42
    ))
