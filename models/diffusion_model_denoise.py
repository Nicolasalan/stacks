import torch
import torch.nn as nn
from torch import Tensor

# dados limpos -> x0
data: Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)

print("Dados limpos", data)

# modelo -> Sθ
class Model(nn.Module):
    def __init__(self):
        super().__init__() # ??
        self.net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x_noisy, time):
        input_tensor = torch.stack([x_noisy, time])
        return self.net(input_tensor).squeeze()

model = Model()
optm = torch.optim.Adam(model.parameters(), lr=0.02)

for ep in range(200):
    total_loss = 0
    for x0 in data:
        sigma = torch.rand(1).item() # nivel de ruido -> σ(τ)
        # adiciona ruido nos dados -> x_tau = x0 + sigma * epsilon
        # drif para adicionar ruido
        x_tau = x0 + sigma * torch.randn(1) # xτ -> x_tau (ou seja nivel de ruido)

        # Dθ - x0_pred
        x0_pred = model(x_tau.squeeze(), torch.tensor(sigma).squeeze())

        # L(θ) = E[‖Sθ(xτ,τ) - ∇xτ log p0τ(xτ|x0)‖²] -> loss: ||D_theta(x_tau, tau) - x0||^2
        loss = (x0_pred - x0)**2

        optm.zero_grad()
        loss.backward()
        optm.step()

        total_loss += loss.item()

    if ep % 50 == 0:
        print("Loss: ", total_loss)


original: float = 3.0
sigma: float = 0.5

noisy: float = original + sigma + torch.randn(1).item()
denoised: Tensor = model(torch.tensor(noisy), torch.tensor(sigma))

print("original: ", original)
print("Noisy: ", noisy)
print("Denoised: ", denoised.item())
