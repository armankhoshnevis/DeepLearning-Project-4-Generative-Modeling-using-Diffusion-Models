import os
import sys
import time
import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore
from tqdm import trange  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "plots/unconditional_generation"
os.makedirs(plot_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 10_000
num_epochs = 500
lr = 1e-3
weight_decay = 1e-4

mlp = MLP(
    input_dim=3,  # x, y, t
    hidden_layers=[256, 256, 256, 256],
    output_dim=2,  # x, y
)
mlp.to(device)
mse_loss = nn.MSELoss()

optimizer = torch.optim.Adam(
    mlp.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)

print("Data loading starts!")
start_time = time.time()
num_steps = 500
dataset = States(num_steps=num_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))
end_time = time.time()
print("Data loaded!")
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

disable_tqdm = not sys.stdout.isatty()
def train_one_epoch():
    avg_loss = 0
    total = dataset.all_data.shape[0]
    indices = torch.randperm(total)
    
    mlp.train()
    for i in range(0, total, batch_size):
        batch_idx = indices[i:i+batch_size]
        
        x_t = dataset.all_data[batch_idx]
        t = dataset.all_steps[batch_idx].unsqueeze(1)
        eps = dataset.eps[batch_idx]
        
        model_input = torch.cat([x_t, t], dim=1)
        
        optimizer.zero_grad()
        eps_pred = mlp(model_input)
        loss = mse_loss(eps_pred, eps)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
    
    avg_loss /= (total // batch_size)
    
    return avg_loss

@torch.no_grad()
def sample(num_samples=2_000):
    mlp.eval()
    
    z = torch.randn(num_samples, 2).to(device)

    for i in np.arange(num_steps-1, -1, -1):
        t_scalar = torch.full(
            (num_samples,), (i + 1) / num_steps, dtype=torch.float32, device=device
        )
        t_bar = 2 * (t_scalar - 0.5)
        t_bar = t_bar.unsqueeze(1)
        
        z_input = torch.cat([z, t_bar], dim=1)
        
        eps_pred = mlp(z_input)
        
        alpha_t = dataset.alpha[i]
        alpha_bar_t = dataset.alpha_bar[i]
        
        if i > 0:
            z_noise = torch.randn_like(z)
            noise_scale = torch.sqrt(1 - alpha_t)
        else:
            z_noise = 0
            noise_scale = 0
        
        z = (
            1 / torch.sqrt(alpha_t) * (
                z - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
            ) + noise_scale * z_noise
        )
    
    z = z.cpu().numpy()
    nll = dataset.calc_nll(z)

    return nll, z

print("Training started!")
train_loss_list = []
nll_list = []
for e in trange(num_epochs):
    train_loss_list.append(train_one_epoch())
    nll, z = sample()
    nll_list.append(nll)
    dataset.show(z, os.path.join(plot_dir, f"latest.png"))
    print(f"Epoch {e+1}/{num_epochs}, Loss: {train_loss_list[-1]:.4f}")
    cosine_scheduler.step()
    if (e + 1) % 50 == 0:
        dataset.show(z, os.path.join(plot_dir, f"epoch_{e+1}.png"))
        dataset.mix_data()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_loss_list)
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")

    axs[1].plot(nll_list)
    axs[1].set_title("Negative Log Likelihood")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("NLL")
    axs[1].set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "train_logs.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

nll, z = sample(5000)
dataset.show(z, os.path.join(plot_dir, "final.png"))
np.save(os.path.join(plot_dir, "uncond_gen_samples.pt"), z)
torch.save(mlp.state_dict(), "denoiser.pt")
