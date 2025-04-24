import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "plots/conditional_diffusion"
os.makedirs(plot_dir, exist_ok=True)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10_000
num_epochs = 500
num_steps = 500

# Instantiate the classifier and load its trained weights
classifier = MLP(
    input_dim=3,
    hidden_layers=[100, 200, 500],
    output_dim=5  # 5 US states
)
classifier.load_state_dict(torch.load("classifier.pt", map_location=device, weights_only=True))
classifier = classifier.to(device)
classifier.eval()

logsoftmax = nn.LogSoftmax(dim=1)

# Instantiate the denoiser and load its trained weights
mlp = MLP(
    input_dim=3,
    hidden_layers=[256, 256, 256, 256],
    output_dim=2
)
mlp.load_state_dict(torch.load("denoiser.pt", map_location=device, weights_only=True))
mlp = mlp.to(device)
mlp.eval()

print("Data loading starts!")
dataset = States(num_steps=num_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))
print("Data loaded!")

def sample(label, num_samples=1000):
    mlp.eval()
    z = torch.randn(num_samples, 2).to(device).detach().requires_grad_()
    
    for i in np.arange(num_steps-1, 0, -1):
        t_scaler = torch.full((num_samples,), i / num_steps, device=device)
        t_bar = 2 * (t_scaler - 0.5)
        t_bar = t_bar.unsqueeze(1).detach()
        
        z_input = torch.cat([z, t_bar], dim=1)
        
        eps_pred = mlp(z_input)
        
        logits = classifier(z_input)
        log_probs = logsoftmax(logits)
        out_label = log_probs[:, label]
        
        grad = torch.autograd.grad(
            outputs=out_label,
            inputs=z,
            grad_outputs=torch.ones_like(out_label),
            retain_graph=False,
            create_graph=False,
        )[0]
        
        alpha_bar_t = dataset.alpha_bar[i]
        eps_hat = eps_pred - torch.sqrt(1 - alpha_bar_t) * grad
        alpha_bar_t_minus_1 = dataset.alpha_bar[i - 1]
        
        z_0 = (z - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
        
        z = (
            torch.sqrt(alpha_bar_t_minus_1) * z_0 + 
            torch.sqrt(1 - alpha_bar_t_minus_1) * eps_hat
        ).detach().requires_grad_()
 
    z = z.detach().cpu().numpy()
    nll = dataset.calc_nll(z)

    return nll, z

print("Sampling starts!")
for label in range(5):
    full_z = []
    for i in range(5):
        nll, z = sample(label, num_samples=5000)
        full_z.append(z)
    full_z = np.concatenate(full_z, axis=0)
    nll = dataset.calc_nll(full_z)
    print(f"Label {label}, NLL: {nll:.4f}")
    dataset.show(z, os.path.join(plot_dir, f"label_{label}.png"))
    np.save(os.path.join(plot_dir, f"cond_gen_samples_{label}.npy"), full_z)
