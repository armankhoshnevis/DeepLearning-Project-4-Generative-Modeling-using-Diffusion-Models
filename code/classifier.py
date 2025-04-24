import os
import numpy as np
import torch  # type: ignore
from torch import nn  # type: ignore
from models import MLP  # type: ignore
from data import States  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore

plot_dir = "plots/conditional_generation"
os.makedirs(plot_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataset and dataloader (The same number of steps in the generation code)
print("Data loading starts!")
num_steps = 500
batch_size = 20_000
dataset = States(num_steps=num_steps)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Data loaded!")

# Instantiate classifier
classifier = MLP(
    input_dim=3,
    hidden_layers=[100, 200, 500],
    output_dim=5
).to(device)

# Set the training parameters
lr = 1e-3
weight_decay = 1e-5
num_epochs = 10

# Create loss function, optimizer, and scheduler
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

label_to_states = {0: "Michigan",
                   1: "Idaho",
                   2: "Ohio",
                   3: "Oklahoma",
                   4: "Wisconsin"}
colors = ["red", "blue", "green", "orange", "purple"]
cmap = ListedColormap(colors)

# Train the classifier
print("Training starts!")
loss_list = []
classifier.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for x_t, t, _, _, y in tqdm(loader, disable=False):
        x_t, t, y = x_t.to(device), t.to(device), y.to(device)
        t = t.unsqueeze(1)
        inp = torch.cat([x_t, t], dim=1)
        
        logits = classifier(inp)
        loss = ce_loss(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    loss_list.append(avg_loss)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    if (epoch + 1) % 1 == 0:
        plt.plot(loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(plot_dir, f"loss_epoch_{epoch+1}.png"))
        plt.close()

torch.save(classifier.state_dict(), "classifier.pt")

classifier.eval()
X_clean = dataset.data
labels = dataset.labels
t_clean = torch.full((X_clean.shape[0], 1), -1.0).to(device)
inp = torch.cat([X_clean, t_clean], dim=1).to(device)
with torch.no_grad():
    all_preds = classifier(inp).argmax(1).detach().cpu()

X_clean = X_clean.cpu()
labels = labels.cpu()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.set_title("Classifier Predictions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
im = ax.scatter(X_clean[:, 0], X_clean[:, 1], s=1,
           c=all_preds, cmap=cmap)

X_max1 = X_clean.max(0).values[0].item()
X_max2 = X_clean.max(0).values[1].item()
X_min1 = X_clean.min(0).values[0].item()
X_min2 = X_clean.min(0).values[1].item()

X, Y = torch.meshgrid(
    torch.linspace(X_min1, X_max1, 100),
    torch.linspace(X_min2, X_max2, 100)
)
X = X.flatten()
Y = Y.flatten()

with torch.no_grad():
    grid_X = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), t_clean], dim=1)
    grid_X = grid_X.to(device)
    grid_preds = classifier(grid_X)
    grid_preds = grid_preds.argmax(1).detach().cpu()
    grid_preds = grid_preds.reshape(100, 100)

X = X.reshape(100, 100)
Y = Y.reshape(100, 100)

ax.contourf(X, Y, grid_preds, alpha=0.3,
            cmap=cmap, levels=5)
cbar = fig.colorbar(im, label="States")
cbar.set_ticks(np.arange(5)*0.8 + 0.4, labels=list(label_to_states.values()))
plt.savefig(os.path.join(plot_dir, "classifier_predictions.png"))
plt.close()
