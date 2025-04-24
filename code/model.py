import torch.nn as nn  # type: ignore

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[]):
        super().__init__()
        self.act = nn.PReLU()
        layers = []
        if len(hidden_layers) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            for h in hidden_layers:
                layers.append(nn.Linear(input_dim, h))
                layers.append(self.act)
                input_dim = h
            layers.append(nn.Linear(input_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
