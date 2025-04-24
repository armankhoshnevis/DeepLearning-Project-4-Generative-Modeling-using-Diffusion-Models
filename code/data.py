from scipy.stats import gaussian_kde  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
plt.switch_backend("agg") # this is to avoid a Matplotlib issue.

class States(Dataset):
    def __init__(self, num_steps=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        states_data = torch.load("states.pt", weights_only=True)
        self.data = states_data["data"].float().to(self.device)
        self.labels = states_data["labels"]
        self.n_points = self.data.shape[0]
        self.num_steps = num_steps
        self.steps = torch.linspace(-1, 1, self.num_steps).to(self.device)
        
        self.beta = torch.linspace(1e-4, 0.02, self.num_steps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.mix_data()
    
    def refresh_eps(self):
        self.eps = torch.randn(self.__len__(), self.data.shape[1])

    def mix_data(self):
        self.all_data = []
        self.all_labels = []
        self.all_times = []
        self.refresh_eps()
        self.eps = self.eps.to(self.device)
        total_samples = len(self)

        for i in range(total_samples):
            data_idx = i % self.n_points
            step = i // self.n_points
            x = self.data[data_idx].to(self.device)
            t = self.steps[step].to(self.device)
            e = self.eps[i].to(self.device)
            x_ = torch.sqrt(self.alpha_bar[step]) * x + torch.sqrt(1 - self.alpha_bar[step]) * e
            if self.labels is None:
                y = 0
            else:
                y = self.labels[data_idx]

            self.all_data.append(x_)
            self.all_times.append(t)
            self.all_labels.append(y)

        self.all_data = torch.stack(self.all_data).to(self.device)
        self.all_labels = torch.tensor(self.all_labels).to(self.device)
        self.all_steps = torch.tensor(self.all_times).to(self.device)
        self.eps = self.eps.to(self.device)

    def __len__(self):
        return self.n_points * self.num_steps

    def __getitem__(self, idx):
        data_idx = idx % self.n_points
        step = idx // self.n_points
        x = self.data[data_idx].to(self.device)
        t = self.steps[step].to(self.device)
        eps = torch.randn_like(x).to(self.device)
        x_ = torch.sqrt(self.alpha_bar[step]) * x + torch.sqrt(1 - self.alpha_bar[step]) * eps
        if self.labels is None:
            y = 0
        else:
            y = self.labels[data_idx]

        return x_, t, eps, x, y

    def show(self, samples=None, save_to=None):
        if samples is None:
            samples = self.data

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.axis('equal')
        if save_to is not None:
            plt.savefig(save_to)
        plt.close()
        plt.clf()

    def calc_nll(self, generated):
        data_ = self.data.cpu().numpy()

        kde = gaussian_kde(data_.T)
        nll = -kde.logpdf(generated.T)

        return nll.mean()
