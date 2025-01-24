import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLayer, self).__init__()

        self.mu = nn.Parameter(
            torch.zeros(in_features, out_features).uniform_(-0.6, 0.6)
        )
        self.rho = nn.Parameter(torch.zeros(in_features, out_features) - 3)

    def forward(self, x):
        epsilon = torch.normal(mean=0.0, std=1.0, size=self.mu.shape)
        w = self.mu + F.softplus(self.rho) * epsilon
        return torch.matmul(x, w)

    # total kl loss for the weights in this layer
    def compute_layer_kl_loss(self):
        layer_kl_loss = torch.sum(self._kl_loss(self.mu, self.rho))

        return layer_kl_loss

    # kl loss between one weight's posterior and unit Gaussian prior (closed form complexity cost)
    def _kl_loss(self, temp_mu, temp_rho):
        sigma_squared = F.softplus(temp_rho) ** 2

        return -0.5 * (1 + torch.log(sigma_squared) - temp_mu**2 - sigma_squared)


class BayesianNN(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, layer_width):
        super(BayesianNN, self).__init__()

        layers = []
        layers.append(BayesianLayer(input_size, layer_width))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(BayesianLayer(layer_width, layer_width))
            layers.append(nn.ReLU())

        layers.append(BayesianLayer(layer_width, output_size))
        layers.append(nn.LogSoftmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_total_kl_loss(self):
        total_kl_loss = 0

        for i in self.children():
            for j in i.children():
                if isinstance(j, BayesianLayer):
                    total_kl_loss += j.compute_layer_kl_loss()

        return total_kl_loss
