import torch
import torch.nn as nn
import torch.nn.functional as F


# Gaussian MLP as encoder
class gaussian_MLP_encoder(nn.Module):
    """
    Class for Encoder.
    """
    def __init__(self, input_size, n_hidden, n_output, p_zeroed):
        super(gaussian_MLP_encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output * 2)
        self.dropout = nn.Dropout(p_zeroed)
        self.softplus = nn.Softplus()
        self.n_output = n_output

    def forward(self, x):
        out = self.fc1(x)
        out = F.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        gaussian_params = self.fc3(out)

        mean = gaussian_params[:, :self.n_output]
        stddev = 1e-6 + self.softplus(gaussian_params[:, self.n_output:])

        return mean, stddev


# Bernoulli MLP as decoder
class bernoulli_MLP_decoder(nn.Module):
    """
    Class for Decoder.
    """
    def __init__(self, z_size, n_hidden, n_output, p_zeroed):
        super(bernoulli_MLP_decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(p_zeroed)
        self.n_output = n_output

    def forward(self, z):
        out = self.fc1(z)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.elu(out)
        out = self.dropout(out)
        y = self.fc3(out)

        return y


# Gateway
class autoencoder(nn.Module):
    """
    Class autoencoder.
    """
    def __init__(self, dim_img, dim_z, n_hidden, p_zeroed):
        super(autoencoder, self).__init__()
        self.Encoder = gaussian_MLP_encoder(dim_img, n_hidden, dim_z, p_zeroed)
        self.Decoder = bernoulli_MLP_decoder(dim_z, n_hidden, dim_img, p_zeroed)

    def forward(self, x):
        mu, sigma = self.Encoder(x)
        z = mu + sigma * torch.normal(0, 1, mu.size())
        y = self.Decoder(z)

        # KL term의 loss
        KL_divergence = 0.5 * torch.sum(torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma, 2)) - 1, 1)
        KL_divergence = torch.mean(KL_divergence)

        return y, KL_divergence

