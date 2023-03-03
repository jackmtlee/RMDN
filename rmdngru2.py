import math
import numpy as np
from scipy.stats import norm, t, uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, StudentT, Uniform

LOG2PI = math.log(2 * math.pi)


# gru_cell = nn.GRU

# the ANN for weights
class MixingNetwork(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        """
        feedforward neural network
        """

        self.hidden = nn.Linear(in_features=2, out_features=3, bias=True)
        # output layer
        self.output = nn.Linear(in_features=3, out_features=2, bias=True)
        self.activation = nn.Softmax(0)

    def forward(self, x):
        x = self.hidden(x.squeeze(0))
        # activation layer
        # x = torch.tanh(x)     #  question: why tanh is applied on x1, x2 but x0
        x = torch.cat((
            x[0:1],
            torch.tanh(x[1:])
        ), dim=0)

        x = self.output(x)
        x = self.activation(x)
        return x

    def start_pretrain(self):
        self.__hook = []
        for p in self.hidden.parameters():
            self.__hook.append(p.register_hook(MixingNetwork.__pretraining_hook))

        params = []
        for p in self.output.parameters():
            params.append(p)

        self.__hook.append(params[0].register_hook(MixingNetwork.__out_pretraining_hook))

    def stop_pretrain(self):
        if self.__hook is not None:
            for h in self.__hook:
                h.remove()

    @staticmethod
    def __pretraining_hook(grad):
        grad[1:] = 0.0
        return grad

    @staticmethod
    def __out_pretraining_hook(grad):
        grad[:, 1:] = 0.0
        return grad

    def init_weight(self):
        nn.init.zeros_(self.hidden.weight[1:])
        nn.init.zeros_(self.output.weight[1:])
        nn.init.zeros_(self.hidden.bias[1:])
        nn.init.zeros_(self.output.bias[1:])


# the ANN for means
class MeanLevelNetwork(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        """
        feedforward neural network
        config: list containing the spec for each layer
        """
        self.n_components = n_components
        # hidden layer
        self.hidden = nn.Linear(in_features=2, out_features=3, bias=True)

        # output layer
        self.output = nn.Linear(in_features=3, out_features=self.n_components, bias=True)

        """
        print("HIDDEN MEAN")
        for param in self.hidden.parameters():
            print(param, end=" ")

        print("OUTPUT MEAN")
        for param in self.output.parameters():
            print(param, end=" ")
        """

    def forward(self, x):
        x = self.hidden(x.squeeze(0))

        # activation layer
        # x = torch.tanh(x)
        x = torch.cat((
            x[0:1],
            torch.tanh(x[1:])
        ), dim=0)

        x = self.output(x)
        return x

    def start_pretrain(self):
        self.__hook = []
        for p in self.hidden.parameters():
            self.__hook.append(p.register_hook(MeanLevelNetwork.__pretraining_hook))

        params = []
        for p in self.output.parameters():
            params.append(p)

        self.__hook.append(params[0].register_hook(MeanLevelNetwork.__out_pretraining_hook))

    def stop_pretrain(self):
        if self.__hook is not None:
            for h in self.__hook:
                h.remove()

    @staticmethod
    def __pretraining_hook(grad):
        grad[1:] = 0.0
        return grad

    @staticmethod
    def __out_pretraining_hook(grad):
        grad[:, 1:] = 0.0
        return grad

    def init_weight(self):
        nn.init.zeros_(self.hidden.weight[1:])
        nn.init.zeros_(self.output.weight[1:])
        nn.init.zeros_(self.hidden.bias[1:])
        nn.init.zeros_(self.output.bias[1:])


# the ANN for variance
class VarianceRecurrentNetwork(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        """
        Recurrent neural network
        """
        self.n_components = n_components
        self.hidden_size = 2
        # hidden layer
        self.hidden = nn.Linear(in_features=3, out_features=4, bias=True)
        # output layer
        self.output = nn.Linear(in_features=4, out_features=self.n_components, bias=True)
        self.__hook = None

    def forward(self, x, hidden_state):
        x = torch.cat((x, hidden_state.squeeze(0)), 0)
        x = self.hidden(x.float())
        # print(x[:,0])
        # x = torch.tanh(x)
        # heterogenous activation layer
        x = torch.cat((
            x[0:1],
            torch.tanh(x[1:2]),
            x[2:3],
            torch.tanh(x[3:])
        ), 0)
        x = self.output(x)
        return F.elu(x) + 1 + 1e-15

    def init_hidden(self, data):
        return torch.Tensor([np.var(np.array(data))]).repeat(1, self.n_components)
        # return torch.Tensor([np.var(data)]).repeat(1, self.n_components)

    def start_pretrain(self):
        self.__hook = []
        for p in self.hidden.parameters():
            self.__hook.append(p.register_hook(VarianceRecurrentNetwork.__pretraining_hook))

        params = []
        for p in self.output.parameters():
            params.append(p)
        self.__hook.append(params[0].register_hook(VarianceRecurrentNetwork.__out_pretraining_hook))

    def stop_pretrain(self):
        if self.__hook is not None:
            for h in self.__hook:
                h.remove()

    @staticmethod
    def __pretraining_hook(grad):
        grad[1] = 0.0
        grad[3] = 0.0
        # print("variance:"+str(grad))
        return grad

    @staticmethod
    def __out_pretraining_hook(grad):
        grad[:, 1] = 0.0
        grad[:, 3] = 0.0
        return grad

    def init_weight(self):
        nn.init.zeros_(self.hidden.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.ones_(self.hidden.bias)
        nn.init.ones_(self.output.bias)


#
class RMDN(nn.Module):
    def __init__(self, p, q, n_components):
        super().__init__()
        self.p = p
        self.q = q
        self.n_components = n_components
        # nn.Sequential(*[nn.Linear(in_features=1, out_features=3, bias=True), nn.Tanh(), nn.Linear(in_features=3, out_features=self.n_components, bias=True), nn.Softmax(1)])
        self.mixing_network = MixingNetwork(self.n_components)
        # Mean Level Network
        # nn.Sequential(*[nn.Linear(in_features=1, out_features=3, bias=True), nn.Tanh(), nn.Linear(in_features=3, out_features=self.n_components, bias=True)])
        self.mean_level_network = MeanLevelNetwork(self.n_components)
        # Variance Recurrence Network
        self.variance_recurrent_network = VarianceRecurrentNetwork(self.n_components)

    def forward(self, x, hidden_state):
        eta = self.mixing_network(x)
        mu = self.mean_level_network(x)

        error = torch.pow(x[:1] - torch.matmul(eta, mu.reshape(-1, 1)), 2)

        sigma = self.variance_recurrent_network(error, hidden_state)

        return (eta, mu, sigma)

    def loss(self, target, args):
        # dist = Normal(args[1], torch.sqrt(args[2]))

        loglik = torch.log(args[0]) - torch.log(torch.sqrt(args[2])) - 0.5 * LOG2PI - 0.5 * torch.pow(
            (target - args[1]) / torch.sqrt(args[2]), 2)

        nll = -torch.logsumexp(loglik, dim=0)
        # l2_mix = torch.sum(torch.pow(torch.nn.utils.parameters_to_vector(self.mixing_network.hidden.parameters()), 2)/2,axis=0)
        # l2_mean = torch.sum(torch.pow(torch.nn.utils.parameters_to_vector(self.mean_level_network.hidden.parameters()), 2)/2,axis=0)
        # l2_var = torch.sum(torch.pow(torch.nn.utils.parameters_to_vector(self.variance_recurrent_network.hidden.parameters()), 2)/2,axis=0)
        # l2 = self.alpha*l2_var
        # to be able to keep track of the loss without the weight regularization
        return nll, nll  # + l2

    def pretrain_loss(self, target, args):
        mix_var = (args[2] + torch.pow(args[1] - torch.transpose(args[0], 0, -1) @ args[1], 2)) @ args[0]
        mse = torch.sqrt(torch.pow(mix_var - torch.pow(target.unsqueeze(1), 2), 2))
        return mse

    def update_alpha(self):

        # params = torch.nn.utils.parameters_to_vector(self.variance_recurrent_network.hidden.parameters()).detach()
        # mix_params = torch.nn.utils.parameters_to_vector(self.mixing_network.hidden.parameters()).detach()
        # mean_params = torch.nn.utils.parameters_to_vector(self.mean_level_network.hidden.parameters()).detach()
        var_params = torch.nn.utils.parameters_to_vector(self.variance_recurrent_network.hidden.parameters()).detach()
        # l2_mix = torch.sum(torch.pow(mix_params, 2)/2,axis=0)
        # l2_mean = torch.sum(torch.pow(mean_params, 2)/2,axis=0)
        l2_var = torch.sum(torch.pow(var_params, 2) / 2, axis=0)
        # n_params = len(mix_params)+len(mean_params)+len(var_params)
        self.alpha = len(var_params) / (2 * l2_var)

    def sample(self, x, hidden_state, horizon=1, n=1000):
        eta, mu, sigma = self.forward(x, hidden_state)
        cat = Categorical(eta).sample([n]).unsqueeze(1)

        dist = Normal(mu, torch.sqrt(sigma))

        samples = torch.gather(dist.sample([n]), 1, cat)
        if horizon > 1:
            etas = eta.detach().numpy()
            mus = mu.detach().numpy()
            sigmas = sigma.detach().numpy()
            for t in range(1, horizon):

                if etas.shape[0] > 2:
                    prob = np.zeros(n)
                    samples = samples.detach().numpy()

                    for i in range(n):
                        prob[i] = RMDN.pdf(samples[i], etas[i], mus[i], sigmas[i])

                    idx = torch.argsort(torch.Tensor(prob), descending=True)[:100]
                    etas = etas[idx]
                    mus = mus[idx]
                    sigmas = sigmas[idx]
                else:
                    prob = RMDN.pdf(samples.detach().numpy(), etas, mus, sigmas)
                    idx = torch.argsort(torch.Tensor(prob), descending=True)[:100]

                    sigmas = np.array([sigmas] * 100)
                sub_sample = samples[idx]
                # print(sigmas.shape)
                samples = []
                etas_temp = []
                mus_temp = []
                sigmas_temp = []

                for i in range(len(sub_sample)):
                    eta, mu, sigma = self.forward(torch.Tensor(sub_sample[i].reshape(1, 1)).reshape(-1, 1),
                                                  torch.Tensor(sigmas[i]))
                    cat = Categorical(eta).sample([int(n / 100)]).unsqueeze(1)

                    dist = Normal(mu, torch.sqrt(sigma))
                    s = torch.gather(dist.sample([int(n / 100)]), 1, cat)
                    samples.append(s)
                    etas_temp.append(eta.repeat(int(n / 100), 1).detach().numpy())

                    mus_temp.append(mu.repeat(int(n / 100), 1).detach().numpy())
                    sigmas_temp.append(sigma.repeat(int(n / 100), 1).detach().numpy())
                samples = torch.cat(samples)
                etas = np.concatenate(etas_temp, axis=0)
                mus = np.concatenate(mus_temp, axis=0)
                sigmas = np.concatenate(sigmas_temp, axis=0)

        return samples, sigma.detach()

    def init_hidden(self, data):
        return self.variance_recurrent_network.init_hidden(data)

    def start_pretrain(self):
        self.mixing_network.start_pretrain()
        self.mean_level_network.start_pretrain()
        self.variance_recurrent_network.start_pretrain()

    def stop_pretrain(self):
        self.mixing_network.stop_pretrain()
        self.mean_level_network.stop_pretrain()
        self.variance_recurrent_network.stop_pretrain()

    def init_weight(self):
        self.mixing_network.init_weight()
        self.mean_level_network.init_weight()
        self.variance_recurrent_network.init_weight()
        self.update_alpha()

    @staticmethod
    def mean(eta, mu):
        return eta.T @ mu

    @staticmethod
    def variance(eta, mu, sigma):
        mix_mean = RMDN.mean(eta, mu)
        v = 0.0
        for i in range(len(eta)):
            v += (sigma[i] + np.power(mu[i] - mix_mean, 2)) * eta[i]
        return v

    @staticmethod
    def skewness(eta, mu, sigma):
        sigma_m = np.sqrt(RMDN.variance(eta, mu, sigma))
        mix_mean = RMDN.mean(eta, mu)
        s = 0.0

        for i in range(len(eta)):
            s += eta[i] * (3 * sigma[i] * (mu[i] - mix_mean) + np.power(mu[i] - mix_mean, 3))
        return 1 / np.power(sigma_m, 3) * s

    @staticmethod
    def kurtosis(eta, mu, sigma):
        sigma_m = np.sqrt(RMDN.variance(eta, mu, sigma))
        mix_mean = RMDN.mean(eta, mu)
        s = 0.0

        for i in range(len(eta)):
            s += eta[i] * (3 * np.power(sigma[i], 2) + 6 * sigma[i] * np.power(mu[i] - mix_mean, 2) + np.power(
                mu[i] - mix_mean, 4))
        return 1 / np.power(sigma_m, 4) * s

    @staticmethod
    def cdf(x, *args):
        return norm.cdf(x, args[1], args[2])

    @staticmethod
    def pdf(x, *args):
        x = np.hstack([x.copy(), x.copy()])
        return args[0] @ norm.pdf(x, args[1], args[2]).T
