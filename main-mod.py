import time
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import torch.optim as optim
# from rmdn import RMDN
from rmdngru import RMDN
import matplotlib.dates as mdates
import tqdm

torch.manual_seed(1234)
LAG = 1

EPOCH = 10
LR = 0.000025
N_COMPONENTS = 2

"""
Data
"""
data = pd.read_csv('yellow-agg-demand-2016-06.csv')
data = data.loc[data['region'] == 1]
data = data[['time_slot', 'demand']]
data = data.set_index('time_slot')

returns = ((np.log(data) - np.log(data.shift(1))) * 100).loc[:, 'demand'].dropna()
# returns = (np.log1p(data)).loc[:, 'demand'].dropna()
print(returns.shape)
returns = returns.loc[~(returns == 0)]

print(returns.shape)
#
train_set = returns.loc["2016-06-01 00:1": "2016-06-23 00:0"]
print(train_set.shape)

test_set = returns.loc['2016-06-23 00:0':]
print(test_set.shape)


"""
Model
"""
rmdn = RMDN(1, 1, 2)

print('creating RMDN')

optimizer = optim.Adam(rmdn.parameters(), lr=LR, amsgrad=True)
hidden_state = rmdn.init_hidden(train_set)

losses = np.zeros(EPOCH)
etas = dict()
means = dict()
variances = dict()
for k in tqdm.tqdm(range(EPOCH)):
    rmdn.zero_grad()
    loss = 0.0
    etas[k] = []
    means[k] = []
    variances[k] = []

    for i in range(LAG, len(train_set)):
        minibatch = train_set.iloc[i - LAG:i]
        labels = train_set.iloc[i]

        out = rmdn(torch.Tensor([minibatch]).reshape(-1, 1), hidden_state)

        nll, reg_nll = rmdn.loss(torch.Tensor([labels]), out)

        reg_nll.backward()
        optimizer.step()
        loss = loss + nll

        (eta, mu, sigma) = out

        # compute the total likelihood
        hidden_state = sigma.detach()

        etas[k].append(eta.detach().numpy())
        means[k].append(mu.detach().numpy())
        variances[k].append(sigma.detach().numpy())
    etas[k] = np.array(etas[k])
    means[k] = np.array(means[k])
    variances[k] = np.array(variances[k])
    losses[k] = loss

print('training finished')

"""
IN SAMPLE TEST
"""
# x = np.array(train_set.values)
n = len(train_set) - LAG
ind = np.argsort(train_set.iloc[:-1].values)
# y = np.take_along_axis(train_set.iloc[:-1].values, ind, axis=0)
# ind_params = ind.reshape(n, 1)

eta = etas[EPOCH - 1].reshape(n, 2)
mean = means[EPOCH - 1].reshape(n, 2)
variance = variances[EPOCH - 1].reshape(n, 2)
var = []
mix_mean = []
for i in range(n):
    mix_mean.append(eta[i].T @ mean[i])
    v = (variance[i] - np.power(mean[i] - mix_mean[-1], 2)) @ eta[i].T
    var.append(v)
mix_mean = np.array(mix_mean)
# is_var = np.array(var)
var = np.array(var)




def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

print("Log Likelihood: ", end="")
print(-losses[-1])
print("AIC: ", end="")
print(2 * get_n_params(rmdn) + 2 * losses[-1])
print("BIC: ", end="")
print(np.log(n) * get_n_params(rmdn) + 2 * losses[-1])
print(train_set[1:])

rmse1 = np.sqrt(np.sum((var.reshape(var.shape[0]) - train_set[1:] ** 2) ** 2) / len(train_set[1:]))
print("RMSE: ", end="")
print(rmse1)
print("NMAE: ", end="")
mae1 = np.sum(np.abs(var - train_set[1:] ** 2)) / len(train_set[1:])
print(mae1)







