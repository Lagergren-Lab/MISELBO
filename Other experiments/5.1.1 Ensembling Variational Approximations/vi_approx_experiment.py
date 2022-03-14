#%% Imports and base seed
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.distributions import MultivariateNormal
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
from target_distribution import TargetDistribution

#%% Training
td_name = 'U_2'
td = TargetDistribution(td_name)
z = np.linspace(-10, 10, 1000)
z = np.array((z, z)).reshape(-1, 2)
z = torch.Tensor(z)
S = 2 if td_name == 'U_1' else 3

ensemble = []
cov_params = []
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

for s in range(0, S):
    mu_init = [6 * s - 3, 0] if td == 'U_1' else [3 * s - 3, 0]
    mu = nn.Parameter(Tensor(mu_init))
    log_std = nn.Parameter(torch.log(torch.ones(2) * 0.8))
    optimiser = torch.optim.Adam([mu], lr=0.001)
    num_batches = 10000
    batch_size = 1000

    for batch_num in range(num_batches):
        # Get batch from N(0,I).
        std = torch.exp(log_std)
        batch = std * torch.zeros(size=(batch_size, 2)).normal_(mean=0, std=1) + mu
        log_q = MultivariateNormal(mu, std * torch.eye(2)).log_prob(batch).mean()
        log_p = (-td(batch)).mean()
        loss = log_q - log_p
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch_num % 1000 == 0:
            # print(mu)
            print(loss)

    ensemble.append([mu, log_std])

#%% Plot settings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

sns.set()

sns.set_theme(style="ticks")

legend_labels = [r"$q_{\phi_{1}}(z)$",
                 r"$q_{\phi_{2}}(z)$",
                 r"$q_{\phi_{3}}(z)$",
                 r"$q_{\phi_{4}}(z)$",
                 r"$q_{\phi_{5}}(z)$"]

params = {'legend.fontsize': 'large',
          # 'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'medium',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)
matplotlib.rcParams['figure.dpi'] = 100

#%% Visualize p and q's
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

n = 500
xlim = 4
ylim = 4
cmaps = ['winter', 'gray', 'summer']
colors = ['blue', 'gray', 'green']
x = torch.linspace(-xlim, xlim, n)
xx, yy = torch.meshgrid(x, x)
zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()
zk = zz
patches = []
fig = plt.figure(figsize=[7, 7])
ax = fig.add_subplot(111)
ax.set_xlim(-xlim, xlim)
ax.set_ylim(-ylim, ylim)
ax.set_aspect(1)
p = np.exp(-td(zz))
ax.pcolormesh(
    zk[:, 0].detach().data.reshape(n, n),
    zk[:, 1].detach().data.reshape(n, n),
    p.detach().data.reshape(n, n),
    cmap='Reds',
    rasterized=True,
)
for s in range(S):
    mu = ensemble[s][0]
    log_std = ensemble[s][1]
    print(f"mu {s}: {mu}, log_std {s}: {log_std}")
    std = torch.exp(log_std.detach())
    base_log_prob = MultivariateNormal(mu, std * torch.eye(2)).log_prob(zz)
    final_log_prob = base_log_prob
    qk = torch.exp(final_log_prob)
    p = torch.exp(-td(zz))

    cnt = plt.contourf(xx, yy, qk.detach().numpy().reshape(n, n),
                       norm=LogNorm(vmin=0.1, vmax=100.0),
                       levels=np.logspace(-1.5, 1, 10),
                       cmap=cmaps[s],
                       alpha=0.5)
    patches.append(mpatches.Patch(color=colors[s], label=legend_labels[s]))

plt.xlabel(r"$z_1$")
plt.ylabel(r"$z_2$")
plt.legend(loc='upper right', handles=patches)
plt.show()

#%% Evaluation
import scipy.stats as sp_stats
from scipy.special import logsumexp


def compute_mis_kl(logP, logQ):
    # assuming uniform weights
    S, J = logP.shape
    logP = logP.squeeze()
    # logsumexp over the S mixture components in the denominator of the log ratio
    logQ_mixtures = logsumexp(logQ, axis=-1) - np.log(S)
    # logsumexp over the L importance samples, results in S log ratios
    log_ratios = logQ_mixtures - logP
    # sum over both batch and S weights
    mis_kl = 1 / J * np.sum(np.sum(1 / S * log_ratios, axis=0))
    return mis_kl


def compute_avg_kl(logP, logQ):
    # assuming uniform weights
    S, J = logP.shape
    quotient = logQ - logP
    # logsumexp over the S mixture components in the denominator of the log ratio
    avg_kl = 1 / J * np.sum(1 / S * np.sum(quotient, axis=0))
    return avg_kl


def get_jsd(ensemble):
    q = [MultivariateNormal(param[0], torch.exp(param[1]) * torch.eye(2)) for param in ensemble]
    jsd = np.log(S)
    for q_s in q:
        z = q_s.sample((10000, 1))
        log_q_norm_temp = torch.stack([q_j.log_prob(z) for q_j in q], dim=0)
        log_q_norm = logsumexp(log_q_norm_temp.detach().numpy(), axis=0)
        log_q = q_s.log_prob(z)
        jsd += 1 / S * np.mean(log_q.detach().numpy() - log_q_norm)

    print(f"JSD: {jsd}")


def norm_constant_p(td, n=10000, xlim=4):
    x = torch.linspace(-xlim, xlim, n)
    xx, yy = torch.meshgrid(x, x)
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    u = td(zz)
    p_u = torch.exp(-u).numpy()
    c = np.sum(p_u)
    p_norm = p_u / c
    return c


def compare_kls(Q, p, c=1, J=10000, GMM=False):
    S = len(Q)
    logQ = np.zeros((S, J, S))
    logQ_avg = np.zeros((S, J))
    logP = np.zeros((S, J))

    for s in range(0, S):
        # sample from each q
        if GMM:
            q_s = Q[s]
        else:
            mu_s = Q[s][0].detach().numpy()
            cov_s = np.exp(Q[s][1].detach().numpy()) * np.eye(2)
            q_s = sp_stats.multivariate_normal(mu_s, cov_s)
        z_s = q_s.rvs(size=J)

        logQ_avg[s, :] = np.log(q_s.pdf(z_s))
        # evaluate on all q
        for k in range(0, S):
            if GMM:
                q_k = Q[k]
            else:
                mu_k = Q[k][0].detach().numpy()
                cov_k = np.exp(Q[k][1].detach().numpy()) * np.eye(2)
                q_k = sp_stats.multivariate_normal(mu_k, cov_k)
            logQ[s, :, k] = np.log(q_k.pdf(z_s))

        samps_s_torch = torch.from_numpy(z_s)
        if J > 1:
            samps_s_torch = torch.stack((samps_s_torch[:, 0].flatten(), samps_s_torch[:, 1].flatten()),
                                        dim=-1).squeeze()
        else:
            samps_s_torch = torch.stack((samps_s_torch[0].flatten(), samps_s_torch[1].flatten()), dim=-1)

        logP[s] = -p(samps_s_torch) - np.log(c)

    mis_kl = compute_mis_kl(logP, logQ)
    avg_kl = compute_avg_kl(logP, logQ_avg)

    print(f"mis_kl: {mis_kl}")
    print(f"avg_kl: {avg_kl}")


# c = norm_constant_p(td)
torch.manual_seed(1)
np.random.seed(1)
compare_kls(ensemble, td, c=1)
get_jsd(ensemble)
