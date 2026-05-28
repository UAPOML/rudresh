import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

mu = np.array([0.15, 0.08, 0.05])

Sigma = np.array([
    [0.0625,   0.012,    0.001],
    [0.012,    0.0144,   0.00096],
    [0.001,    0.00096,  0.0016]
])

rf = 0.04

n_portfolios = 20000

weights = np.random.dirichlet(np.ones(3), size=n_portfolios)

portfolio_returns = weights @ mu

portfolio_variances = np.sum((weights @ Sigma) * weights, axis=1)

portfolio_stds = np.sqrt(portfolio_variances)

sharpe_ratios = (portfolio_returns - rf) / portfolio_stds

max_index = np.argmax(sharpe_ratios)

max_return = portfolio_returns[max_index]

max_std = portfolio_stds[max_index]

max_weights = weights[max_index]

sigma1 = 0.20
sigma2 = 0.10

w1 = 0.6
w2 = 0.4

rho = np.linspace(-1, 1, 200)

sigma_p = np.sqrt(
    (w1**2)*(sigma1**2)
    + (w2**2)*(sigma2**2)
    + 2*w1*w2*rho*sigma1*sigma2
)

weighted_avg_risk = w1*sigma1 + w2*sigma2

fig, ax = plt.subplots(1, 2, figsize=(14,6))

scatter = ax[0].scatter(
    portfolio_stds,
    portfolio_returns,
    c=sharpe_ratios,
    cmap='viridis',
    s=10
)

ax[0].scatter(
    max_std,
    max_return,
    color='gold',
    marker='*',
    s=250,
    edgecolor='black',
    label='Max Sharpe'
)

individual_stds = np.sqrt(np.diag(Sigma))

asset_labels = ['Asset 1', 'Asset 2', 'Asset 3']

ax[0].scatter(
    individual_stds,
    mu,
    color='red',
    s=120
)

for i in range(3):
    ax[0].annotate(
        asset_labels[i],
        (individual_stds[i], mu[i]),
        textcoords="offset points",
        xytext=(10,5)
    )

ax[0].set_title('Efficient Frontier')

ax[0].set_xlabel('Portfolio Risk')

ax[0].set_ylabel('Expected Return')

ax[0].xaxis.set_major_formatter(PercentFormatter(1))

ax[0].yaxis.set_major_formatter(PercentFormatter(1))

ax[0].grid(True)

ax[0].legend()

cbar = fig.colorbar(scatter, ax=ax[0])

cbar.set_label('Sharpe Ratio')

ax[1].plot(rho, sigma_p)

ax[1].axhline(
    weighted_avg_risk,
    color='red',
    linestyle='--',
    label='Weighted Avg. Risk'
)

ax[1].fill_between(
    rho,
    sigma_p,
    weighted_avg_risk,
    where=(sigma_p < weighted_avg_risk),
    color='lightgreen',
    alpha=0.5
)

ax[1].set_title('Correlation Sensitivity')

ax[1].set_xlabel('Correlation')

ax[1].set_ylabel('Portfolio Risk')

ax[1].yaxis.set_major_formatter(PercentFormatter(1))

ax[1].grid(True)

ax[1].legend()

fig.suptitle('Portfolio Theory --- Week 1 Visualisations')

plt.tight_layout()

plt.savefig('week1-plots.png', dpi=150)

plt.show()