import numpy as np



mu = np.array([0.15, 0.08, 0.05])

Sigma = np.array([
    [0.0625,   0.012,    0.001],
    [0.012,    0.0144,   0.00096],
    [0.001,    0.00096,  0.0016]
])


w = np.array([1/3, 1/3, 1/3])


portfolio_return = w @ mu


portfolio_variance = w @ Sigma @ w


portfolio_std = np.sqrt(portfolio_variance)

print("Equal Weight Portfolio Return:")
print(portfolio_return)

print("\nEqual Weight Portfolio Variance:")
print(portfolio_variance)

print("\nEqual Weight Portfolio Standard Deviation:")
print(portfolio_std)



weights = np.random.dirichlet(np.ones(3), size=10000)



print("\nShape of random weights matrix:")
print(weights.shape)



portfolio_returns = weights @ mu

print("\nShape of portfolio returns array:")
print(portfolio_returns.shape)


portfolio_variances = np.sum((weights @ Sigma) * weights, axis=1)


portfolio_stds = np.sqrt(portfolio_variances)

print("\nShape of portfolio std array:")
print(portfolio_stds.shape)



risk_free_rate = 0.04

sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_stds

print("\nShape of Sharpe ratio array:")
print(sharpe_ratios.shape)



max_index = np.argmax(sharpe_ratios)

max_sharpe = sharpe_ratios[max_index]

best_weights = weights[max_index]

best_return = portfolio_returns[max_index]

best_std = portfolio_stds[max_index]



print("\nMaximum Sharpe Ratio:")
print(max_sharpe)

print("\nBest Portfolio Weights:")
print(best_weights)

print("\nBest Portfolio Return:")
print(best_return)

print("\nBest Portfolio Risk:")
print(best_std)