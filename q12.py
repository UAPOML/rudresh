import numpy as np

mu1 = 0.12
sigma1 = 0.20

mu2 = 0.06
sigma2 = 0.10

w1 = 0.6
w2 = 1 - w1

rho = np.linspace(-1, 1, 200)

portfolio_variance = (
    (w1**2) * (sigma1**2)
    + (w2**2) * (sigma2**2)
    + 2 * w1 * w2 * rho * sigma1 * sigma2
)

portfolio_std = np.sqrt(portfolio_variance)

print(portfolio_std)

print(portfolio_std.shape)

min_index = np.argmin(portfolio_std)

min_rho = rho[min_index]

min_sigma = portfolio_std[min_index]

print(min_rho)

print(min_sigma)

dvariance_drho = 2 * w1 * w2 * sigma1 * sigma2

print(dvariance_drho)