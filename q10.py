import numpy as np


prices = np.array([
    [100,108,103,115,110,119,125,121,130,127,135,140], # Stock A
    [200,195,210,205,220,215,225,230,222,235,240,238]  # Stock B
])



returns = (prices[:,1:] - prices[:,:-1]) / prices[:,:-1]

print("Monthly Returns:\n")
print(returns)

print("\nShape of returns matrix:")
print(returns.shape)


monthly_mean = np.mean(returns, axis=1)

annual_mean = monthly_mean * 12

print("\nAnnualised Mean Returns:\n")
print(annual_mean)



monthly_std = np.std(returns, axis=1, ddof=1)

annual_std = monthly_std * np.sqrt(12)

print("\nAnnualised Standard Deviations:\n")
print(annual_std)



cov_matrix = np.cov(returns)

print("\nCovariance Matrix:\n")
print(cov_matrix)



corr_matrix = np.corrcoef(returns)

print("\nCorrelation Matrix:\n")
print(corr_matrix)


rho = corr_matrix[0,1]

sigma_A = np.std(returns[0], ddof=1)

sigma_B = np.std(returns[1], ddof=1)

cov_AB = cov_matrix[0,1]

check = rho * sigma_A * sigma_B

print("\nCovariance from covariance matrix:")
print(cov_AB)

print("\nrho * sigma_A * sigma_B:")
print(check)