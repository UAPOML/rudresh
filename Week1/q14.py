import pandas as pd
import numpy as np

np.random.seed(0)

dates = pd.date_range('2023-01-02', periods=52, freq='W-MON')

mu_weekly = np.array([0.003, 0.002, 0.001, 0.0015])

sig_weekly = np.array([0.04, 0.03, 0.02, 0.025])

returns_sim = np.random.normal(mu_weekly, sig_weekly, (52,4))

prices_sim = 100 * np.cumprod(1 + returns_sim, axis=0)

df = pd.DataFrame(
    prices_sim,
    index=dates,
    columns=['AAPL','MSFT','GOOGL','AMZN']
)

returns_df = df.pct_change().dropna()

correlation_matrix = returns_df.corr()

print("Correlation Matrix:\n")
print(correlation_matrix)

corr_no_diag = correlation_matrix.where(
    ~np.eye(correlation_matrix.shape[0], dtype=bool)
)

lowest_pair = corr_no_diag.stack().idxmin()

lowest_corr = corr_no_diag.stack().min()

print("\nLowest Correlation Pair:")
print(lowest_pair)

print("\nLowest Correlation Value:")
print(lowest_corr)

weights = pd.Series(
    [0.25,0.25,0.25,0.25],
    index=['AAPL','MSFT','GOOGL','AMZN']
)

portfolio_returns = returns_df.dot(weights)

print("\nPortfolio Return Series:\n")
print(portfolio_returns.head())

monthly_returns = portfolio_returns.resample('ME').apply(
    lambda x: (1 + x).prod() - 1
)

print("\nMonthly Portfolio Returns:\n")
print(monthly_returns)

monthly_mean = monthly_returns.mean()

monthly_std = monthly_returns.std()

print("\nMean Monthly Return:")
print(monthly_mean)

print("\nMonthly Standard Deviation:")
print(monthly_std)