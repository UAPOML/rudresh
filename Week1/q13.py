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

print("Price DataFrame:\n")
print(df.head())

returns_df = df.pct_change().dropna()

print("\nWeekly Returns:\n")
print(returns_df.head(3))

print("\nShape of Returns DataFrame:")
print(returns_df.shape)

description = returns_df.describe()

print("\nDescribe Output:\n")
print(description)

mean_returns = returns_df.mean()

std_returns = returns_df.std()

highest_mean_asset = mean_returns.idxmax()

highest_std_asset = std_returns.idxmax()

print("\nAsset with Highest Mean Return:")
print(highest_mean_asset)

print("\nAsset with Highest Standard Deviation:")
print(highest_std_asset)

rf_annual = 0.02

rf_weekly = rf_annual / 52

annual_mean = returns_df.mean() * 52

annual_std = returns_df.std() * np.sqrt(52)

sharpe_ratios = (annual_mean - rf_annual) / annual_std

print("\nAnnualised Sharpe Ratios:\n")
print(sharpe_ratios)