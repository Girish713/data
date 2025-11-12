import pandas as pd
import numpy as np

# Sample Time Series Data (e.g., Daily COVID Cases)
dates = pd.to_datetime(pd.date_range(start='2025-01-01', periods=10, freq='D'))
cases = [100, 110, 105, 120, 130, 125, 140, 150, 145, 160]
ts_data = pd.Series(cases, index=dates)

## --- Descriptive Analytics Techniques ---

# 1. Basic Statistics
print("Basic Statistics:\n", ts_data.describe())

# 2. Daily Change (Rate of Increase/Decrease)
daily_change = ts_data.diff().dropna()
print("\nDaily Change in Cases:\n", daily_change)

# 3. Rolling Average (Trend Smoothing)
# Calculate a 3-day simple moving average
rolling_mean = ts_data.rolling(window=3).mean().dropna()
print("\n3-Day Rolling Mean (Smoothed Trend):\n", rolling_mean)

# 4. Stationarity Check (Basic Mean over halves)
# A stationary series has consistent statistical properties over time
split_point = len(ts_data) // 2
mean_first_half = ts_data[:split_point].mean()
mean_second_half = ts_data[split_point:].mean()
print(f"\nMean First Half: {mean_first_half}, Mean Second Half: {mean_second_half}")

# Full Time Series Decomposition and ARIMA require dedicated libraries.
