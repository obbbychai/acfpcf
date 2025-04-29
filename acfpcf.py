import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import sys


# run with python3 acfpcf.py --lag 672 672 is changeable, keep in mind its 15min data
# lag is how far back in time you want to look for correlations
# lag of 4 compares each return with the return 4 periods back
# lag of 16 compares each return with the return 16 periods back
# when we set --lag 96, use all of our data(every 15 minute return we have)
# for each point, calculate correlation with up to 96 periods back
# this will give us a sense of how returns are correlated over time


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze return correlations with customizable lag periods')
    parser.add_argument('--lag', type=int, default=16,
                        help='Number of lag periods (each period is 15 minutes)')
    return parser.parse_args()

# Read the data efficiently
df = pd.read_csv('volatility_metrics.csv', parse_dates=['timestamp'], index_col='timestamp')

# Function to calculate ACF/PACF in parallel
def calculate_correlation(data, max_lag, method='acf'):
    if method == 'acf':
        return acf(data, nlags=max_lag, fft=True)
    else:
        return pacf(data, nlags=max_lag, method='ywm')

# Plot ACF/PACF
def plot_correlation(ax, corr, conf_level, title):
    lags = np.arange(len(corr))
    ax.vlines(lags, [0], corr, color='blue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(lags, conf_level, -conf_level, alpha=0.2, color='gray')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Lag (periods of 15min)', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    
    # Set y-axis limits to zoom in even more
    ax.set_ylim(-0.05, 0.05)  # Focus on correlations between -0.02 and 0.02
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines at key correlation levels
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.01, color='r', linestyle='--', alpha=0.3)
    
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=9)
    
    # Add vertical lines every 4 lags (1 hour)
    for i in range(0, len(lags), 4):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.2)

def get_time_description(periods):
    """Convert number of 15-min periods to a human-readable time description"""
    hours = (periods * 15) / 60
    if hours >= 24:
        days = hours / 24
        return f"{days:.1f} days"
    return f"{hours:.1f} hours"

# Parse command line arguments
args = parse_args()
LAG_PERIODS = args.lag
CONF_LEVEL = 1.80 / np.sqrt(len(df))  # 95% confidence interval

print(f"\nAnalyzing correlations up to {LAG_PERIODS} periods ({get_time_description(LAG_PERIODS)})")

# Calculate ACF and PACF for both simple and log returns
simple_returns_clean = df['simple_returns'].dropna().values
log_returns_clean = df['log_returns'].dropna().values
    
acf_simple = calculate_correlation(simple_returns_clean, LAG_PERIODS, 'acf')
pacf_simple = calculate_correlation(simple_returns_clean, LAG_PERIODS, 'pacf')
acf_log = calculate_correlation(log_returns_clean, LAG_PERIODS, 'acf')
pacf_log = calculate_correlation(log_returns_clean, LAG_PERIODS, 'pacf')

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot correlations
plot_correlation(axes[0,0], acf_simple, CONF_LEVEL, 'ACF - Simple Returns')
plot_correlation(axes[0,1], pacf_simple, CONF_LEVEL, 'PACF - Simple Returns')
plot_correlation(axes[1,0], acf_log, CONF_LEVEL, 'ACF - Log Returns')
plot_correlation(axes[1,1], pacf_log, CONF_LEVEL, 'PACF - Log Returns')

# Save plot
plt.tight_layout()
plt.savefig('acf_pacf_original_vs_forward.png')
plt.close()

# Print statistics
result = adfuller(df['simple_returns'].dropna())
print("\nAugmented Dickey-Fuller Test Results:")
print(f"ADF Statistic: {result[0]:.16f}")
print(f"p-value: {result[1]:.16f}")
print("\nCritical Values:")
for key, value in result[4].items():
    print(f"{key}: {value:.16f}")

# Define interesting lags to show in statistics
lag_points = [
    (1, "15 min"),
    (4, "1 hour"),
    (16, "4 hours"),
    (32, "8 hours"),
    (LAG_PERIODS-1, f"{get_time_description(LAG_PERIODS)}")
]

print("\nSimple Returns Correlation Statistics:")
for lag, desc in lag_points:
    if lag < LAG_PERIODS:
        print(f"Lag {lag} correlation ({desc}): {acf_simple[lag]:.4f}")

print("\nLog Returns Correlation Statistics:")
for lag, desc in lag_points:
    if lag < LAG_PERIODS:
        print(f"Lag {lag} correlation ({desc}): {acf_log[lag]:.4f}")
