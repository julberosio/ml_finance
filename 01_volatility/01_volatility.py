"""01_volatility.py
Results are commented.
"""

# Import required libraries for all questions

# For data manipulation and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For downloading data from Yahoo Finance
# Need to upgrade yfinance every time rate limit is reached
# !pip install yfinance --upgrade
# !pip show yfinance
import yfinance as yf

# Fix random seed for reproducibility
np.random.seed(1521)

# Download S&P 500 data from Yahoo Finance
# Focus on 2015 to 2024 so that we can capture the dynamics in 2020
sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31')
# Save data to CSV to avoid reaching rate limit with repeated requests
# sp500.to_csv('sp500_data.csv')
# sp500 = pd.read_csv('sp500_data.csv', index_col=0, parse_dates=True)

# Compute log returns using 'Close'
sp500['log_return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
returns = sp500['log_return'].dropna()

# Compute EWMA variance recursively
lambda_ = 0.94
ewma_var = [returns.iloc[0]**2]  # initial value

for r in returns[1:]:
    new_var = lambda_ * ewma_var[-1] + (1 - lambda_) * r**2
    ewma_var.append(new_var)

# Align sp500 data frame with ewma_var
sp500 = sp500.iloc[1:]
sp500['EWMA_var'] = ewma_var
sp500['EWMA_vol'] = np.sqrt(sp500['EWMA_var']) * np.sqrt(252)  # Annualised

# Historical volatility: 21-day rolling standard deviation
sp500['Hist_vol'] = returns.rolling(window=21).std() * np.sqrt(252)

# Parkinson volatility
hl_range = (1 / (4 * np.log(2))) * ((np.log(sp500['High'] / sp500['Low']))**2)
sp500['Park_var'] = hl_range.rolling(window=21).mean()
sp500['Park_vol'] = np.sqrt(sp500['Park_var']) * np.sqrt(252)

# Garman-Klass volatility
log_hl = np.log(sp500['High'] / sp500['Low'])**2
log_co = np.log(sp500['Close'] / sp500['Open'])**2
gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
sp500['GK_var'] = gk_var.rolling(window=21).mean()
sp500['GK_vol'] = np.sqrt(sp500['GK_var']) * np.sqrt(252)

# Plot the volatilities for comparison
plt.figure(figsize=(10, 6))
plt.plot(sp500['EWMA_vol'], label='EWMA volatility', alpha=0.9)
plt.plot(sp500['Hist_vol'], label='Historical volatility (21d)', alpha=0.7)
plt.plot(sp500['Park_vol'], label='Parkinson volatility', alpha=0.7)
plt.plot(sp500['GK_vol'], label='Garman-Klass volatility', alpha=0.7)
plt.title('S&P 500 Volatility Estimates (2015–2024)')
plt.xlabel('Date')
plt.ylabel('Annualised Volatility')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Summary statistics
vols = sp500[['EWMA_vol', 'Hist_vol', 'Park_vol', 'GK_vol']].dropna()
summary_stats = vols.describe().loc[['mean', 'std', 'min', 'max']]
print(summary_stats)

"""
Price   EWMA_vol  Hist_vol  Park_vol    GK_vol
Ticker
mean    0.153027  0.148866  0.114948  0.112043
std     0.093843  0.100982  0.062298  0.060181
min     0.046090  0.034688  0.035060  0.033925
max     0.848267  0.975552  0.532510  0.522887
"""

# Correlations
correlations = vols.corr()
print(correlations)

"""
Price            EWMA_vol  Hist_vol  Park_vol    GK_vol
Ticker
Price    Ticker
EWMA_vol         1.000000  0.966206  0.930299  0.922903
Hist_vol         0.966206  1.000000  0.960002  0.950902
Park_vol         0.930299  0.960002  1.000000  0.995918
GK_vol           0.922903  0.950902  0.995918  1.000000
"""

# Focus on the COVID crash
vols['2020-02':'2020-06'].plot(figsize=(12, 5), title="Volatility Around COVID Crash")

# Box plot to better understand distribution of each estimator
sp500[['EWMA_vol', 'Hist_vol', 'Park_vol', 'GK_vol']].plot.box(figsize=(5, 5))
plt.title('Distribution of Volatility Estimates')
plt.ylabel('Annualised Volatility')
plt.grid(True)
plt.show()
