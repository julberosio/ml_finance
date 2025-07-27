# ml_finance

# Machine Learning for Finance 🧠📈

This repository contains a collection of Python and R modules exploring advanced topics in **quantitative finance**, including volatility modelling, causal networks, options pricing, price forecasting, and algorithmic trading strategies.

Each module is self-contained, reproducible, and suitable for both academic and practical applications.

---

## 📁 Project Structure

### `01_volatility/` – **Volatility Estimation**
Estimates and compares the volatility of the S&P 500 index from 2015 to 2024 using various techniques:

- **Models implemented**:
  - Exponentially Weighted Moving Average (EWMA)
  - Historical rolling standard deviation
  - Parkinson estimator
  - Garman–Klass estimator

- **Features**:
  - Annualised volatility visualisation
  - Comparison of estimators during the COVID-19 crash
  - Summary statistics and correlation matrix

---

### `02_causality_network/` – **Causality & Correlation Networks**
Builds Granger causality and correlation-based networks across global stock indices:

- **Data**: Weekly and monthly return/volatility data (2005–2009)
- **Methods**:
  - Granger causality testing (lagged linear models)
  - Correlation thresholding
- **Output**:
  - Directed network graphs (`igraph`)
  - Exportable LaTeX tables of network structure

---

### `03_options_pricing/` – **Options Pricing & ML Approximations**
Priced American and Barrier options using both numerical and machine learning techniques:

#### 📌 American Options
- **Benchmark**: Binomial trees (CRR model)
- **ML Approximators**:
  - Gaussian Process Regression
  - Neural Networks (MLP with tuning)

#### 📌 Barrier Options under the Heston Model
- Monte Carlo pricing via:
  - Euler discretisation
  - Milstein scheme
  - Antithetic variates

- **Evaluation**:
  - RMSE and R² metrics
  - Runtime comparisons and speed-up factors

---

### `04_price_forecast/` – **S&P 500 Price & Return Forecasting**
Forecasts S&P 500 return, price, and directional movement using classical and modern machine learning models:

- **Feature Selection**:
  - LASSO regression (with lag engineering)
  - Granger causality (significance tests)

- **Models**:
  - Neural Networks (MLP with hyperparameter tuning)
  - LSTM (1-layer and 2-layer, fully tuned)
  - Gaussian Process Regression (kernel selection and optimisation)
  - ARMA + GARCH models (with normality transformation)

- **Evaluation**:
  - RMSE and accuracy metrics
  - Visual prediction overlays
  - Confidence intervals and sensitivity analysis

---

### `05_algorithmic_trading/` – **Backtesting System**
Designs and backtests long-short equity strategies using historical data from the S&P 500 constituents (2010–2022):

- **Strategies**:
  - Momentum-based (lookback returns)
  - Mean-reversion (MA crossovers)
  - Volume-driven rules

- **Backtesting Framework**:
  - Daily signal generation
  - Position sizing and exposure control
  - Aggregated portfolio P&L and trade logs

- **Performance Metrics**:
  - Cumulative return, Sharpe ratio, max drawdown
  - Exposure and position diagnostics
  - Visual analytics (heatmaps, return plots)

---

## 📦 Dependencies

Install the required Python packages:

```bash
pip install numpy pandas matplotlib yfinance scikit-learn statsmodels arch tensorflow igraph
```

> ⚠️ `yfinance` may require frequent updates due to rate limits:
```bash
pip install --upgrade yfinance
```

---

## 📊 Datasets

- [Yahoo Finance](https://finance.yahoo.com/) — Market data (S&P 500, equities)
- [Goyal & Welch Predictors](https://www.hec.ca/iea/data/GoyalWelch.html) — Monthly economic indicators
- Proprietary Bloomberg/FTSE data (where applicable, anonymised)

---

## 📄 Licence

This project is licensed under the MIT Licence. See the `LICENSE` file for details.

---

## 🧠 Acknowledgements

This work draws from:
- Academic literature in empirical finance
- Practical quantitative modelling methods
- Personal research in machine learning and financial econometrics

---

## 🚀 Getting Started

Clone the repository and run any module individually:

```bash
git clone https://github.com/your-username/quant-finance-lab.git
cd quant-finance-lab/01_volatility
python 01_volatility.py
```

> Note: Each folder contains a standalone `.py` file and requires internet access (e.g., for downloading market data via `yfinance`).

---

## 🧭 Future Extensions

- Integrate multi-asset forecasting (commodities, FX)
- Build real-time dashboards (e.g., Streamlit or Dash)
- Deploy trading strategies to live backtesting environments (e.g., QuantConnect)
- Add model explainability (e.g., SHAP values)

---

## 🙌 Contributions

Contributions and ideas are welcome. Please open an issue or submit a pull request.
