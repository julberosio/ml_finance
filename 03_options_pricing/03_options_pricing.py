# -*- coding: utf-8 -*-
"""03_options_pricing.py

Results are commented as requested.
Note that the run times here are from a local run.

## American options with Binomial tree pricing (CRR)

### Generating the synthetic dataset
"""

# Import libraries

# Core packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gc
import random

# Machine learning tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set a global seed for reproducibility
seed = 2025
np.random.seed(seed)
random.seed(seed)

# Generate training and test parameter grids for American option pricing
# We are using Schoutens et al. (2018): they used diff. configs for training and testing

# Training range: wide
TRAIN_RANGES = {
    'S':     {'dist': 'uniform', 'low': 40,  'high': 160}, # Spot price
    'K':     {'dist': 'uniform', 'low': 40,  'high': 160}, # Strike price
    'T':     {'dist': 'uniform', 'low': 0.1, 'high': 2.0}, # Time to maturity (years)
    'r':     {'dist': 'uniform', 'low': 0.01, 'high': 0.05}, # Risk-free interest rate (annualised)
    'q':     {'dist': 'uniform', 'low': 0.00, 'high': 0.03}, # Dividend yield (annualised)
    'sigma': {'dist': 'uniform', 'low': 0.05, 'high': 0.55} # Volatility (standard deviation, annual)
}

# Testing range: narrower interior of training range
TEST_RANGES = {
    'S':     {'dist': 'uniform', 'low': 50,  'high': 150},
    'K':     {'dist': 'uniform', 'low': 50,  'high': 150},
    'T':     {'dist': 'uniform', 'low': 0.1, 'high': 2.0},
    'r':     {'dist': 'uniform', 'low': 0.015, 'high': 0.045},
    'q':     {'dist': 'uniform', 'low': 0.005, 'high': 0.025},
    'sigma': {'dist': 'uniform', 'low': 0.1,  'high': 0.5}
}

def sample_param(cfg, n):
    """
    Sample n values from a specified distribution config.

    Args:
        cfg (dict): Contains keys like 'dist', 'low', 'high'
        n (int): Number of values to sample

    Returns:
        np.ndarray of samples
    """
    if cfg['dist'] == 'uniform':
        return np.random.uniform(cfg['low'], cfg['high'], n)
    else:
        raise ValueError("Only 'uniform' dist supported for this task")

def generate_dataset(dists, n):
    """
    Generate a DataFrame of n parameter combinations based on given distribution configs.

    Args:
        dists (dict): Mapping of parameter names to sampling config
        n (int): Number of samples

    Returns:
        pd.DataFrame: Sampled parameter combinations
    """
    data = {key: sample_param(cfg, n) for key, cfg in dists.items()}
    return pd.DataFrame(data)

# Create training and test input parameter sets
size = 5000
n_train = size
n_test = size

# Training dataset
np.random.seed(seed)
train_df = generate_dataset(TRAIN_RANGES, n_train)
print("Training Set (first 5 rows):")
print(train_df.head())

# Testing dataset
test_df = generate_dataset(TEST_RANGES, n_test)
print("\nTest Set (first 5 rows):")
print(test_df.head())

"""
Training Set (first 5 rows):
            S           K         T         r         q     sigma
0   56.258580  155.363264  0.441919  0.015082  0.022108  0.301687
1  146.542204   47.051647  1.681895  0.010494  0.010157  0.330949
2  151.912677   83.798348  0.702643  0.041301  0.007603  0.157388
3   93.468180   66.809570  1.036689  0.033468  0.001357  0.481919
4   86.588266   45.686625  0.336911  0.049897  0.028060  0.372798

Test Set (first 5 rows):
            S           K         T         r         q     sigma
0  105.434521   63.557637  1.194770  0.039511  0.009955  0.409877
1  137.271494  132.656970  0.448406  0.028365  0.022268  0.262229
2  116.089853  118.298767  1.693308  0.039460  0.013992  0.237078
3   59.419918  110.554968  0.644918  0.016107  0.018726  0.298406
4   65.008021   66.315837  0.957036  0.023542  0.011020  0.435481
"""

"""### Binomial tree pricing (CRR)"""

# Binomial tree pricing function for American options

def binomial_price_am(S, K, T, r, q, sigma, steps=100, option_type='put'):
    """
    Computes the price of an American option using the Cox-Ross-Rubinstein (CRR) binomial tree model.

    Args:
        S (float): Spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualised).
        q (float): Dividend yield (annualised).
        sigma (float): Volatility of the underlying asset.
        steps (int): Number of time steps in the binomial tree.
        option_type (str): Type of the option: 'put' or 'call'.

    Returns:
        float: Price of the American option.
    """
    # Length of each time step
    dt = T / steps

    # Up and down movement factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)

    # Discount factor for one step
    discount = np.exp(-r * dt)

    # Generate asset prices at maturity
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    # Compute option payoffs at maturity
    if option_type == 'call':
        values = np.maximum(ST - K, 0)
    elif option_type == 'put':
        values = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        ST = ST[:i+1] / u  # roll back asset prices
        # Compute expected value at each node
        values = discount * (p * values[1:] + (1 - p) * values[:-1])
        # Apply early exercise condition at each node
        if option_type == 'call':
            values = np.maximum(values, ST - K)
        else:
            values = np.maximum(values, K - ST)

    # Return the value at the root node (t = 0)
    return values[0]

# Compute ground truth prices using binomial tree and preview labeled data

def price_all_options(df, steps=100, option_type='put'):
    """
    Computes binomial prices for each row in a DataFrame of option parameters.

    Args:
        df (pd.DataFrame): DataFrame with columns ['S', 'K', 'T', 'r', 'q', 'sigma'].
        steps (int): Number of binomial tree steps.
        option_type (str): 'put' or 'call'.

    Returns:
        np.ndarray: Array of computed option prices.
    """
    return np.array([
        binomial_price_am(
            S=row.S, K=row.K, T=row.T, r=row.r, q=row.q, sigma=row.sigma,
            steps=steps, option_type=option_type
        )
        for row in df.itertuples(index=False)
    ])

# Compute and time training prices
gc.collect()
start = time.time()
y_train = price_all_options(train_df, steps=100, option_type='put')
train_price_time = time.time() - start

# Compute and time test prices
gc.collect()
start = time.time()
y_test = price_all_options(test_df, steps=100, option_type='put')
test_price_time = time.time() - start

# Add price column to the datasets
train_df_labeled = train_df.copy()
train_df_labeled['price'] = y_train

test_df_labeled = test_df.copy()
test_df_labeled['price'] = y_test

# Print timing and datasets
print(f"Pricing time — train: {train_price_time:.2f}s | test: {test_price_time:.2f}s")

print("\nLabeled training set (first 5 rows):")
print(train_df_labeled.head())

print("\nLabeled test set (first 5 rows):")
print(test_df_labeled.head())

"""
Pricing time — train: 3.81s | test: 3.88s

Labeled training set (first 5 rows):
            S           K         T         r         q     sigma       price
0   56.258580  155.363264  0.441919  0.015082  0.022108  0.301687  154.344175
1  146.542204   47.051647  1.681895  0.010494  0.010157  0.330949   47.024232
2  151.912677   83.798348  0.702643  0.041301  0.007603  0.157388   72.942237
3   93.468180   66.809570  1.036689  0.033468  0.001357  0.481919   66.804457
4   86.588266   45.686625  0.336911  0.049897  0.028060  0.372798   44.543888

Labeled test set (first 5 rows):
            S           K         T         r         q     sigma       price
0  105.434521   63.557637  1.194770  0.039511  0.009955  0.409877   63.544100
1  137.271494  132.656970  0.448406  0.028365  0.022268  0.262229  128.560929
2  116.089853  118.298767  1.693308  0.039460  0.013992  0.237078  118.056013
3   59.419918  110.554968  0.644918  0.016107  0.018726  0.298406  110.062426
4   65.008021   66.315837  0.957036  0.023542  0.011020  0.435481   66.302877
"""


"""### Gaussian process regression"""

# Train and evaluate Gaussian Process Regressor (GPR) on the synthetic dataset

# Extract features and labels
X_train = train_df.values
X_test = test_df.values

# Scaling necessary for GPR robustness
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define kernel: Constant * RBF
kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0)

# Initialise GPR model
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-4,
    n_restarts_optimizer=3,
    random_state=seed
)

# Train model
gc.collect()
start = time.time()
gpr.fit(X_train_scaled, y_train)
gpr_train_time = time.time() - start

# Predict
gc.collect()
start = time.time()
y_pred_gpr = gpr.predict(X_test_scaled)
gpr_predict_time = time.time() - start

# Evaluation metrics
gpr_mae = mean_absolute_error(y_test, y_pred_gpr)
gpr_mse = mean_squared_error(y_test, y_pred_gpr)
gpr_r2 = r2_score(y_test, y_pred_gpr)

# Print
print(f"GPR — Train time: {gpr_train_time:.2f}s | Predict time: {gpr_predict_time:.2f}s")
print(f"GPR — MAE: {gpr_mae:.4f} | MSE: {gpr_mse:.4f} | R^2: {gpr_r2:.4f}")

"""
GPR — Train time: 124.21s | Predict time: 0.86s
GPR — MAE: 0.3189 | MSE: 0.2762 | R^2: 0.9997
"""

"""### Neural network"""

# Train and evaluate a neural network on the same dataset

# Initialise neural network
nn_model = MLPRegressor(
    hidden_layer_sizes=(256, 256, 256, 256), # Two hidden layers with 128 neurons each
    activation='relu', # ReLU activation
    solver='adam', # Adam optimiser
    max_iter=2000, # Maximum epochs
    early_stopping=True, # Enable early stopping
    validation_fraction=0.1, # 10% of training data for validation
    n_iter_no_change=20, # Stop if no improvement for 10 iterations
    random_state=seed
)

# Train the model
gc.collect()
start = time.time()
nn_model.fit(X_train_scaled, y_train)
nn_train_time = time.time() - start

# Predict on test data
gc.collect()
start = time.time()
y_pred_nn = nn_model.predict(X_test_scaled)
nn_predict_time = time.time() - start

# Compute performance metrics
nn_mae = mean_absolute_error(y_test, y_pred_nn)
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_r2 = r2_score(y_test, y_pred_nn)

# Print
print(f"NN — Train time: {nn_train_time:.2f}s | Predict time: {nn_predict_time:.4f}s")
print(f"NN — MAE: {nn_mae:.4f} | MSE: {nn_mse:.4f} | R^2: {nn_r2:.4f}")

"""
NN — Train time: 10.66s | Predict time: 0.0810s
NN — MAE: 0.3536 | MSE: 0.3029 | R^2: 0.9996
"""

"""### Evaluation"""

# Scatter plot of true vs predicted prices for GPR and NN

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axes[0].scatter(y_test, y_pred_gpr, s=10, alpha=0.3, color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title("Gaussian Process Regression")

axes[1].scatter(y_test, y_pred_nn, s=10, alpha=0.3, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title("Neural Network")

for ax in axes:
    ax.set_xlabel("True Price")
    ax.set_ylabel("Predicted Price")
    ax.grid(False)

plt.tight_layout()
plt.show(block=False)
plt.savefig("american.png")
plt.pause(10)
plt.close()

# Compile and display performance summary table

summary_df = pd.DataFrame([
    {
        "Model": "GPR",
        "MAE": round(gpr_mae, 4),
        "MSE": round(gpr_mse, 4),
        "R^2": round(gpr_r2, 4),
        "Train Time (s)": round(gpr_train_time, 2),
        "Predict Time (s)": round(gpr_predict_time, 4),
        "Speedup ×": round(test_price_time / gpr_predict_time, 2)
    },
    {
        "Model": "Neural Network",
        "MAE": round(nn_mae, 4),
        "MSE": round(nn_mse, 4),
        "R^2": round(nn_r2, 4),
        "Train Time (s)": round(nn_train_time, 2),
        "Predict Time (s)": round(nn_predict_time, 4),
        "Speedup ×": round(test_price_time / nn_predict_time, 2)
    }
])

print("Summary of Model Performance:")
print(summary_df)

"""
Summary of Model Performance:
            Model     MAE     MSE     R^2  Train Time (s)  Predict Time (s)  Speedup ×
0             GPR  0.3189  0.2762  0.9997          124.21            0.8566       4.53
1  Neural Network  0.3536  0.3029  0.9996           10.66            0.0810      47.84
"""

"""## Barrier Options with Heston + Euler discretisation + Monte Carlo

### Generating the synthetic dataset
"""

# Define Heston parameter distributions and generate synthetic dataset for barrier options

# Distribution configs for training (wide) and testing (narrower interior)
TRAIN_BARRIER_DISTS = {
    'S0':     {'dist': 'uniform', 'low': 40,   'high': 160},   # Spot
    'K':      {'dist': 'uniform', 'low': 40,   'high': 160},   # Strike
    'T':      {'dist': 'uniform', 'low': 0.1,  'high': 2.0},   # Time to maturity
    'r':      {'dist': 'uniform', 'low': 0.01, 'high': 0.05},  # Risk-free rate
    'q':      {'dist': 'uniform', 'low': 0.00, 'high': 0.03},  # Dividend yield
    'nu0':    {'dist': 'uniform', 'low': 0.01, 'high': 0.3},   # Initial variance
    'theta':  {'dist': 'uniform', 'low': 0.01, 'high': 0.3},   # Long-run variance
    'kappa':  {'dist': 'uniform', 'low': 1.0,  'high': 5.0},   # Mean reversion speed
    'eta':    {'dist': 'uniform', 'low': 0.1,  'high': 1.0},   # Volatility of volatility
    'rho':    {'dist': 'uniform', 'low': -0.9, 'high': 0.0}    # Correlation
}

TEST_BARRIER_DISTS = {
    'S0':     {'dist': 'uniform', 'low': 50,   'high': 150},
    'K':      {'dist': 'uniform', 'low': 50,   'high': 150},
    'T':      {'dist': 'uniform', 'low': 0.1,  'high': 2.0},
    'r':      {'dist': 'uniform', 'low': 0.015, 'high': 0.045},
    'q':      {'dist': 'uniform', 'low': 0.005, 'high': 0.025},
    'nu0':    {'dist': 'uniform', 'low': 0.02, 'high': 0.25},
    'theta':  {'dist': 'uniform', 'low': 0.02, 'high': 0.25},
    'kappa':  {'dist': 'uniform', 'low': 1.5,  'high': 4.5},
    'eta':    {'dist': 'uniform', 'low': 0.15, 'high': 0.9},
    'rho':    {'dist': 'uniform', 'low': -0.8, 'high': -0.1}
}

def sample_param(cfg, n):
    """
    Sample n values from a specified distribution config.

    Args:
        cfg (dict): Dictionary specifying 'dist', 'low', 'high', etc.
        n (int): Number of samples

    Returns:
        np.ndarray: Sampled values
    """
    if cfg['dist'] == 'uniform':
        return np.random.uniform(cfg['low'], cfg['high'], n)
    else:
        raise ValueError(f"Unsupported distribution: {cfg['dist']}")

def generate_barrier_dataset(dists, n):
    """
    Generate synthetic dataset for barrier options with Heston parameters.

    Args:
        dists (dict): Dictionary of parameter sampling configs
        n (int): Number of samples

    Returns:
        pd.DataFrame: Parameter combinations
    """
    return pd.DataFrame({
        key: sample_param(cfg, n) for key, cfg in dists.items()
    })

# Set dataset sizes
n_train_barrier = size
n_test_barrier = size

# Generate training and test parameter grids
np.random.seed(seed)
barrier_train_df = generate_barrier_dataset(TRAIN_BARRIER_DISTS, n_train_barrier)
barrier_test_df = generate_barrier_dataset(TEST_BARRIER_DISTS, n_test_barrier)

# Print
print("Barrier training set (first 5 rows):")
print(barrier_train_df.head())

print("\nBarrier test set (first 5 rows):")
print(barrier_test_df.head())

"""
Barrier training set (first 5 rows):
           S0           K         T         r         q       nu0     theta     kappa       eta       rho
0   56.258580  155.363264  0.441919  0.015082  0.022108  0.155978  0.170760  1.542305  0.618575 -0.164662
1  146.542204   47.051647  1.681895  0.010494  0.010157  0.172950  0.263087  4.306279  0.265034 -0.499039
2  151.912677   83.798348  0.702643  0.041301  0.007603  0.072285  0.201661  3.731951  0.854725 -0.166195
3   93.468180   66.809570  1.036689  0.033468  0.001357  0.260513  0.037318  3.422199  0.358119 -0.866798
4   86.588266   45.686625  0.336911  0.049897  0.028060  0.197223  0.053523  1.652633  0.505965 -0.643754

Barrier test set (first 5 rows):
           S0           K         T         r         q       nu0     theta     kappa       eta       rho
0   74.776586  127.469337  0.844708  0.031100  0.008677  0.079673  0.042911  3.419872  0.150711 -0.450336
1  136.338128   90.557371  1.935408  0.028063  0.014849  0.120408  0.089285  2.107450  0.533094 -0.790843
2   94.960770   84.269620  0.236460  0.041069  0.008030  0.094619  0.057631  3.818165  0.486110 -0.269459
3  118.628080   99.601467  1.012676  0.030485  0.015629  0.171607  0.173400  4.498103  0.813108 -0.510114
4   80.097743  133.870203  1.651095  0.029677  0.010910  0.085447  0.186467  3.659814  0.565833 -0.651652
"""

"""### Heston path simulation"""

# Simulate asset price paths under the Heston stochastic volatility model

def simulate_heston_paths(
    S0, nu0, theta, kappa, eta, rho, r, q, T,
    n_steps=252, n_paths=1000
):
    """
    Simulates asset price paths using the Heston stochastic volatility model via Euler discretisation.

    Args:
        S0 (float): Initial asset price
        nu0 (float): Initial variance
        theta (float): Long-run variance
        kappa (float): Mean reversion speed
        eta (float): Volatility of volatility
        rho (float): Correlation between asset and variance
        r (float): Risk-free interest rate
        q (float): Dividend yield
        T (float): Time to maturity (in years)
        n_steps (int): Number of time steps
        n_paths (int): Number of Monte Carlo simulation paths

    Returns:
        np.ndarray: Simulated asset price paths, shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps  # time step
    S = np.zeros((n_paths, n_steps + 1))  # asset prices
    nu = np.zeros((n_paths, n_steps + 1))  # variances

    # Initialise first column
    S[:, 0] = S0
    nu[:, 0] = nu0

    for t in range(1, n_steps + 1):
        # Generate correlated Brownian increments
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = np.random.normal(0, 1, n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Update variance
        nu[:, t] = nu[:, t-1] + kappa * (theta - nu[:, t-1]) * dt + eta * np.sqrt(np.maximum(nu[:, t-1], 0)) * np.sqrt(dt) * W2
        nu[:, t] = np.maximum(nu[:, t], 0)  # enforce positivity

        # Update asset price
        S[:, t] = S[:, t-1] * np.exp(
            (r - q - 0.5 * nu[:, t-1]) * dt + np.sqrt(nu[:, t-1]) * np.sqrt(dt) * W1
        )

    return S

# Compute price of barrier option using Monte Carlo and simulated Heston paths
# Since we used put for American options, we're using call here

def price_barrier_option(
    S_paths, K, r, T, barrier, option_type='call'
):
    """
    Computes the price of a down-and-out barrier option from simulated Heston paths.

    Args:
        S_paths (np.ndarray): Simulated asset paths (n_paths, n_steps + 1)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        barrier (float): Barrier level (knock-out if breached)
        option_type (str): 'call' or 'put'

    Returns:
        float: Discounted expected payoff of the barrier option
    """
    # Final asset prices
    final_prices = S_paths[:, -1]

    # Determine which paths breached the barrier
    knocked_out = np.any(S_paths <= barrier, axis=1)

    # Compute payoff only for surviving paths
    if option_type == 'call':
        payoff = np.maximum(final_prices - K, 0)
    else:
        payoff = np.maximum(K - final_prices, 0)

    payoff[knocked_out] = 0  # zero out knocked-out paths

    # Return discounted average payoff
    return np.exp(-r * T) * np.mean(payoff)

# Compute prices for entire barrier option dataset using Monte Carlo + Heston model

def price_all_barriers(
    df, barrier_factor=0.8, n_paths=1000, n_steps=252, option_type='call'
):
    """
    Computes barrier option prices for all rows in a DataFrame of Heston parameters.

    Args:
        df (pd.DataFrame): Input parameters, one row per option
        barrier_factor (float): Barrier = barrier_factor * S0
        n_paths (int): Number of Monte Carlo paths per option
        n_steps (int): Number of time steps per path
        option_type (str): 'call' or 'put'

    Returns:
        pd.Series: Vector of computed prices
    """
    prices = []

    for row in df.itertuples(index=False):
        # Simulate paths
        paths = simulate_heston_paths(
            S0=row.S0, nu0=row.nu0, theta=row.theta, kappa=row.kappa,
            eta=row.eta, rho=row.rho, r=row.r, q=row.q, T=row.T,
            n_paths=n_paths, n_steps=n_steps
        )

        # Set barrier level
        barrier = barrier_factor * row.S0

        # Compute option price
        price = price_barrier_option(
            S_paths=paths, K=row.K, r=row.r, T=row.T,
            barrier=barrier, option_type=option_type
        )

        prices.append(price)

    return pd.Series(prices)

# Time the pricing for the full training and test sets
gc.collect()
start = time.time()
barrier_train_prices = price_all_barriers(barrier_train_df)
barrier_train_time = time.time() - start

gc.collect()
start = time.time()
barrier_test_prices = price_all_barriers(barrier_test_df)
barrier_test_time = time.time() - start

# Append price columns
barrier_train_labeled = barrier_train_df.copy()
barrier_train_labeled['price'] = barrier_train_prices

barrier_test_labeled = barrier_test_df.copy()
barrier_test_labeled['price'] = barrier_test_prices

# Print
print(f"Barrier pricing time — train: {barrier_train_time:.2f}s | test: {barrier_test_time:.2f}s")

print("\nBarrier training set (first 5 rows):")
print(barrier_train_labeled.head())

print("\nBarrier test set (first 5 rows):")
print(barrier_test_labeled.head())

"""
Barrier pricing time — train: 132.91s | test: 135.01s

Barrier training set (first 5 rows):
           S0           K         T         r         q       nu0     theta     kappa       eta       rho      price
0   56.258580  155.363264  0.441919  0.015082  0.022108  0.155978  0.170760  1.542305  0.618575 -0.164662   0.002626
1  146.542204   47.051647  1.681895  0.010494  0.010157  0.172950  0.263087  4.306279  0.265034 -0.499039  51.390052
2  151.912677   83.798348  0.702643  0.041301  0.007603  0.072285  0.201661  3.731951  0.854725 -0.166195  53.299365
3   93.468180   66.809570  1.036689  0.033468  0.001357  0.260513  0.037318  3.422199  0.358119 -0.866798  23.854926
4   86.588266   45.686625  0.336911  0.049897  0.028060  0.197223  0.053523  1.652633  0.505965 -0.643754  32.049226

Barrier test set (first 5 rows):
           S0           K         T         r         q       nu0     theta     kappa       eta       rho      price
0   74.776586  127.469337  0.844708  0.031100  0.008677  0.079673  0.042911  3.419872  0.150711 -0.450336   0.063448
1  136.338128   90.557371  1.935408  0.028063  0.014849  0.120408  0.089285  2.107450  0.533094 -0.790843  38.053379
2   94.960770   84.269620  0.236460  0.041069  0.008030  0.094619  0.057631  3.818165  0.486110 -0.269459  12.774356
3  118.628080   99.601467  1.012676  0.030485  0.015629  0.171607  0.173400  4.498103  0.813108 -0.510114  23.508980
4   80.097743  133.870203  1.651095  0.029677  0.010910  0.085447  0.186467  3.659814  0.565833 -0.651652   3.304922
"""

"""### Gaussian process regression"""

# Train and evaluate GPR model on Heston barrier option dataset

# Extract features and labels
Xb_train = barrier_train_df.values
Xb_test = barrier_test_df.values
yb_train = barrier_train_prices.values
yb_test = barrier_test_prices.values

scaler_barrier = MinMaxScaler()
Xb_train_scaled = scaler_barrier.fit_transform(Xb_train)
Xb_test_scaled = scaler_barrier.transform(Xb_test)

# Define GPR kernel
kernel_barrier = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0)

# Initialise and train model
gpr_barrier = GaussianProcessRegressor(
    kernel=kernel_barrier,
    alpha=1e-4,
    n_restarts_optimizer=3,
    random_state=seed
)

gc.collect()
start = time.time()
gpr_barrier.fit(Xb_train_scaled, yb_train)
gpr_train_time = time.time() - start

# Predict
gc.collect()
start = time.time()
yb_pred_gpr = gpr_barrier.predict(Xb_test_scaled)
gpr_predict_time = time.time() - start

# Evaluate
gpr_mae = mean_absolute_error(yb_test, yb_pred_gpr)
gpr_mse = mean_squared_error(yb_test, yb_pred_gpr)
gpr_r2 = r2_score(yb_test, yb_pred_gpr)

# Print
print(f"GPR — Train time: {gpr_train_time:.2f}s | Predict time: {gpr_predict_time:.2f}s")
print(f"GPR — MAE: {gpr_mae:.4f} | MSE: {gpr_mse:.4f} | R^2 {gpr_r2:.4f}")

"""
GPR — Train time: 168.87s | Predict time: 0.96s
GPR — MAE: 1.3143 | MSE: 3.1248 | R^2 0.9880
"""

"""### Neural network"""

# Train and evaluate neural network on Heston barrier option dataset

# Initialise MLP model
nn_barrier = MLPRegressor(
    hidden_layer_sizes=(256, 256, 256, 256),
    activation='relu',
    solver='adam',
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=seed
)

# Train model
gc.collect()
start = time.time()
nn_barrier.fit(Xb_train_scaled, yb_train)
nn_train_time = time.time() - start

# Predict on test set
gc.collect()
start = time.time()
yb_pred_nn = nn_barrier.predict(Xb_test_scaled)
nn_predict_time = time.time() - start

# Evaluate
nn_mae = mean_absolute_error(yb_test, yb_pred_nn)
nn_mse = mean_squared_error(yb_test, yb_pred_nn)
nn_r2 = r2_score(yb_test, yb_pred_nn)

# Print
print(f"NN — Train time: {nn_train_time:.2f}s | Predict time: {nn_predict_time:.4f}s")
print(f"NN — MAE: {nn_mae:.4f} | MSE: {nn_mse:.4f} | R^2: {nn_r2:.4f}")

"""
NN — Train time: 16.61s | Predict time: 0.0873s
NN — MAE: 0.7064 | MSE: 1.1285 | R^2: 0.9957
"""

"""### Evaluation"""

# Scatter plots for barrier options

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

# GPR plot
axes[0].scatter(yb_test, yb_pred_gpr, s=10, alpha=0.3, color='blue')
axes[0].plot([yb_test.min(), yb_test.max()], [yb_test.min(), yb_test.max()], 'r--')
axes[0].set_title("Gaussian Process Regression")
axes[0].set_xlabel("True Price")
axes[0].set_ylabel("Predicted Price")
axes[0].grid(False)

# NN plot
axes[1].scatter(yb_test, yb_pred_nn, s=10, alpha=0.3, color='orange')
axes[1].plot([yb_test.min(), yb_test.max()], [yb_test.min(), yb_test.max()], 'r--')
axes[1].set_title("Neural Network")
axes[1].set_xlabel("True Price")
axes[1].grid(False)

plt.tight_layout()
plt.show(block=False)
plt.savefig("barrier.png")
plt.pause(10)
plt.close()

# Compile and display performance summary table for barrier option models

barrier_summary_df = pd.DataFrame([
    {
        "Model": "GPR",
        "MAE": round(gpr_mae, 4),
        "MSE": round(gpr_mse, 4),
        "R^2": round(gpr_r2, 4),
        "Train Time (s)": round(gpr_train_time, 2),
        "Predict Time (s)": round(gpr_predict_time, 4),
        "Speedup ×": round(barrier_test_time / gpr_predict_time, 2)
    },
    {
        "Model": "Neural Network",
        "MAE": round(nn_mae, 4),
        "MSE": round(nn_mse, 4),
        "R^2": round(nn_r2, 4),
        "Train Time (s)": round(nn_train_time, 2),
        "Predict Time (s)": round(nn_predict_time, 4),
        "Speedup ×": round(barrier_test_time / nn_predict_time, 2)
    }
])

print("Barrier Option Model Performance Summary:")
print(barrier_summary_df)

"""
Barrier Option Model Performance Summary:
            Model     MAE     MSE     R^2  Train Time (s)  Predict Time (s)  Speedup ×
0             GPR  1.3143  3.1248  0.9880          168.87            0.9567     141.12
1  Neural Network  0.7064  1.1285  0.9957           16.61            0.0873    1547.13
"""

"""
## Heston + Milstein discretisation
"""

# Simulate asset price paths under Heston model using Milstein discretisation for variance

def simulate_heston_milstein(
    S0, nu0, theta, kappa, eta, rho, r, q, T,
    n_steps=252, n_paths=1000
):
    """
    Simulates Heston paths using Milstein discretisation for variance.

    Args:
        S0 (float): Initial asset price
        nu0 (float): Initial variance
        theta (float): Long-run variance
        kappa (float): Mean reversion speed
        eta (float): Volatility of volatility
        rho (float): Correlation
        r (float): Risk-free rate
        q (float): Dividend yield
        T (float): Time to maturity
        n_steps (int): Time steps per path
        n_paths (int): Number of paths

    Returns:
        np.ndarray: Simulated asset price paths, shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    nu = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    nu[:, 0] = nu0

    for t in range(1, n_steps + 1):
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = np.random.normal(0, 1, n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        sqrt_nu = np.sqrt(np.maximum(nu[:, t-1], 0))
        dW_v = np.sqrt(dt) * W2

        # Milstein update for variance
        nu[:, t] = nu[:, t-1] + kappa * (theta - nu[:, t-1]) * dt + eta * sqrt_nu * dW_v \
                 + 0.25 * eta**2 * (dW_v**2 - dt)
        nu[:, t] = np.maximum(nu[:, t], 0)

        # Euler update for asset price
        S[:, t] = S[:, t-1] * np.exp(
            (r - q - 0.5 * nu[:, t-1]) * dt + sqrt_nu * np.sqrt(dt) * W1
        )

    return S

"""## Heston + Euler + antithetic variates"""

# Simulate Heston asset paths using Euler method with antithetic variates

def simulate_heston_euler_antithetic(
    S0, nu0, theta, kappa, eta, rho, r, q, T,
    n_steps=252, n_paths=500  # Generates 2 * n_paths total
):
    """
    Simulates Heston paths using Euler method with antithetic variates for variance reduction.

    Args:
        S0 (float): Initial asset price
        nu0 (float): Initial variance
        theta (float): Long-run variance
        kappa (float): Mean reversion speed
        eta (float): Volatility of volatility
        rho (float): Correlation
        r (float): Risk-free rate
        q (float): Dividend yield
        T (float): Time to maturity
        n_steps (int): Time steps per path
        n_paths (int): Half the number of paths (total = 2 * n_paths)

    Returns:
        np.ndarray: Simulated asset price paths, shape (2 * n_paths, n_steps + 1)
    """
    dt = T / n_steps
    n_total = 2 * n_paths

    S = np.zeros((n_total, n_steps + 1))
    nu = np.zeros((n_total, n_steps + 1))

    S[:, 0] = S0
    nu[:, 0] = nu0

    for t in range(1, n_steps + 1):
        Z1_half = np.random.normal(0, 1, n_paths)
        Z2_half = np.random.normal(0, 1, n_paths)

        # Generate antithetic pairs
        Z1 = np.concatenate([Z1_half, -Z1_half])
        Z2 = np.concatenate([Z2_half, -Z2_half])

        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        sqrt_nu = np.sqrt(np.maximum(nu[:, t-1], 0))

        # Update variance using Euler
        nu[:, t] = nu[:, t-1] + kappa * (theta - nu[:, t-1]) * dt + eta * sqrt_nu * np.sqrt(dt) * W2
        nu[:, t] = np.maximum(nu[:, t], 0)

        # Update asset price using Euler
        S[:, t] = S[:, t-1] * np.exp(
            (r - q - 0.5 * nu[:, t-1]) * dt + sqrt_nu * np.sqrt(dt) * W1
        )

    return S

"""## Evaluation"""

# Price barrier options using Milstein and Antithetic simulation methods

def price_barrier_all_milstein(df, barrier_factor=0.8, n_paths=1000, n_steps=252, option_type='call'):
    """
    Computes barrier option prices using Milstein-discretised Heston paths.

    Args:
        df (pd.DataFrame): Heston parameter inputs
        barrier_factor (float): Barrier = barrier_factor * S0
        n_paths (int): Number of MC paths
        n_steps (int): Number of time steps
        option_type (str): 'call' or 'put'

    Returns:
        pd.Series: Barrier option prices
    """
    prices = []

    for row in df.itertuples(index=False):
        paths = simulate_heston_milstein(
            S0=row.S0, nu0=row.nu0, theta=row.theta, kappa=row.kappa,
            eta=row.eta, rho=row.rho, r=row.r, q=row.q, T=row.T,
            n_paths=n_paths, n_steps=n_steps
        )
        barrier = barrier_factor * row.S0
        price = price_barrier_option(paths, K=row.K, r=row.r, T=row.T, barrier=barrier, option_type=option_type)
        prices.append(price)

    return pd.Series(prices)


def price_barrier_all_antithetic(df, barrier_factor=0.8, n_paths=500, n_steps=252, option_type='call'):
    """
    Computes barrier option prices using Euler simulation with antithetic variates.

    Args:
        df (pd.DataFrame): Heston parameter inputs
        barrier_factor (float): Barrier = barrier_factor * S0
        n_paths (int): Half number of paths (total = 2 * n_paths)
        n_steps (int): Number of time steps
        option_type (str): 'call' or 'put'

    Returns:
        pd.Series: Barrier option prices
    """
    prices = []

    for row in df.itertuples(index=False):
        paths = simulate_heston_euler_antithetic(
            S0=row.S0, nu0=row.nu0, theta=row.theta, kappa=row.kappa,
            eta=row.eta, rho=row.rho, r=row.r, q=row.q, T=row.T,
            n_paths=n_paths, n_steps=n_steps
        )
        barrier = barrier_factor * row.S0
        price = price_barrier_option(paths, K=row.K, r=row.r, T=row.T, barrier=barrier, option_type=option_type)
        prices.append(price)

    return pd.Series(prices)

# Run pricing using Euler (from before), Milstein, and Antithetic methods

# Euler prices (already computed in Q2)
euler_prices = barrier_test_prices  # reuse from earlier

# Milstein prices
gc.collect()
start = time.time()
milstein_prices = price_barrier_all_milstein(barrier_test_df)
milstein_time = time.time() - start

# Antithetic prices
gc.collect()
start = time.time()
antithetic_prices = price_barrier_all_antithetic(barrier_test_df)
antithetic_time = time.time() - start

# Comparison summary
comparison_df = pd.DataFrame({
    "Method": ["Euler", "Milstein", "Antithetic"],
    "Total Time (s)": [round(barrier_test_time, 2), round(milstein_time, 2), round(antithetic_time, 2)],
    "Mean Price": [euler_prices.mean(), milstein_prices.mean(), antithetic_prices.mean()],
    "Std Dev": [euler_prices.std(), milstein_prices.std(), antithetic_prices.std()],
    "Min": [euler_prices.min(), milstein_prices.min(), antithetic_prices.min()],
    "Max": [euler_prices.max(), milstein_prices.max(), antithetic_prices.max()],
    "Mean Abs Diff vs Euler": [
        0.0,
        np.mean(np.abs(milstein_prices - euler_prices)),
        np.mean(np.abs(antithetic_prices - euler_prices))
    ],
    "Max Abs Diff vs Euler": [
        0.0,
        np.max(np.abs(milstein_prices - euler_prices)),
        np.max(np.abs(antithetic_prices - euler_prices))
    ]
}).round(4)

# Print
print("Comparison of Euler, Milstein, and Antithetic methods:")
print(comparison_df)

"""
Comparison of Euler, Milstein, and Antithetic methods:
       Method  Total Time (s)  Mean Price  Std Dev  Min      Max  Mean Abs Diff vs Euler  Max Abs Diff vs Euler
0       Euler          135.01     16.6238  16.1554  0.0  84.5356                  0.0000                 0.0000
1    Milstein          143.50     16.6181  16.1655  0.0  84.6492                  0.8618                10.1371
2  Antithetic          118.15     16.6212  16.1334  0.0  84.6572                  0.7807                 8.3797
"""
