"""04_price_forecast.py
Results are commented.
"""

# Import required libraries for all questions

# For data manipulation and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LASSO feature selection
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# For Granger Causality feature selection
from statsmodels.tsa.stattools import grangercausalitytests

# For neural networks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import itertools

# For LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# For GP regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as C

# Transformations
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest

# For ARMA and GARCH models
# !pip install arch
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic

# Fix random seed for reproducibility
np.random.seed(1521)

"""
============================================================================
MACHINE LEARNING MODELS
============================================================================

=== Data preparation and feature engineering ===============================
"""

# Load and parse the dataset
df = pd.read_csv('goyal-welch2022Monthly.csv')
df['date'] = pd.to_datetime(df['yyyymm'].astype(str), format='%Y%m') # Set dates to datetime
df.set_index('date', inplace=True) # Set date as index

# There are errors in the S&P 500 Index column when plotted
# Clean 'Index' column: remove commas and spaces BEFORE converting
df['Index'] = df['Index'].astype(str).str.replace(',', '').str.strip()
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')

# Only work with data from 1927 to 2021
df = df[(df.index >= '1927-01-01') & (df.index <= '2021-12-31')]

# Specify targets
df['SP500'] = df['Index'] # Price
df['SP500_return'] = np.log(df['SP500'] / df['SP500'].shift(1)) # Return
# Direction (binary target: 1 if return > 0, else 0)
df['SP500_direction'] = (df['SP500_return'] > 0).astype(int)

# Plots for visual inspection if we did the data cleaning correctly
# Price
plt.figure(figsize=(8, 4))
df['SP500'].plot(title='S&P500 Index')
plt.show()

# Returns
plt.figure(figsize=(8, 4))
df['SP500_return'].plot(title='S&P500 Monthly Log Returns')
plt.show()

# Direction
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df.index, df['SP500'], color='black', label='SP500 Index')
ax.fill_between(df.index, df['SP500'].min(), df['SP500'].max(),
                where=df['SP500_direction'] == 1,
                color='green', alpha=0.2, label='Up Months')
ax.fill_between(df.index, df['SP500'].min(), df['SP500'].max(),
                where=df['SP500_direction'] == 0,
                color='red', alpha=0.2, label='Down Months')
ax.set_title("S&P 500 Index with Direction Shading")
ax.legend()
plt.tight_layout()
plt.show()

# Preview cleaned dataset
df.head()

# Feature engineering
predictors = ['D12', 'b/m', 'svar']
targets = ['SP500_return', 'SP500', 'SP500_direction']
lagged_datasets = {}
max_lag = 12

# Create lagged datasets for each target
for target in targets:
    df_model = df[[target] + predictors].copy().dropna()

    # Lag the target and predictors
    for var in [target] + predictors:
        for lag in range(1, max_lag + 1):
            df_model[f'{var}_lag{lag}'] = df_model[var].shift(lag)

    # Drop rows with NaNs due to lagging
    df_model = df_model.dropna()
    lagged_datasets[target] = df_model

df_model.head()

"""
=== Feature selection ======================================================
"""

# Feature selection: LASSO

selected_features_lasso = {} # Define dictionary for selected features
alphas = np.logspace(-4, 0, 100)

for target, df_model in lagged_datasets.items():
    # Define X (exclude target and lagged target)
    feature_cols = [col for col in df_model.columns if col not in [target] and not col.startswith(target)]
    X = df_model[feature_cols].values
    y = df_model[target].values

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO with CV
    lasso = LassoCV(cv=5, alphas=alphas, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Store selected (non-zero coef) features in dictionary
    coef = pd.Series(lasso.coef_, index=feature_cols)
    selected_features_lasso[target] = coef[coef != 0].index.tolist()

selected_features_lasso

"""
{'SP500_return': ['b/m',
  'svar',
  'D12_lag11',
  'D12_lag12',
  'b/m_lag1',
  'b/m_lag2',
  'b/m_lag4',
  'b/m_lag5',
  'b/m_lag8',
  'b/m_lag10',
  'svar_lag1',
  'svar_lag2',
  'svar_lag3',
  'svar_lag4',
  'svar_lag5',
  'svar_lag6',
  'svar_lag7',
  'svar_lag8',
  'svar_lag9',
  'svar_lag10',
  'svar_lag11',
  'svar_lag12'],
 'SP500': ['D12',
  'b/m',
  'svar',
  'D12_lag12',
  'b/m_lag3',
  'b/m_lag6',
  'b/m_lag7',
  'b/m_lag12',
  'svar_lag1',
  'svar_lag2',
  'svar_lag3',
  'svar_lag4',
  'svar_lag8',
  'svar_lag9',
  'svar_lag10',
  'svar_lag11',
  'svar_lag12'],
 'SP500_direction': ['b/m',
  'svar',
  'D12_lag7',
  'D12_lag8',
  'D12_lag9',
  'D12_lag11',
  'b/m_lag1',
  'b/m_lag6',
  'b/m_lag9',
  'svar_lag2',
  'svar_lag3',
  'svar_lag4',
  'svar_lag7',
  'svar_lag8',
  'svar_lag10',
  'svar_lag12']}
"""

# Define X and y vectors from LASSO feature selection

Xy_datasets_lasso = {}  # format: {'SP500_return': (X, y), ...}

for target, df_model in lagged_datasets.items():
    selected = selected_features_lasso[target]

    # Define feature matrix X and target vector y
    X = df_model[selected].copy()
    y = df_model[target].copy()

    # Store
    Xy_datasets_lasso[target] = (X, y)

# Feature selection: Granger causality

selected_features_granger = {} # Define dictionary for selected features

for target in ['SP500_return', 'SP500', 'SP500_direction']: # Iterate through targets
    df_model = lagged_datasets[target]
    selected = []

    for predictor in ['D12', 'b/m', 'svar']:
        # Prepare pairwise data (dropna for Granger input)
        test_df = df_model[[target, predictor]].dropna()

        try:
            result = grangercausalitytests(test_df, maxlag=12, verbose=False)

            # Check if any lag had p-value < 0.05 in F-test
            significant = any(
                result[lag][0]['ssr_ftest'][1] < 0.05
                for lag in result
            )
            if significant:
                selected.append(predictor)
        except Exception as e:
            print(f"Failed Granger test for {predictor} → {target}: {e}")

    selected_features_granger[target] = selected

selected_features_granger

"""
{'SP500_return': ['b/m', 'svar'],
 'SP500': ['D12', 'svar'],
 'SP500_direction': ['D12']}
"""

# Use Granger-selected predictors (only raw vars, not lags)
Xy_datasets_granger = {}

for target, df_model in lagged_datasets.items():
    base_features = selected_features_granger[target]

    # Collect lags of those features only
    selected = [col for col in df_model.columns if any(f in col for f in base_features) and col != target]

    X = df_model[selected].copy()
    y = df_model[target].copy()

    Xy_datasets_granger[target] = (X, y)

# Define matrices for prediction
# Assuming we are using LASSO-selected features

X_ret, y_ret = Xy_datasets_lasso['SP500_return'] # Return
X_price, y_price = Xy_datasets_lasso['SP500'] # Price
X_dir, y_dir = Xy_datasets_lasso['SP500_direction'] # Direction

"""
=== MODEL 1: Neural network ================================================
"""

# Model specification

# Initial configuration
hidden_layer_sizes = (20,)
activation_function = 'relu'

results_nn = {} # Define dictionary for results

for target in ['SP500_return', 'SP500', 'SP500_direction']: # Iterate through targets
    X, y = Xy_datasets_lasso[target]

    # For price and return, use regression
    # For direction, use classification
    is_classification = (target == 'SP500_direction')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model
    if is_classification:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              activation=activation_function,
                              max_iter=1000,
                              random_state=0)
    else:
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation=activation_function,
                             max_iter=1000,
                             random_state=0)

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    if is_classification:
        score = accuracy_score(y_test, y_pred.round())
    else:
        score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE

    results_nn[target] = {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'score': score
    }

# Print initial results
print('Return (RMSE): ', results_nn['SP500_return']['score'])   # RMSE
print('Price (RMSE): ', results_nn['SP500']['score'] )         # RMSE
print('Direction (accuracy): ', results_nn['SP500_direction']['score']) # Accuracy

"""
Return (RMSE):  0.2329106307044783
Price (RMSE):  315.19946550124934
Direction (accuracy):  0.668141592920354
"""

# Hyperparameter tuning

# Configurations to try
layer_options = [(10,), (20,), (50,), (20, 10), (50, 25)]
activations = ['relu', 'tanh']

# Store best results
best_nn_results = {}

for target in ['SP500_return', 'SP500', 'SP500_direction']: # Iterate through targets
    X, y = Xy_datasets_lasso[target]

    # For price and return, use regression
    # For direction, use classification
    is_classification = (target == 'SP500_direction')

    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, shuffle=False, test_size=0.2)

    best_score = None
    best_config = None
    best_model = None

    for layers, act in itertools.product(layer_options, activations):
        if is_classification:
            model = MLPClassifier(hidden_layer_sizes=layers,
                                  activation=act,
                                  max_iter=1000,
                                  random_state=0)
        else:
            model = MLPRegressor(hidden_layer_sizes=layers,
                                 activation=act,
                                 max_iter=1000,
                                 random_state=0)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if is_classification:
                score = accuracy_score(y_test, y_pred.round())
            else:
                score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE

            if (best_score is None) or \
               (is_classification and score > best_score) or \
               (not is_classification and score < best_score):
                best_score = score
                best_config = (layers, act)
                best_model = model

        except Exception as e:
            print(f"Model failed for {target} with {layers}, {act}: {e}")

    best_nn_results[target] = {
        'best_score': best_score,
        'best_config': best_config,
        'model': best_model
    }

# Display summary
for target, result in best_nn_results.items():
    print(f"\nTarget: {target}")
    print(f"Best config: Layers={result['best_config'][0]}, Activation={result['best_config'][1]}")
    if target == 'SP500_direction':
        print(f"Best Accuracy: {result['best_score']:.3f}")
    else:
        print(f"Best RMSE: {result['best_score']:.3f}")

"""
Target: SP500_return
Best config: Layers=(20, 10), Activation=tanh
Best RMSE: 0.097

Target: SP500
Best config: Layers=(20, 10), Activation=relu
Best RMSE: 299.055

Target: SP500_direction
Best config: Layers=(50,), Activation=tanh
Best Accuracy: 0.757
"""

# Plot predictions from best_nn_results overlaid on the real time series
for target in ['SP500_return', 'SP500', 'SP500_direction']:
    # Get full data and re-create the same train-test split
    X, y = Xy_datasets_lasso[target]
    is_classification = (target == 'SP500_direction')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, shuffle=False, test_size=0.2
    )

    # Get best model and predict
    model = best_nn_results[target]['model']
    y_pred = model.predict(X_test)

    # Plot actual full series with prediction overlay on test window
    full_y = y.reset_index(drop=True)
    test_start_idx = len(y_train)

    plt.figure(figsize=(10, 4))
    plt.plot(full_y, label='Actual (Full Series)', color='black')
    plt.plot(range(test_start_idx, len(full_y)),
             y_pred, label='Predicted (NN)', color='blue')
    plt.axvline(test_start_idx, color='gray', linestyle='--', label='Train/Test Split')
    plt.title(f"NN Prediction — {target}")
    plt.xlabel("Time Index")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
=== MODEL 2: LSTM ==========================================================
"""

# Model specification

# Initial configuration
timesteps = 12
n_variables = 3
units = 50
epochs = 20
batch_size = 32

results_lstm = {} # Define dictionary for results

for target in ['SP500_return', 'SP500', 'SP500_direction']: # Iterate through targets
    df_model = lagged_datasets[target]
    y = df_model[target]
    is_classification = (target == 'SP500_direction')

    # Build X from all lags of ['D12', 'b/m', 'svar']
    ordered_cols = []
    for lag in range(1, timesteps + 1):
        for var in ['D12', 'b/m', 'svar']:
            ordered_cols.append(f'{var}_lag{lag}')

    X = df_model[ordered_cols]

    # Safety check
    expected_features = timesteps * n_variables
    if X.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features (got {X.shape[1]}). Check column ordering or lag count.")

    # Standardise and reshape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((-1, timesteps, n_variables))

    # Align y shape
    y_seq = y[-X_seq.shape[0]:].values if hasattr(y, 'values') else y[-X_seq.shape[0]:]

    # Train-test split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Model
    model = Sequential()
    model.add(LSTM(units, input_shape=(timesteps, n_variables)))
    model.add(Dense(1, activation='sigmoid' if is_classification else 'linear'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if is_classification else 'mse',
                  metrics=['accuracy'] if is_classification else ['mse'])

    # Fit
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Evaluate
    if is_classification:
        score = accuracy_score(y_test, np.round(y_pred))
    else:
        score = np.sqrt(mean_squared_error(y_test, y_pred))

    results_lstm[target] = {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'score': score
    }

# Print summary
for target, result in results_lstm.items():
    if target == 'SP500_direction':
        print(f"{target} — Accuracy: {result['score']:.3f}")
    else:
        print(f"{target} — RMSE: {result['score']:.3f}")

"""
SP500_return — RMSE: 0.172
SP500 — RMSE: 2041.114
SP500_direction — Accuracy: 0.597
"""

# Hyperparameter tuning (1 layer)

# Configurations to try
units_list = [20, 50, 100]
dense_activations = ['linear', 'relu', 'sigmoid']

best_lstm_results = {}

for target in ['SP500_return', 'SP500', 'SP500_direction']:
    df_model = lagged_datasets[target]
    y = df_model[target]
    is_classification = (target == 'SP500_direction')

    # Use full lags for LSTM
    ordered_cols = []
    for lag in range(1, timesteps + 1):
        for var in ['D12', 'b/m', 'svar']:
            ordered_cols.append(f'{var}_lag{lag}')
    X = df_model[ordered_cols]

    # Standardise and reshape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((-1, timesteps, n_variables))
    y_seq = y[-X_seq.shape[0]:].values if hasattr(y, 'values') else y[-X_seq.shape[0]:]

    # Train-test split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    best_score = None
    best_config = None
    best_model = None

    for units in units_list:
        for dense_act in dense_activations:
            model = Sequential()
            model.add(LSTM(units, input_shape=(timesteps, n_variables)))
            model.add(Dense(1, activation='sigmoid' if is_classification else dense_act))

            model.compile(optimizer='adam',
                          loss='binary_crossentropy' if is_classification else 'mse',
                          metrics=['accuracy'] if is_classification else ['mse'])

            try:
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
                y_pred = model.predict(X_test).flatten()

                if is_classification:
                    score = accuracy_score(y_test, np.round(y_pred))
                else:
                    score = np.sqrt(mean_squared_error(y_test, y_pred))

                if (best_score is None) or \
                   (is_classification and score > best_score) or \
                   (not is_classification and score < best_score):
                    best_score = score
                    best_config = (units, dense_act)
                    best_model = model
            except Exception as e:
                print(f"Failed config for {target}: {units} units, {dense_act} activation. Error: {e}")

    best_lstm_results[target] = {
        'best_score': best_score,
        'best_config': best_config,
        'model': best_model
    }

# Print summary
for target, result in best_lstm_results.items():
    print(f"\n{target}")
    print(f"Best config: {result['best_config']}")
    if target == 'SP500_direction':
        print(f"Best Accuracy: {result['best_score']:.3f}")
    else:
        print(f"Best RMSE: {result['best_score']:.3f}")

"""
SP500_return
Best config: (20, 'relu')
Best RMSE: 0.042

SP500
Best config: (100, 'linear')
Best RMSE: 2014.726

SP500_direction
Best config: (50, 'relu')
Best Accuracy: 0.615
"""

# Hyperparameter tuning (2 layers)

dense_activations = ['linear', 'relu', 'sigmoid']
results_stacked_lstm = {}

for target in ['SP500_return', 'SP500', 'SP500_direction']:
    df_model = lagged_datasets[target]
    y = df_model[target]
    is_classification = (target == 'SP500_direction')

    # Build X from all lags
    ordered_cols = []
    for lag in range(1, 13):
        for var in ['D12', 'b/m', 'svar']:
            ordered_cols.append(f'{var}_lag{lag}')
    X = df_model[ordered_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((-1, 12, 3))
    y_seq = y[-X_seq.shape[0]:].values

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    best_score = None
    best_config = None
    best_model = None

    for dense_act in dense_activations:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(12, 3)))
        model.add(LSTM(20))
        model.add(Dense(1, activation='sigmoid' if is_classification else dense_act))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy' if is_classification else 'mse',
                      metrics=['accuracy'] if is_classification else ['mse'])

        try:
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            y_pred = model.predict(X_test).flatten()

            if is_classification:
                score = accuracy_score(y_test, np.round(y_pred))
            else:
                score = np.sqrt(mean_squared_error(y_test, y_pred))

            if (best_score is None) or \
               (is_classification and score > best_score) or \
               (not is_classification and score < best_score):
                best_score = score
                best_config = dense_act
                best_model = model

        except Exception as e:
            print(f"Error for {target} with dense activation {dense_act}: {e}")

    results_stacked_lstm[target] = {
        'score': best_score,
        'activation': best_config,
        'model': best_model
    }

# Print results
for target, result in results_stacked_lstm.items():
    print(f"\n{target} — Dense activation: {result['activation']}")
    if target == 'SP500_direction':
        print(f"Accuracy: {result['score']:.3f}")
    else:
        print(f"RMSE: {result['score']:.3f}")

"""
SP500_return — Dense activation: relu
RMSE: 0.042

SP500 — Dense activation: relu
RMSE: 2066.495

SP500_direction — Dense activation: linear
Accuracy: 0.611
"""

# Hyperparameter tuning (timestep)

timestep_options = [6, 9, 12]
units = 50
epochs = 20
batch_size = 32

best_by_timestep = {}

for target in ['SP500_return', 'SP500', 'SP500_direction']:
    is_classification = (target == 'SP500_direction')
    best_score = None
    best_config = None
    best_model = None

    for timesteps in timestep_options:
        df_model = lagged_datasets[target].copy()
        y = df_model[target]

        ordered_cols = []
        for lag in range(1, timesteps + 1):
            for var in ['D12', 'b/m', 'svar']:
                col = f'{var}_lag{lag}'
                if col in df_model.columns:
                    ordered_cols.append(col)

        if len(ordered_cols) != timesteps * 3:
            continue

        X = df_model[ordered_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_seq = X_scaled.reshape((-1, timesteps, 3))
        y_seq = y[-X_seq.shape[0]:].values

        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        model = Sequential()
        model.add(LSTM(units, input_shape=(timesteps, 3)))
        model.add(Dense(1, activation='sigmoid' if is_classification else 'linear'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy' if is_classification else 'mse',
                      metrics=['accuracy'] if is_classification else ['mse'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_test).flatten()

        score = accuracy_score(y_test, np.round(y_pred)) if is_classification \
            else np.sqrt(mean_squared_error(y_test, y_pred))

        if (best_score is None) or \
           (is_classification and score > best_score) or \
           (not is_classification and score < best_score):
            best_score = score
            best_config = timesteps
            best_model = model

    best_by_timestep[target] = {
        'best_score': best_score,
        'timesteps': best_config,
        'model': best_model
    }

# Print results
for target, result in best_by_timestep.items():
    print(f"\n{target} — Best timestep: {result['timesteps']}")
    if target == 'SP500_direction':
        print(f"Accuracy: {result['best_score']:.3f}")
    else:
        print(f"RMSE: {result['best_score']:.3f}")

"""
SP500_return — Best timestep: 9
RMSE: 0.070

SP500 — Best timestep: 6
RMSE: 2038.348

SP500_direction — Best timestep: 6
Accuracy: 0.637
"""

final_lstm_configs = {
    'SP500_return': {
        'units': 20,
        'timesteps': 12,
        'activation': 'relu'
    },
    'SP500': {
        'units': 100,
        'timesteps': 12,
        'activation': 'linear'
    },
    'SP500_direction': {
        'units': 50,
        'timesteps': 6,
        'activation': 'sigmoid'
    }
}

final_lstm_results = {}

for target, config in final_lstm_configs.items():
    is_classification = (target == 'SP500_direction')
    timesteps = config['timesteps']
    units = config['units']
    act = config['activation']

    df_model = lagged_datasets[target]
    y = df_model[target]

    ordered_cols = []
    for lag in range(1, timesteps + 1):
        for var in ['D12', 'b/m', 'svar']:
            col = f'{var}_lag{lag}'
            if col in df_model.columns:
                ordered_cols.append(col)

    X = df_model[ordered_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((-1, timesteps, 3))
    y_seq = y[-X_seq.shape[0]:].values

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Final model
    model = Sequential()
    model.add(LSTM(units, input_shape=(timesteps, 3)))
    model.add(Dense(1, activation=act))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if is_classification else 'mse',
                  metrics=['accuracy'] if is_classification else ['mse'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()

    score = accuracy_score(y_test, np.round(y_pred)) if is_classification \
        else np.sqrt(mean_squared_error(y_test, y_pred))

    final_lstm_results[target] = {
        'score': score,
        'model': model,
        'config': config
    }

# Print summary
for target, result in final_lstm_results.items():
    print(f"\nFinal {target}")
    print(f"Config: {result['config']}")
    if target == 'SP500_direction':
        print(f"Accuracy: {result['score']:.3f}")
    else:
        print(f"RMSE: {result['score']:.3f}")

"""
Final SP500_return
Config: {'units': 20, 'timesteps': 12, 'activation': 'relu'}
RMSE: 0.044

Final SP500
Config: {'units': 100, 'timesteps': 12, 'activation': 'linear'}
RMSE: 2014.733

Final SP500_direction
Config: {'units': 50, 'timesteps': 6, 'activation': 'sigmoid'}
Accuracy: 0.637
"""

for target, config in final_lstm_configs.items():
    is_classification = (target == 'SP500_direction')
    timesteps = config['timesteps']
    model = final_lstm_results[target]['model']

    df_model = lagged_datasets[target].copy()
    y_full = df_model[target].reset_index(drop=True)

    ordered_cols = []
    for lag in range(1, timesteps + 1):
        for var in ['D12', 'b/m', 'svar']:
            col = f'{var}_lag{lag}'
            if col in df_model.columns:
                ordered_cols.append(col)

    X = df_model[ordered_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((-1, timesteps, 3))

    y_seq = y_full[-len(X_seq):].reset_index(drop=True)

    # Split index (no retraining)
    split_idx = int(len(X_seq) * 0.8)
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Time index alignment with original series
    aligned_start = len(y_full) - len(y_seq)
    aligned_split = aligned_start + split_idx
    test_range = range(aligned_split, aligned_split + len(y_pred))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(aligned_start, len(y_full)), y_seq, label='Actual (Aligned)', color='black')
    plt.plot(test_range, y_pred, label='Predicted (LSTM)', color='green')
    plt.axvline(aligned_split, color='gray', linestyle='--', label='Train/Test Split')
    plt.title(f"LSTM Prediction — {target}")
    plt.xlabel("Time Index")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
=== MODEL 3: GP regression =================================================
"""

# Model specification

# Kernel structures
gp_kernels = {
    'RBF': C(1.0) * RBF(length_scale=1.0),
    'DotProduct + White': DotProduct() + WhiteKernel(),
    'RBF + White': C(1.0) * RBF(length_scale=1.0) + WhiteKernel()
}

results_gp = {}

for target in ['SP500_return', 'SP500']:
    X, y = Xy_datasets_lasso[target]

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Subsample (GP is slow on large sets)
    max_samples = 500
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y = y[-max_samples:]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    best_score = None
    best_kernel = None
    best_model = None

    for name, kernel in gp_kernels.items():
        try:
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            model.fit(X_train, y_train)
            y_pred, std = model.predict(X_test, return_std=True)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            if best_score is None or rmse < best_score:
                best_score = rmse
                best_kernel = name
                best_model = model
        except Exception as e:
            print(f"Failed GP for {target} with kernel {name}: {e}")

    results_gp[target] = {
        'score': best_score,
        'kernel': best_kernel,
        'model': best_model
    }

# Print results
for target, result in results_gp.items():
    print(f"\n{target}")
    print(f"Best Kernel: {result['kernel']}")
    print(f"Best RMSE: {result['score']:.3f}")

"""
SP500_return
Best Kernel: RBF + White
Best RMSE: 0.030

SP500
Best Kernel: DotProduct + White
Best RMSE: 469.730
"""

# Hyperparameter tuning (kernel)

# Define kernels with tunable parameters
gp_kernels = {
    'RBF + White': C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                   WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)),

    'DotProduct + White': DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) +
                          WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)),

    'RBF only': C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
}

results_gp_tuned = {}

for target in ['SP500_return', 'SP500']:
    X, y = Xy_datasets_lasso[target]

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Subsample
    max_samples = 500
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y = y[-max_samples:]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    best_score = None
    best_kernel_name = None
    best_model = None
    best_log_marginal_likelihood = None
    best_y_pred = None
    best_y_std = None

    for name, kernel in gp_kernels.items():
        try:
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
            model.fit(X_train, y_train)

            y_pred, y_std = model.predict(X_test, return_std=True)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            lml = model.log_marginal_likelihood_value_

            if best_score is None or rmse < best_score:
                best_score = rmse
                best_kernel_name = name
                best_model = model
                best_log_marginal_likelihood = lml
                best_y_pred = y_pred
                best_y_std = y_std

        except Exception as e:
            print(f"Failed GP for {target} with kernel {name}: {e}")

    results_gp_tuned[target] = {
        'model': best_model,
        'kernel': best_kernel_name,
        'rmse': best_score,
        'lml': best_log_marginal_likelihood,
        'y_pred': best_y_pred,
        'y_std': best_y_std,
        'y_test': y_test
    }

# Print results
for target, result in results_gp_tuned.items():
    print(f"\n{target}")
    print(f"Best Kernel: {result['kernel']}")
    print(f"Best RMSE: {result['rmse']:.3f}")
    print(f"Log Marginal Likelihood: {result['lml']:.2f}")

"""
SP500_return
Best Kernel: DotProduct + White
Best RMSE: 0.032
Log Marginal Likelihood: -492.22

SP500
Best Kernel: DotProduct + White
Best RMSE: 469.730
Log Marginal Likelihood: -243.41
"""

for target in ['SP500_return', 'SP500']:
    result = results_gp_tuned[target]
    y_test = result['y_test'].values if hasattr(result['y_test'], 'values') else result['y_test']
    y_pred = result['y_pred']
    y_std = result['y_std']

    upper = y_pred + 2 * y_std
    lower = y_pred - 2 * y_std

    plt.figure(figsize=(10, 4))
    plt.plot(y_test, label='Actual', color='black')
    plt.plot(y_pred, label='Predicted', color='blue')
    plt.fill_between(range(len(y_pred)), lower, upper, color='blue', alpha=0.2, label='95% CI')
    plt.title(f"{target} — GP Prediction with 95% Interval")
    plt.xlabel("Test Sample Index")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
=== Sensitivity analysis ===================================================
"""

# Setup
target = 'SP500_return'
X_full, y_full = Xy_datasets_lasso[target]

# Ensure we're using only ['D12', 'b/m', 'svar'] and their lags
relevant_vars = ['D12', 'b/m', 'svar']
drop_vars = relevant_vars.copy()

kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) + \
         WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))

rmse_scores = {}

# Baseline with all predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full, shuffle=False, test_size=0.2)

model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_scores['All features'] = rmse_full

# Try removing one predictor at a time
for var in drop_vars:
    cols_to_keep = [col for col in X_full.columns if not col.startswith(var)]
    X_sub = X_full[cols_to_keep]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full, shuffle=False, test_size=0.2)

    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    rmse_scores[f"Removed {var}"] = rmse

# Print results
print("\n Sensitivity Analysis — GP on SP500_return")
for k, v in rmse_scores.items():
    print(f"{k:<20}: RMSE = {v:.4f}")

"""
Sensitivity Analysis — GP on SP500_return
All features        : RMSE = 0.0291
Removed D12         : RMSE = 0.0291
Removed b/m         : RMSE = 0.0371
Removed svar        : RMSE = 0.0313
"""

# Sort by RMSE for visual clarity
sorted_scores = dict(sorted(rmse_scores.items(), key=lambda x: x[1]))

plt.figure(figsize=(5, 5))
bars = plt.bar(sorted_scores.keys(), sorted_scores.values(), color='skyblue')
plt.axhline(rmse_scores['All features'], color='gray', linestyle='--', label='Full model RMSE')
plt.ylabel("RMSE")
plt.title("Sensitivity Analysis — GP on SP500_return")

# Annotate bars with RMSE values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f"{yval:.4f}",
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

"""
============================================================================
TRADITIONAL STATISTICAL MODELS
============================================================================

=== ARMA(p,q) + GARCH(1,1) =================================================
"""

# Choosing the best transformation

def try_normalizations(y):
    results = {}

    # Yeo-Johnson
    pt_yj = PowerTransformer(method='yeo-johnson')
    y_yj = pt_yj.fit_transform(y.values.reshape(-1, 1)).flatten()
    pval_yj = normaltest(y_yj).pvalue
    results['yeo-johnson'] = {'data': y_yj, 'pval': pval_yj, 'inverse': pt_yj.inverse_transform}

    # Standard Z-score
    scaler = StandardScaler()
    y_z = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    pval_z = normaltest(y_z).pvalue
    results['z-score'] = {'data': y_z, 'pval': pval_z, 'inverse': scaler.inverse_transform}

    # Log-transform (only if positive)
    if (y > 0).all():
        y_log = np.log(y)
        pval_log = normaltest(y_log).pvalue
        results['log'] = {'data': y_log, 'pval': pval_log, 'inverse': np.exp}

    # Pick best based on highest p-value
    best_method = max(results, key=lambda k: results[k]['pval'])
    best_data = results[best_method]['data']
    inverse_func = results[best_method]['inverse']

    print(f"Best transformation: {best_method}, Normality p-value: {results[best_method]['pval']:.4f}")
    return best_data, inverse_func, best_method

# Use the same dataset as before
# Only trying this for target with the best RMSE: return
y_raw = lagged_datasets['SP500_return']['SP500_return'].copy()
y_transformed, inverse_func, transform_name = try_normalizations(y_raw)

"""
Best transformation: yeo-johnson, Normality p-value: 0.0000
"""

# Select best (p, q) for ARMA(p, q) moel
order_selection = arma_order_select_ic(
    y_train,
    max_ar=5,
    max_ma=5,
    ic=['aic', 'bic'], # Use AIC or BIC
    trend='n'
)
print(order_selection.aic)

# Choose best based on AIC since we are doing forecasting
best_order = order_selection.aic.idxmin()
print(f"Selected ARMA order: {best_order}")

"""
             0            1            2            3            4  \
0  2650.260469  2646.542645  2648.479275  2638.213672  2640.200176
1  2646.772794  2648.526956  2648.723892  2640.208409  2639.093641
2  2647.837562  2647.362363  2635.324980  2633.828886  2635.684919
3  2639.441243  2641.077453  2633.933267  2635.780136  2637.826352
4  2640.539955  2640.257399  2635.603818  2637.598830  2635.769081
5  2637.926786  2636.816223  2638.374226  2640.190891  2639.604128

             5
0  2634.631012
1  2636.486061
2  2636.661722
3  2638.497715
4  2639.495528
5  2628.652247
Selected ARMA order: 0    5
1    5
2    3
3    2
4    2
5    5
"""

# Train-test split
split = int(len(y_transformed) * 0.8)
y_train = y_transformed[:split]
y_test = y_transformed[split:]

# ARMA(0,5)
arma_model = ARIMA(y_train, order=(best_order[0], 0, best_order[1])).fit()
resid = arma_model.resid

# GARCH(1,1) on ARMA residuals
garch_model = arch_model(resid, vol='Garch', p=1, q=1)
garch_fitted = garch_model.fit(disp='off')

# ARMA
resid_forecast = arma_model.forecast(steps=len(y_test))
resid_forecast = np.asarray(resid_forecast).flatten()

# Invert transformation to return to original domain
forecast_orig = inverse_func(resid_forecast.reshape(-1, 1)).flatten()
y_test_orig = y_raw[split:]

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(y_test_orig, forecast_orig))
print(f"\n ARMA+GARCH RMSE (original domain): {rmse:.4f}")

"""
ARMA+GARCH RMSE (original domain): 0.0413
"""
