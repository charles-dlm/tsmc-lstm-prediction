"""
TSMC Stock Price Prediction using LSTM Neural Network
=====================================================
LSTM model for predicting TSMC closing stock prices from historical data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Reproducibility
tf.random.set_seed(7)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

# Load data
df = pd.read_csv('finance.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "AdjustedClose", "Volume"]
df = df.drop("Date", axis=1)

# Data cleaning
df['Volume'] = df['Volume'].str.replace(',', '', regex=False)
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df = df.dropna()

# Chronological order (oldest to newest)
df = df.iloc[::-1].reset_index(drop=True)

# Feature-target split
X = df.drop(["Close"], axis=1)
y = df['Close']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sequences(X, y, timesteps):
    """Create sequences for time series prediction."""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps].values)
        y_seq.append(y.iloc[i + timesteps])
    return np.array(X_seq), np.array(y_seq)


def train_test_split_temporal(X, y, train_size):
    """Temporal train-test split preserving chronological order."""
    split_idx = int(len(X) * train_size)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    if isinstance(y, pd.Series):
        y_train = y.iloc[:split_idx].to_frame()
        y_test = y.iloc[split_idx:].to_frame()
    else:
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# Train-test split
X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, train_size=0.80)

# Feature scaling
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_train_scaled = scaler_X.transform(X_train)

scaler_y.fit(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_train_scaled = scaler_y.transform(y_train)

# =============================================================================
# SEQUENCE CREATION
# =============================================================================

# Hyperparameter
timesteps = 5

# Create sequences
X_test_seq, y_test_seq = create_sequences(
    pd.DataFrame(X_test_scaled),
    pd.DataFrame(y_test_scaled),
    timesteps
)

X_train_seq, y_train_seq = create_sequences(
    pd.DataFrame(X_train_scaled),
    pd.DataFrame(y_train_scaled),
    timesteps
)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Build LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(timesteps, 5)))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam')

# =============================================================================
# TRAINING
# =============================================================================

# Training hyperparameters
epochs = 18
batch_size = 64

print("Training started...")
model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=2)
print("Training completed.")

# =============================================================================
# INFERENCE AND EVALUATION
# =============================================================================

# Predictions
y_train_pred = model.predict(X_train_seq)
y_test_pred = model.predict(X_test_seq)

# Inverse transform
y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_train_true = scaler_y.inverse_transform(y_train_seq)
y_test_pred = scaler_y.inverse_transform(y_test_pred)
y_test_true = scaler_y.inverse_transform(y_test_seq)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train_true[:, 0], y_train_pred[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test_true[:, 0], y_test_pred[:, 0]))

train_r2 = r2_score(y_train_true[:, 0], y_train_pred[:, 0])
test_r2 = r2_score(y_test_true[:, 0], y_test_pred[:, 0])

# Results
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Predictions vs ground truth
plt.figure(figsize=(12, 6))
plt.plot(y_test_true, label='Ground Truth', color='blue', linewidth=2)
plt.plot(y_test_pred, label='Predictions', color='red', linewidth=2)
plt.title('TSMC Stock Price Prediction: Ground Truth vs Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Residuals
plt.figure(figsize=(12, 4))
residuals = y_test_true[:, 0] - y_test_pred[:, 0]
plt.plot(residuals, label="Residuals")
plt.title("Residual Analysis")
plt.xlabel("Time Steps")
plt.ylabel("Residual Error")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()