# TSMC Stock Price Prediction with LSTM

A deep learning project implementing Long Short-Term Memory (LSTM) neural networks to predict Taiwan Semiconductor Manufacturing Company (TSMC) stock closing prices based on historical financial data.

## Project Overview

This project demonstrates the application of recurrent neural networks for financial time series forecasting. The LSTM model processes historical stock data (Open, High, Low, Volume, Adjusted Close) to predict future closing prices, providing insights into TSMC's stock price movements.

## Dataset

The model uses historical TSMC stock data with the following features:
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price (target variable)
- **AdjustedClose**: Dividend-adjusted closing price
- **Volume**: Trading volume

## Model Architecture

- **Input Layer**: LSTM with 256 units
- **Regularization**: Dropout layer (0.2)
- **Hidden Layer**: Dense layer with 32 neurons
- **Output Layer**: Single neuron for price prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

### Required Libraries

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tensorflow>=2.8.0
scikit-learn>=1.0.0
```



## Model Performance

The model is evaluated using:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **R² Score**: Coefficient of determination for model fit quality

Training configuration:
- **Sequence Length**: 5 timesteps
- **Train/Test Split**: 80/20
- **Epochs**: 18
- **Batch Size**: 64

## Visualizations

The script generates two key plots:

1. **Predictions vs Ground Truth**: Comparison of predicted and actual stock prices
2. **Residual Analysis**: Error distribution to assess model performance

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `timesteps` | 5 | Number of previous days for prediction |
| `train_size` | 0.80 | Training data proportion |
| `epochs` | 18 | Training iterations |
| `batch_size` | 64 | Batch size for training |
| `lstm_units` | 256 | LSTM layer neurons |
| `dropout_rate` | 0.2 | Regularization rate |



## Technical Details

### Data Preprocessing
- **Normalization**: MinMaxScaler for feature and target scaling
- **Sequence Creation**: Sliding window approach for temporal sequences
- **Temporal Split**: Chronological train-test split to prevent data leakage

### Model Features
- **LSTM Architecture**: Captures long-term dependencies in time series
- **Dropout Regularization**: Prevents overfitting
- **Sequential Processing**: Maintains temporal order in data



## Results Interpretation

- **Lower RMSE**: Better prediction accuracy
- **Higher R²**: Better model fit (closer to 1.0 is better)
- **Residual Analysis**: Random distribution indicates good model performance

## Limitations

**IMPORTANT: This model is for educational and research purposes only and should NOT be used for actual investment decisions.** Financial markets exhibit highly non-linear behavior influenced by numerous unpredictable factors including economic events, geopolitical factors, market sentiment, and company-specific news. The model has inherent limitations such as historical bias, feature constraints (only OHLCV data), inability to predict unprecedented events, and susceptibility to different market regimes. Additionally, the model operates with a short prediction horizon, may underestimate volatility, and lacks risk management mechanisms. Remember that past performance is not indicative of future results, and the complexity of financial markets extends far beyond what any historical data-driven model can capture.

## Disclaimer



## Author

- GitHub: [](https://github.com/charles-dlm)
- LinkedIn: [](https://www.linkedin.com/in/charles-delemolle/)

---
