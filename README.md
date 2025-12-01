# Stock Prediction Project

Multimodal Market Return Forecasting with Risk-Aware Decisions

**⚠️ DISCLAIMER**: This is for research and educational purposes only. Not financial advice.

## Project Overview

This project implements machine learning models for stock price prediction with proper backtesting, validation, and risk management. It includes:

- Multiple baseline and advanced models (Ridge, Lasso, XGBoost, LightGBM, LSTM)
- Walk-forward validation to avoid look-ahead bias
- Comprehensive technical feature engineering
- Cost-aware backtesting with transaction costs and slippage
- Risk management with VIX filtering

## Setup

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd stock_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### Configuration

Edit `config/config.yaml` to configure:
- Stock universe (`spy_only`, `top_10_sp500`, `top_50_sp500`)
- Date ranges
- Model parameters
- Validation settings
- Backtest parameters

## Usage

### 1. Download Data

```bash
python scripts/download_data.py --config config/config.yaml
```

Add `--force` to ignore cache and re-download.

### 2. Train Models

Train a Ridge regression model:
```bash
python scripts/train_model.py --model ridge --config config/config.yaml
```

Available models:
- `naive` - Random walk baseline
- `momentum` - Momentum baseline
- `ridge` - Ridge regression
- `lasso` - Lasso regression
- `xgboost` - XGBoost
- `lightgbm` - LightGBM

### 3. Run Backtest

```bash
python scripts/run_backtest.py --model ridge --config config/config.yaml
```

## Project Structure

```
stock_prediction/
├── config/                 # Configuration files
├── data/                   # Data storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── cache/
├── src/                    # Source code
│   ├── data/               # Data acquisition and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model implementations
│   ├── validation/         # Validation strategies
│   ├── backtest/           # Backtesting engine
│   ├── metrics/            # Performance metrics
│   └── utils/              # Utilities
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
└── reports/                # Generated reports and figures
```

## Features

### Technical Indicators
- RSI, MACD, ATR, Bollinger Bands
- Rolling statistics (mean, std, skew, kurtosis)
- Volume features
- Price momentum features

### Market Context
- VIX (volatility index)
- Sector ETFs (XLF, XLK)
- Gold (GLD) and Treasuries (IEF)
- US Dollar Index

### Models

**Baselines:**
- Naive (random walk)
- Momentum
- Ridge/Lasso regression

**Tree Models:**
- XGBoost
- LightGBM

**Sequence Models:**
- LSTM (Long Short-Term Memory)

### Validation

- Walk-forward validation with expanding window
- Purged, embargoed cross-validation
- Proper time-series splits to avoid look-ahead bias

### Backtesting

- Transaction costs and slippage modeling
- VIX-based risk filtering
- Position sizing
- Comprehensive performance metrics

## Testing

Run tests:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Results

Results include:

**Prediction Metrics:**
- MSE, RMSE, MAE
- R²
- Information Coefficient
- Directional Accuracy

**Trading Metrics:**
- Total/Annual Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate

## Development

### Adding a New Model

1. Create model file in appropriate directory (`src/models/`)
2. Implement `.fit()` and `.predict()` methods
3. Add to model factory
4. Update training script

### Adding New Features

1. Create feature module in `src/features/`
2. Add to feature pipeline
3. Update configuration

## License

MIT License - See LICENSE file for details

## Citation

If you use this project in your research, please cite:

```bibtex
@software{stock_prediction_2024,
  title = {Multimodal Market Return Forecasting},
  year = {2024},
  author = {Research Team}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
