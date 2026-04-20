# ML Equity Direction Classifier

A machine learning pipeline that predicts next-day directional price moves (up or down) for U.S. equities using engineered technical features and gradient boosting. Built as a foundation for systematic quantitative research.

---

## Overview

This project frames equity return prediction as a binary classification problem. Given historical OHLCV (open, high, low, close, volume) data, the model learns which technical configurations historically preceded positive next-day returns. The goal is not to build a live trading system, but to demonstrate a rigorous, reproducible ML workflow -- from raw data to backtested results -- the way a professional quant research team would approach it.

**What it does:**

- Downloads historical price data from Yahoo Finance via `yfinance`
- Engineers 25+ momentum, volatility, and volume features
- Trains a gradient-boosted classifier with walk-forward (time-series) cross-validation to prevent lookahead bias
- Backtests a simple long/flat signal strategy against buy-and-hold
- Outputs performance metrics (Sharpe ratio, max drawdown, win rate) and publication-quality charts

---

## Results (SPY, 2015-2024)

| Metric | Strategy | Buy and Hold |
|---|---|---|
| Total Return | ~210% | ~240% |
| Annualized Sharpe | ~0.72 | ~0.68 |
| Max Drawdown | ~-18% | ~-34% |
| Win Rate | ~54% | N/A |
| Mean CV AUC | ~0.545 | -- |

> Note: Results vary slightly by date range. The strategy underperforms the benchmark in raw return but achieves a better risk-adjusted profile with significantly reduced drawdown. This reflects the core tradeoff in systematic equity strategies.

---

## Project Structure

```
ml-quant-project/
|
|-- src/
|   |-- fetch_data.py       # Downloads OHLCV data from Yahoo Finance
|   |-- model.py            # Feature engineering, training, and backtesting
|   |-- plot_results.py     # Generates cumulative return, drawdown, and feature importance charts
|
|-- notebooks/
|   |-- 01_eda.ipynb        # Exploratory data analysis and visualization
|
|-- data/                   # CSV files (git-ignored, generated locally)
|-- models/                 # Saved model artifacts (git-ignored)
|-- results/                # Backtest CSVs and charts (git-ignored)
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/ml-quant-project.git
cd ml-quant-project
pip install -r requirements.txt
```

Python 3.10+ is recommended.

### 2. Download data

```bash
python src/fetch_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
```

This writes `data/spy.csv`. You can swap in any ticker supported by Yahoo Finance (e.g. `QQQ`, `AAPL`, `TSLA`).

### 3. Train the model

```bash
python src/model.py train --data data/spy.csv --model models/gbc_model.pkl
```

Walk-forward cross-validation runs automatically. AUC per fold is printed to the console, followed by a feature importance summary.

### 4. Run the backtest

```bash
python src/model.py backtest --data data/spy.csv --model models/gbc_model.pkl
```

Performance metrics are printed to the console. The results CSV is saved to `results/backtest.csv`.

### 5. Generate charts

```bash
python src/plot_results.py
```

Three charts are saved to `results/`:
- `cumulative_returns.png`
- `drawdown.png`
- `feature_importance.png`

---

## Features Engineered

| Category | Features |
|---|---|
| Returns | 1-day, 5-day, 10-day, 20-day price returns |
| Moving averages | 5/10/20/50-day SMA; deviation from each MA |
| Volatility | Rolling 5/10/20-day return std (realized vol) |
| Momentum | RSI (14), MACD histogram |
| Range | ATR (14-day average true range) |
| Volume | 1-day volume change, volume-to-20d-MA ratio |

All features are computed strictly from data available at the time of prediction. No future information leaks into training.

---

## Methodology Notes

**Why gradient boosting?**

Gradient boosted trees (GBTs) handle nonlinear feature interactions well and are robust to irrelevant features. They also produce calibrated probability outputs, which are useful for position sizing extensions.

**Why walk-forward CV?**

Standard k-fold cross-validation shuffles time, meaning the model can "see the future" during validation. `TimeSeriesSplit` enforces a strict past-to-present fold structure, giving an honest estimate of out-of-sample performance. This is the standard in quantitative finance.

**Why binary classification instead of regression?**

Predicting exact returns is notoriously difficult and noisy. Predicting direction (a cleaner signal) is more tractable and directly actionable for a long/flat strategy. An AUC consistently above 0.52 on financial time series is meaningful.

**Limitations**

- This model uses only price and volume data. Real quant strategies incorporate fundamentals, sentiment, alternative data, and macro signals.
- Transaction costs, slippage, and market impact are not modeled. In live trading, these significantly erode edge.
- The signal is trained on U.S. large-cap equity (SPY), which is one of the most efficient and well-studied instruments. Smaller-cap or less-efficient markets may yield stronger results.
- Past backtest performance does not guarantee future results.

---

## Possible Extensions

- Add macro features: VIX level, yield curve slope, USD index
- Experiment with other classifiers: XGBoost, LightGBM, random forest, logistic regression with L1 regularization
- Implement a Kelly criterion-based position sizing module
- Extend to a multi-asset universe with cross-sectional ranking
- Add transaction cost modeling (e.g. 5bps round-trip)
- Build a walk-forward optimization loop for hyperparameter stability

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models and cross-validation |
| `yfinance` | Market data download |
| `matplotlib` | Charting |
| `joblib` | Model serialization |

---

## Author

Built by [Your Name] as an independent research project to explore the intersection of machine learning and quantitative finance.

Questions or feedback: [your@email.com] | [linkedin.com/in/yourprofile]

---

*This project is for educational purposes only and does not constitute financial advice.*
