"""
model.py
--------
Trains a gradient-boosted classifier to predict next-day directional moves
on S&P 500 constituents using engineered technical features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ── Feature Engineering ──────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds momentum, volatility, and volume-based features to OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame with engineered features appended.
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Returns
    df["ret_1d"]  = close.pct_change(1)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # Moving averages and deviation from them
    for w in [5, 10, 20, 50]:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"dev_ma_{w}"] = (close - ma) / ma

    # Volatility (rolling std of daily returns)
    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std()

    # RSI (14-day)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR (14-day) -- measures true range volatility
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Volume features
    df["vol_change"]  = df["Volume"].pct_change(1)
    df["vol_ma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # MACD signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # Label: 1 if next-day close > today's close, else 0
    df["label"] = (close.shift(-1) > close).astype(int)

    return df


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads raw OHLCV CSV, engineers features, and returns a clean DataFrame
    along with the feature column names.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df = add_technical_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in {"Open", "High", "Low", "Close", "Volume", "label"}
    ]

    df = df.dropna()
    return df, feature_cols


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str, model_out: str = "models/gbc_model.pkl") -> None:
    """
    Trains the model using walk-forward (time-series) cross-validation,
    then retrains on the full dataset and saves the artifact.
    """
    print("Loading and preparing data...")
    df, features = load_and_prepare(csv_path)

    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Walk-forward CV -- prevents look-ahead bias
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    print("\nRunning time-series cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=SEED,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, proba)
        auc_scores.append(auc)
        print(f"  Fold {fold}: AUC = {auc:.4f}")

    print(f"\nMean AUC: {np.mean(auc_scores):.4f}  (+/- {np.std(auc_scores):.4f})")

    # Final model on full data
    print("\nTraining final model on full dataset...")
    final_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    )
    final_clf.fit(X_scaled, y)

    # Feature importance report
    importance = pd.Series(final_clf.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)
    print("\nTop 10 features by importance:")
    print(importance.head(10).to_string())

    # Save artifacts
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": final_clf, "scaler": scaler, "features": features}, model_out)
    print(f"\nModel saved to {model_out}")


# ── Backtesting ───────────────────────────────────────────────────────────────

def backtest(csv_path: str, model_path: str = "models/gbc_model.pkl") -> pd.DataFrame:
    """
    Runs a simple long-only backtest: go long when model predicts UP,
    stay flat otherwise. Returns a daily P&L DataFrame.
    """
    artifact = joblib.load(model_path)
    clf, scaler, features = artifact["model"], artifact["scaler"], artifact["features"]

    df, _ = load_and_prepare(csv_path)
    X = scaler.transform(df[features].values)

    df["signal"]    = clf.predict(X)
    df["strat_ret"] = df["signal"] * df["ret_1d"].shift(-1)
    df["bench_ret"] = df["ret_1d"].shift(-1)

    df = df.dropna(subset=["strat_ret"])

    # Cumulative performance
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()
    df["cum_bench"] = (1 + df["bench_ret"]).cumprod()

    # Sharpe ratio (annualized, risk-free = 0 for simplicity)
    sharpe = (df["strat_ret"].mean() / df["strat_ret"].std()) * np.sqrt(252)

    # Max drawdown
    roll_max = df["cum_strat"].cummax()
    drawdown = (df["cum_strat"] - roll_max) / roll_max
    max_dd   = drawdown.min()

    print("\n=== Backtest Results ===")
    print(f"Strategy total return : {df['cum_strat'].iloc[-1] - 1:.2%}")
    print(f"Benchmark total return : {df['cum_bench'].iloc[-1] - 1:.2%}")
    print(f"Annualized Sharpe      : {sharpe:.3f}")
    print(f"Max drawdown           : {max_dd:.2%}")
    print(f"Win rate               : {(df['strat_ret'] > 0).mean():.2%}")

    results_path = "results/backtest.csv"
    os.makedirs("results", exist_ok=True)
    df[["cum_strat", "cum_bench", "strat_ret", "bench_ret"]].to_csv(results_path)
    print(f"Results saved to {results_path}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Quant: Train or backtest.")
    parser.add_argument("mode", choices=["train", "backtest"],
                        help="'train' to fit the model, 'backtest' to evaluate.")
    parser.add_argument("--data", default="data/spy.csv",
                        help="Path to OHLCV CSV file.")
    parser.add_argument("--model", default="models/gbc_model.pkl",
                        help="Path to save/load model.")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data, args.model)
    else:
        backtest(args.data, args.model)
