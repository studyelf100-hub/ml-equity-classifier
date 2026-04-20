"""
plot_results.py
---------------
Generates publication-quality charts from backtest output:
  1. Cumulative returns: strategy vs. buy-and-hold benchmark
  2. Drawdown curve
  3. Feature importance bar chart

Usage:
    python src/plot_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
import os


RESULTS_CSV = "results/backtest.csv"
MODEL_PKL   = "models/gbc_model.pkl"
OUT_DIR     = "results"

plt.rcParams.update({
    "font.family":     "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#0d0d0d",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "text.color":        "#cccccc",
    "grid.color":        "#1e1e1e",
    "grid.linestyle":    "--",
})


def plot_cumulative(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["cum_strat"], color="#00ff88", linewidth=1.5, label="ML Strategy")
    ax.plot(df.index, df["cum_bench"], color="#4488ff", linewidth=1.0, alpha=0.7, label="Buy and Hold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title("Cumulative Returns", fontsize=14, pad=12)
    ax.legend(framealpha=0)
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_drawdown(df: pd.DataFrame) -> None:
    roll_max  = df["cum_strat"].cummax()
    drawdown  = (df["cum_strat"] - roll_max) / roll_max * 100

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(df.index, drawdown, 0, color="#ff3355", alpha=0.6)
    ax.plot(df.index, drawdown, color="#ff3355", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Strategy Drawdown", fontsize=14, pad=12)
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_feature_importance() -> None:
    if not os.path.exists(MODEL_PKL):
        print("Model not found -- skipping feature importance plot.")
        return

    artifact = joblib.load(MODEL_PKL)
    clf, features = artifact["model"], artifact["features"]

    imp = pd.Series(clf.feature_importances_, index=features)
    imp = imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#00ff88" if v == imp.max() else "#4488ff" for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors, height=0.6)
    ax.set_title("Feature Importance (top 15)", fontsize=14, pad=12)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_CSV):
        print(f"Results not found at {RESULTS_CSV}. Run the backtest first.")
    else:
        df = pd.read_csv(RESULTS_CSV, parse_dates=["Date"], index_col="Date")
        os.makedirs(OUT_DIR, exist_ok=True)
        plot_cumulative(df)
        plot_drawdown(df)
        plot_feature_importance()
        print("\nAll charts saved to results/")
