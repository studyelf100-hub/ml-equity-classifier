"""
fetch_data.py
-------------
Downloads historical OHLCV data for a given ticker using yfinance
and saves it to the data/ directory as a clean CSV.

Usage:
    python src/fetch_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
"""

import argparse
import os
import yfinance as yf
import pandas as pd


def fetch(ticker: str, start: str, end: str, out_dir: str = "data") -> str:
    """
    Downloads daily OHLCV from Yahoo Finance and saves to CSV.

    Parameters
    ----------
    ticker  : Stock/ETF ticker symbol (e.g. "SPY").
    start   : Start date string "YYYY-MM-DD".
    end     : End date string "YYYY-MM-DD".
    out_dir : Directory to write the CSV into.

    Returns
    -------
    Path to the written CSV file.
    """
    print(f"Fetching {ticker} from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker.lower()}.csv")
    df.to_csv(out_path)

    print(f"Saved {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data from Yahoo Finance.")
    parser.add_argument("--ticker", default="SPY",   help="Ticker symbol (default: SPY)")
    parser.add_argument("--start",  default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out",    default="data",   help="Output directory")
    args = parser.parse_args()

    fetch(args.ticker, args.start, args.end, args.out)
