"""
Cryptocurrency Market Analysis
================================
Author: Githinji Geoffrey Kanyi
GitHub: https://github.com/Jeff-Kanyi
Description: Analysis of cryptocurrency price action, volume trends,
             market microstructure, and simple trading signal backtesting
             using public CoinGecko API data.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})
COLORS = {"BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff"}

# ── 1. Fetch Data ──────────────────────────────────────────────────────────
def fetch_ohlc(coin_id: str, days: int = 90) -> pd.DataFrame:
    """Fetch OHLC + volume data from CoinGecko public API."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").drop("timestamp", axis=1)

    # Fetch market chart for volume
    url2 = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params2 = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r2 = requests.get(url2, params=params2, timeout=15)
    r2.raise_for_status()
    data2 = r2.json()
    vol_df = pd.DataFrame(data2["total_volumes"], columns=["timestamp", "volume"])
    vol_df["date"] = pd.to_datetime(vol_df["timestamp"], unit="ms").dt.normalize()
    vol_df = vol_df.set_index("date")["volume"]
    df.index = df.index.normalize()
    df = df.join(vol_df, how="left")
    return df.sort_index()


# ── 2. Technical Indicators ────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Moving averages
    df["SMA_7"]  = df["close"].rolling(7).mean()
    df["SMA_21"] = df["close"].rolling(21).mean()
    df["EMA_9"]  = df["close"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_mid"]   = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    # Daily return & volatility
    df["daily_return"] = df["close"].pct_change()
    df["volatility_7"] = df["daily_return"].rolling(7).std() * (365 ** 0.5)

    return df


# ── 3. Simple Signal Backtest ──────────────────────────────────────────────
def backtest(df: pd.DataFrame, initial_capital: float = 10_000) -> pd.DataFrame:
    """
    Golden/Death Cross strategy:
      BUY  when SMA_7 crosses above SMA_21
      SELL when SMA_7 crosses below SMA_21
    """
    df = df.copy()
    df["signal"] = 0
    df.loc[df["SMA_7"] > df["SMA_21"], "signal"] = 1
    df["position"]       = df["signal"].diff()
    df["strategy_return"]= df["signal"].shift(1) * df["daily_return"]
    df["cum_market"]     = (1 + df["daily_return"]).cumprod()
    df["cum_strategy"]   = (1 + df["strategy_return"]).cumprod()
    df["portfolio_value"]= initial_capital * df["cum_strategy"]
    return df


# ── 4. Plotting ────────────────────────────────────────────────────────────
def plot_dashboard(dfs: dict):
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("Cryptocurrency Market Analysis Dashboard\nGithinji Geoffrey Kanyi",
                 fontsize=16, fontweight="bold", color="#c9d1d9", y=0.98)

    gs = fig.add_gridspec(5, 2, hspace=0.55, wspace=0.3)

    # ── Panel 1: Price + Bollinger Bands (BTC) ──
    ax1 = fig.add_subplot(gs[0, :])
    btc = dfs["BTC"]
    ax1.fill_between(btc.index, btc["BB_lower"], btc["BB_upper"],
                     alpha=0.15, color=COLORS["BTC"], label="Bollinger Bands")
    ax1.plot(btc.index, btc["close"],   color=COLORS["BTC"],  lw=1.8, label="BTC Close")
    ax1.plot(btc.index, btc["SMA_7"],   color="#58a6ff",      lw=1.2, linestyle="--", label="SMA 7")
    ax1.plot(btc.index, btc["SMA_21"],  color="#ff7b72",      lw=1.2, linestyle="--", label="SMA 21")
    ax1.set_title("Bitcoin (BTC) — Price Action & Bollinger Bands", fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.grid(True)

    # ── Panel 2: RSI ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(btc.index, btc["RSI"], color=COLORS["BTC"], lw=1.5)
    ax2.axhline(70, color="#ff7b72", lw=1, linestyle="--", alpha=0.8, label="Overbought (70)")
    ax2.axhline(30, color="#3fb950", lw=1, linestyle="--", alpha=0.8, label="Oversold (30)")
    ax2.fill_between(btc.index, 70, btc["RSI"].clip(lower=70),
                     alpha=0.2, color="#ff7b72")
    ax2.fill_between(btc.index, btc["RSI"].clip(upper=30), 30,
                     alpha=0.2, color="#3fb950")
    ax2.set_title("BTC — Relative Strength Index (RSI-14)", fontweight="bold")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.grid(True)

    # ── Panel 3 & 4: ETH and SOL price ──
    for i, (coin, col) in enumerate(zip(["ETH", "SOL"], [gs[2, 0], gs[2, 1]])):
        ax = fig.add_subplot(col)
        d = dfs[coin]
        ax.plot(d.index, d["close"], color=COLORS[coin], lw=1.5)
        ax.plot(d.index, d["SMA_7"],  color="#58a6ff", lw=1, linestyle="--", label="SMA 7")
        ax.plot(d.index, d["SMA_21"], color="#ff7b72", lw=1, linestyle="--", label="SMA 21")
        ax.set_title(f"{coin} — Price & Moving Averages", fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True)

    # ── Panel 5: Volatility comparison ──
    ax5 = fig.add_subplot(gs[3, 0])
    for coin, df in dfs.items():
        ax5.plot(df.index, df["volatility_7"] * 100,
                 color=COLORS[coin], lw=1.5, label=coin)
    ax5.set_title("7-Day Annualised Volatility (%)", fontweight="bold")
    ax5.set_ylabel("Volatility (%)")
    ax5.legend(facecolor="#161b22", edgecolor="#30363d")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax5.tick_params(axis='x', rotation=30)
    ax5.grid(True)

    # ── Panel 6: Return correlation heatmap ──
    ax6 = fig.add_subplot(gs[3, 1])
    returns = pd.DataFrame({c: d["daily_return"] for c, d in dfs.items()}).dropna()
    corr = returns.corr()
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    sns.heatmap(corr, ax=ax6, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax6.set_title("Return Correlation Matrix", fontweight="bold")

    # ── Panel 7: Backtest ──
    ax7 = fig.add_subplot(gs[4, :])
    bt = backtest(btc)
    ax7.plot(bt.index, bt["cum_market"]   * 100 - 100, color="#8b949e",
             lw=1.5, linestyle="--", label="Buy & Hold")
    ax7.plot(bt.index, bt["cum_strategy"] * 100 - 100, color=COLORS["BTC"],
             lw=2, label="SMA Crossover Strategy")

    # Mark buy/sell signals
    buys  = bt[bt["position"] ==  1]
    sells = bt[bt["position"] == -1]
    ax7.scatter(buys.index,  bt.loc[buys.index,  "cum_strategy"] * 100 - 100,
                marker="^", color="#3fb950", s=80, zorder=5, label="Buy Signal")
    ax7.scatter(sells.index, bt.loc[sells.index, "cum_strategy"] * 100 - 100,
                marker="v", color="#ff7b72", s=80, zorder=5, label="Sell Signal")

    ax7.axhline(0, color="#30363d", lw=1)
    ax7.set_title("BTC — SMA Crossover Backtest vs Buy & Hold (Cumulative Return %)",
                  fontweight="bold")
    ax7.set_ylabel("Cumulative Return (%)")
    ax7.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d")
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax7.grid(True)

    plt.savefig("crypto_analysis_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("✅ Dashboard saved → crypto_analysis_dashboard.png")
    return bt


# ── 5. Summary Stats ───────────────────────────────────────────────────────
def print_summary(dfs: dict, bt: pd.DataFrame):
    print("\n" + "="*60)
    print("  MARKET SUMMARY — Last 90 Days")
    print("="*60)
    for coin, df in dfs.items():
        ret   = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        vol   = df["volatility_7"].iloc[-1] * 100
        high  = df["high"].max()
        low   = df["low"].min()
        print(f"\n  {coin}")
        print(f"    Price Return : {ret:+.2f}%")
        print(f"    Current Vol  : {vol:.1f}% (annualised)")
        print(f"    90d High     : ${high:,.2f}")
        print(f"    90d Low      : ${low:,.2f}")

    strat_ret  = (bt["cum_strategy"].iloc[-1] - 1) * 100
    market_ret = (bt["cum_market"].iloc[-1]   - 1) * 100
    print(f"\n  BACKTEST RESULTS (BTC SMA Crossover)")
    print(f"    Strategy Return  : {strat_ret:+.2f}%")
    print(f"    Buy & Hold Return: {market_ret:+.2f}%")
    print(f"    Alpha Generated  : {strat_ret - market_ret:+.2f}%")
    print("="*60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    coins = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    print("📡 Fetching market data from CoinGecko...")
    dfs = {}
    for ticker, coin_id in coins.items():
        print(f"   → {ticker}...", end=" ")
        raw = fetch_ohlc(coin_id, days=90)
        dfs[ticker] = add_indicators(raw)
        print("done")

    print("\n📊 Generating dashboard...")
    bt = plot_dashboard(dfs)
    print_summary(dfs, bt)
