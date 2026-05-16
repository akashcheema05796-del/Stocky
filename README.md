# EMA Cross — BTC-USDT-SWAP Backtest Suite

A professional algorithmic trading strategy built on [NautilusTrader](https://nautilustrader.io/),
backtested on **3 years of real OKX BTC-USDT-SWAP data** with an interactive HTML dashboard.

---

## Table of Contents

- [Strategy Overview](#strategy-overview)
- [Strategy Logic](#strategy-logic)
- [Enhancements](#enhancements)
- [Backtest Results](#backtest-results)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)

---

## Strategy Overview

| Property | Value |
|---|---|
| Instrument | BTC-USDT-SWAP (OKX perpetual) |
| Data | 3 years of 1-minute candles (~1.58 M bars) |
| Default Timeframe | 1-minute bars, 1H chart display |
| Entry | Market order on **true EMA crossover** |
| Primary exit | Two-phase trailing stop-loss |
| Secondary exit | EMA reversal (close + optional reverse) |
| Leverage | 10× |
| Starting balance | 10,000 USDT |

---

## Strategy Logic

### Entry — True Crossover Detection

The strategy compares the **current bar's** EMA relationship with the **previous bar's**
relationship. A trade fires only when the state flips — not when fast is merely above/below slow.

```
Crossed Up   (Fast crosses above Slow) → BUY  market order
Crossed Down (Fast crosses below Slow) → SELL market order
```

If already positioned in the opposite direction, the position is **reversed**
(close at market + new entry) on the crossover signal.

### Exit — Two-Phase Trailing Stop

```
Phase 1  Fixed SL
         Placed at entry ± stop_loss_pct immediately after fill.
         SL stays fixed — gives the trade room to breathe.

Phase 2  Trail activates once price reaches 1:1 R:R
         (price moves stop_loss_pct in our favour)
         SL then trails stop_loss_pct below the rolling peak (long)
         or above the rolling trough (short).
         The SL can only ratchet in our favour — never against us.

Override EMA crosses back before SL fires
         Strategy closes at market and optionally reverses direction.
```

### SL Rejection Guard

On fast-moving bars (common on 1-minute data), a just-submitted stop order can land
inside the current bid/ask spread and get rejected by the exchange. The strategy catches
the `OrderRejected` event and immediately closes the position at market — giving accurate
backtest accounting for gap-through events instead of leaving the position SL-less.

---

## Enhancements

Four optional filters are built into `EMACrossRRConfig` and exposed as top-level
constants in `run_backtest_chart.py`.

### 1. ATR-Based Dynamic Stop-Loss

```python
ATR_SL_MULT = 3.0   # SL = entry ± ATR(14) × 3.0
```

Replaces the fixed-% SL with a volatility-adaptive distance.
Wider during volatile periods (lets winners breathe), tighter during calm
conditions (cuts losses sooner).

**Sweep results — 1H, EMA(100/200):**

| Config | Trades | PF | Net USDT |
|--------|--------|----|----------|
| Fixed 2% SL | 2 | ∞ | +524 |
| ATR × 2 SL | 2 | ∞ | +521 |
| **ATR × 3 SL** | **2** | **∞** | **+527** |

> **Note for 1-minute bars:** ATR(14) on 1-minute data is only $10–30,
> making ATR-based SL too tight and prone to immediate rejection.
> Use `ATR_SL_MULT = 0.0` (fixed %) on 1-minute.

---

### 2. RSI Filter

```python
RSI_LONG_MAX  = 65.0   # skip long  entries when RSI(14) > 65 (overbought)
RSI_SHORT_MIN = 35.0   # skip short entries when RSI(14) < 35 (oversold)
```

Avoids entering trend trades when the market is already stretched.
Set to `70.0 / 30.0` (defaults) to effectively disable.

---

### 3. EMA Gap Filter

```python
MIN_EMA_GAP = 0.003   # require |fast − slow| / slow ≥ 0.3%
```

Skips crossovers where the two EMAs are nearly touching — "brush" crossovers
that are common in ranging/choppy markets and produce whipsaw losses.

> **Note:** For very slow EMAs (EMA 100/200 on 1H) the two lines barely
> separate at the moment of crossing, so this filter eliminates all trades.
> Most useful with faster EMA pairs (20/50, 50/100).

---

### 4. Post-SL Cooldown

```python
SL_COOLDOWN = 5   # skip the next 5 bars after a stop-out
```

Prevents immediate whipsaw re-entry after a stop-out.
The counter ticks down every bar regardless of whether a crossover signal fires.

---

## Backtest Results

All results use 3-year BTC-USDT-SWAP data, 10× leverage, $10,000 starting balance.

---

### 1-Hour Timeframe — EMA Period Comparison

| Strategy | Trades | Win Rate | Avg Win | Avg Loss | PF | Net USDT | Return |
|----------|--------|----------|---------|----------|----|----------|--------|
| EMA(20/50) Fixed 2% | 28 | 43% | +$47 | −$6 | 6.3 | +477 | +4.8% |
| EMA(20/50) ATR×2 | 5 | 60% | +$171 | −$2 | 112 | +509 | +5.1% |
| EMA(50/100) Fixed 2% | 14 | 50% | +$76 | −$5 | 14.7 | +498 | +5.0% |
| EMA(50/100) ATR×3 | 3 | 67% | +$260 | −$4 | 116 | +516 | +5.2% |
| EMA(100/200) Fixed 2% | 2 | 100% | +$262 | — | ∞ | +524 | +5.2% |
| **EMA(100/200) ATR×3** | **2** | **100%** | **+$263** | **—** | **∞** | **+527** | **+5.3%** |

---

### 1-Minute Timeframe — EMA Period Comparison

| Strategy | Trades/yr | Win Rate | PF | Net | Verdict |
|----------|-----------|----------|----|-----|---------|
| EMA(20/50) Fixed 0.5% | 2,213 | 41.7% | 0.66 | −49% | Over-trading / noise |
| EMA(50/200) Fixed 1% | 624 | 46.1% | 0.84 | −12% | Too many false signals |
| EMA(50/200) Fixed 2% | 256 | 47.1% | 0.93 | −4.3% | Near breakeven |
| EMA(50/200) Fixed 3% | 141 | 49.8% | 0.94 | −3.1% | Near breakeven |
| **EMA(6000/12000) Fixed 2%** | **17** | **52.9%** | **1.33** | **+1.23%** | **Profitable** |

**Key insight:** Short EMA pairs on 1-minute bars generate constant noise — every candle is a
potential crossover. EMA(6000/12000) on 1-minute is mathematically identical to
EMA(100/200) on 1-hour *(6,000 min = 100 h)*, but entries and exits are resolved to the
exact minute instead of the hour-bar close — yielding a 52.9% win rate and positive
win/loss asymmetry (+$18.23 avg win vs −$15.38 avg loss).

---

### Stop-Loss Variant Comparison (1H, EMA 100/200)

| Config | Trades | Win Rate | Net USDT | Return |
|--------|--------|----------|----------|--------|
| Fixed SL 0.5% | 2 | 100% | +212 | +2.1% |
| Fixed SL 2.0% | 2 | 100% | +525 | +5.2% |
| Trailing SL 2.0% | 2 | 100% | +525 | +5.2% |
| Fixed SL 2.0% + ATR×3 | 2 | 100% | **+527** | **+5.3%** |
| Fixed SL 2.0% + RSI 65/35 | 2 | 100% | +525 | +5.2% |

---

## Quick Start

### 1. Set up environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
pip install nautilus_trader httpx pandas

# macOS / Linux
source .venv/bin/activate
pip install nautilus_trader httpx pandas
```

### 2. Launch the dashboard

```bash
python run_backtest_chart.py
```

- **First run** — downloads ~3 years of 1-minute OKX candles (~5–8 min, saves to `data/`)
- **Subsequent runs** — loads from cache instantly, reruns backtest in ~3 min
- Opens `dashboard.html` automatically in your default browser

### 3. Run parameter sweeps

```bash
# Compare Fixed SL vs Breakeven variants (1H)
python compare_be.py

# Compare Fixed SL vs Trailing SL variants (1H)
python compare_trail.py

# Sweep all four filter combinations (1H)
python sweep_improvements.py

# Sweep EMA period pairs + filters across EMA sizes (1H)
python sweep_ema_params.py

# Sweep EMA period pairs on 1-minute data
python sweep_1m.py
```

---

## Configuration

All settings are at the top of `run_backtest_chart.py`:

```python
# ── Core ────────────────────────────────────────────────────────────────────
YEARS        = 3              # years of history
TIMEFRAME    = "1m"           # "1m" | "5m" | "15m" | "1H" | "4H"
FAST_EMA     = 6_000          # fast EMA period in bars
SLOW_EMA     = 12_000         # slow EMA period in bars
TRADE_SIZE   = Decimal("0.01")# lot size in BTC
START_BAL    = 10_000         # starting balance USDT
STOP_LOSS    = 0.02           # fixed SL fraction (ignored when ATR_SL_MULT > 0)

# ── Enhancement Filters ─────────────────────────────────────────────────────
ATR_SL_MULT   = 0.0   # 0 = fixed %;  >0 = ATR(14) × mult  (not recommended on 1m)
RSI_LONG_MAX  = 70.0  # skip longs  when RSI(14) > this    (70 = effectively off)
RSI_SHORT_MIN = 30.0  # skip shorts when RSI(14) < this    (30 = effectively off)
MIN_EMA_GAP   = 0.0   # min |fast−slow|/slow to enter      (0 = off)
SL_COOLDOWN   = 0     # bars to skip after SL hit           (0 = off)
```

### Timeframe → Equivalent EMA Period

| Trend Horizon | 1m bars | 5m bars | 15m bars | 1H bars | 4H bars |
|---|---|---|---|---|---|
| 20-hour fast EMA | 1,200 | 240 | 80 | 20 | 5 |
| 50-hour fast EMA | 3,000 | 600 | 200 | 50 | 13 |
| **100-hour fast EMA** | **6,000** | **1,200** | **400** | **100** | **25** |
| **200-hour slow EMA** | **12,000** | **2,400** | **800** | **200** | **50** |

### Full `EMACrossRRConfig` Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instrument_id` | InstrumentId | — | Instrument to trade |
| `bar_type` | BarType | — | Bar series driving signals |
| `trade_size` | Decimal | — | Lot size per trade (BTC) |
| `fast_ema_period` | int | 100 | Fast EMA period in bars |
| `slow_ema_period` | int | 200 | Slow EMA period in bars |
| `stop_loss_pct` | float | 0.005 | Fixed SL as fraction of price |
| `breakeven_rr` | float | 0.0 | Move SL to entry at this R:R multiple (0 = off) |
| `atr_sl_multiplier` | float | 0.0 | ATR(14) × mult for SL; 0 = use `stop_loss_pct` |
| `atr_period` | int | 14 | ATR indicator period |
| `rsi_period` | int | 14 | RSI indicator period |
| `rsi_long_max` | float | 70.0 | Skip long when RSI exceeds this |
| `rsi_short_min` | float | 30.0 | Skip short when RSI is below this |
| `min_ema_gap_pct` | float | 0.0 | Minimum EMA separation fraction to enter |
| `sl_cooldown_bars` | int | 0 | Bars to wait after SL hit |
| `close_positions_on_stop` | bool | True | Close open positions on strategy stop |

---

## File Structure

```
strategies/
  ema_cross_rr.py          Core strategy — EMA cross with two-phase trailing SL,
                           RSI filter, ATR-based SL, EMA gap filter, cooldown,
                           and SL-rejection guard

run_backtest_chart.py      Main runner — fetches data, runs backtest, writes
                           dashboard.html and opens it in your browser

compare_be.py              Compares Fixed SL vs Breakeven variants
compare_trail.py           Compares Fixed SL vs Trailing SL variants
sweep_improvements.py      Sweeps all four filter combinations
sweep_ema_params.py        Sweeps EMA period pairs + filters (1H data)
sweep_1m.py                Sweeps EMA period pairs on 1-minute data

data/
  BTC_USDT_SWAP_1m_3y.csv.gz   Auto-downloaded 3-year 1-min cache (~41 MB)

dashboard.html             Latest generated dashboard (git-ignored)
```

---

## How It Works

### Data Pipeline

```
OKX REST API  →  1-min candles (300/request, polite 25 ms delay)
              →  cached to data/BTC_USDT_SWAP_1m_Xy.csv.gz
              →  resampled in pandas to target TIMEFRAME (5m/15m/1H/4H)
              →  built into NautilusTrader Bar objects
              →  fed into BacktestEngine
```

### Backtest Engine (NautilusTrader)

| Setting | Value |
|---|---|
| Account type | MARGIN (NETTING) |
| Leverage | 10× |
| Fill model | 95% limit-fill probability, 30% slippage |
| Maker fee | 0.02% |
| Taker fee | 0.05% |

### Dashboard Generation

| Step | Detail |
|---|---|
| Chart data | 1-min bars aggregated to 1H for browser performance (26 K candles) |
| EMA overlay | Computed via pandas `.ewm()` on raw bar closes |
| Trade markers | Entry arrows + exit dots with P&L tooltip, snapped to 1H candles |
| Equity curve | Account balance at every event, displayed as area chart |
| Output | Single self-contained HTML (~3–5 MB), no server required |

Chart library: [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)

---

## Built With

- [NautilusTrader](https://nautilustrader.io/) — high-performance backtesting and live trading
- [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) — interactive candlestick charts
- [pandas](https://pandas.pydata.org/) — data resampling and analysis
- [httpx](https://www.python-httpx.org/) — HTTP client for OKX REST API
