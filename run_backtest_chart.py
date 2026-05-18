#!/usr/bin/env python3
"""
EMA Cross Backtest  +  Professional HTML Dashboard
====================================================
Runs the EMA(9/14) strategy on 3 years of real BTC-USDT-SWAP 1-minute
data, then generates a fully self-contained HTML file that opens in any
browser and looks like a TradingView professional dashboard.

Features
--------
* Candlestick chart (TradingView Lightweight Charts)
* EMA 100 and EMA 200 overlaid
* Every trade entry/exit marked with coloured arrows + tooltip
* Equity curve panel below the main chart
* Stats dashboard: P&L, win rate, avg win/loss, profit factor, etc.
* Scrollable trade log table at the bottom

Run
---
    .venv\\Scripts\\python.exe run_backtest_chart.py
"""

import json
import sys
import time
import webbrowser
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
import pandas as pd

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType, AggregationSource, BarAggregation, OmsType, PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

from strategies.ema_cross_rr import EMACrossRR, EMACrossRRConfig

# ── Settings ───────────────────────────────────────────────────────────
YEARS        = 3              # years of history to backtest
TIMEFRAME    = "1m"           # chart + backtest bar size: "1m" "5m" "15m" "1H" "4H"
FAST_EMA     = 9              # fast EMA period (9-candle)
SLOW_EMA     = 14             # slow EMA period (14-candle)
TRADE_SIZE   = Decimal("0.01")
START_BAL    = 10_000
STOP_LOSS    = 0.005          # 0.5 % — tight SL suited for 1-min scalping
OUTPUT_FILE  = Path("dashboard.html").resolve()
CACHE_DIR    = Path("data")   # local cache folder — data downloaded once

# ───────────────────────────────────────────────────────────────────────

# Timeframe metadata table
_TF = {
    "1m":  dict(step=1,  agg=None,         resample=None,   label="1 MIN",  nt="MINUTE"),
    "5m":  dict(step=5,  agg=None,         resample="5min", label="5 MIN",  nt="MINUTE"),
    "15m": dict(step=15, agg=None,         resample="15min",label="15 MIN", nt="MINUTE"),
    "1H":  dict(step=1,  agg=None,         resample="1h",   label="1 HOUR", nt="HOUR"),
    "4H":  dict(step=4,  agg=None,         resample="4h",   label="4 HOUR", nt="HOUR"),
}
assert TIMEFRAME in _TF, f"Unknown TIMEFRAME '{TIMEFRAME}'. Choose: {list(_TF)}"
_tf = _TF[TIMEFRAME]


# ── 1. Fetch data with local cache ──────────────────────────────────────

def fetch(years: int) -> list:
    """
    Download 1-minute BTC-USDT-SWAP candles from OKX.

    * First run  : downloads everything, saves to data/BTC_USDT_SWAP_1m_Xy.csv.gz
    * Later runs : loads from cache instantly (skips download if < 24 h old)
    * OKX limits : 300 candles per request, ~40 req/2 s  → uses limit=300 + polite sleep
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"BTC_USDT_SWAP_1m_{years}y.csv.gz"

    target = years * 365 * 1440   # approx — OKX may have slightly fewer

    # ── load from cache if fresh ──
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 24:
            print(f"Loading from cache ({cache_file.name}, {age_h:.1f} h old) …")
            df = pd.read_csv(cache_file, compression="gzip", header=None)
            raw = df.values.tolist()
            print(f"  Loaded {len(raw):,} bars from cache")
            return raw
        else:
            print(f"Cache is {age_h:.0f} h old — refreshing …")

    # ── download ──
    url     = "https://www.okx.com/api/v5/market/history-candles"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    raw, after = [], ""

    print(f"Downloading ~{target:,} candles ({years} years × 1-min bars) …")
    print("  This takes ~5-8 min on first run. Subsequent runs load from cache.")

    t_start = time.time()
    req_count = 0

    with httpx.Client(headers=headers, timeout=20) as c:
        while len(raw) < target:
            p = {"instId": "BTC-USDT-SWAP", "bar": "1m", "limit": "300"}
            if after:
                p["after"] = after

            try:
                resp = c.get(url, params=p)
                resp.raise_for_status()
                d = resp.json()
            except Exception as e:
                print(f"\n  Request error: {e} — retrying in 2 s …")
                time.sleep(2)
                continue

            if d["code"] != "0" or not d["data"]:
                print(f"\n  OKX returned no more data (code={d['code']})")
                break

            batch = d["data"]
            raw.extend(batch)
            after = batch[-1][0]   # oldest ts in batch → next page cursor
            req_count += 1

            elapsed = time.time() - t_start
            rate    = len(raw) / elapsed if elapsed else 0
            eta_s   = (target - len(raw)) / rate if rate else 0
            print(f"  {len(raw):>8,} / ~{target:,}   "
                  f"[{elapsed/60:.1f} min elapsed  ETA {eta_s/60:.1f} min]",
                  end="\r")

            if len(batch) < 300:
                # OKX has no more history
                break

            # Polite rate limit: OKX allows 40 req / 2 s
            # We use 25 ms sleep → ~40 req/s — well within limits
            time.sleep(0.025)

    raw.reverse()   # oldest → newest
    actual = len(raw)
    print(f"\n  Downloaded {actual:,} bars  ({actual/1440:.0f} days)  "
          f"in {(time.time()-t_start)/60:.1f} min")

    # ── save cache ──
    print(f"  Saving to {cache_file} …")
    pd.DataFrame(raw).to_csv(cache_file, index=False, header=False, compression="gzip")
    print(f"  Cached ({cache_file.stat().st_size / 1e6:.1f} MB compressed)")

    return raw


# ── 1b. Aggregate 1-min raw to target timeframe ──────────────────────────

def aggregate_raw(raw_1m: list, timeframe: str) -> list:
    """
    Resample 1-minute raw OKX rows to a coarser timeframe.
    Returns the same list-of-lists format (ts_ms, o, h, l, c, vol, ...).
    If timeframe == "1m", returns raw unchanged.
    """
    if timeframe == "1m":
        return raw_1m

    rule = _TF[timeframe]["resample"]
    print(f"  Resampling {len(raw_1m):,} 1-min bars to {timeframe} …")

    ncols  = len(raw_1m[0])
    cnames = ["ts_ms","o","h","l","c","v","vc","vcq","confirm"][:ncols]
    df = pd.DataFrame(raw_1m, columns=cnames)
    df["ts_ms"] = df["ts_ms"].astype(int)
    df["o"]     = df["o"].astype(float)
    df["h"]     = df["h"].astype(float)
    df["l"]     = df["l"].astype(float)
    df["c"]     = df["c"].astype(float)
    df["v"]     = df["v"].astype(float)
    df.index    = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)

    agg = df.resample(rule).agg(
        o=("o","first"), h=("h","max"), l=("l","min"), c=("c","last"), v=("v","sum")
    ).dropna(subset=["o"])

    # Convert back to list format [ts_ms_str, o, h, l, c, v]
    result = [
        [str(int(idx.timestamp() * 1000)),
         str(row.o), str(row.h), str(row.l), str(row.c), str(row.v)]
        for idx, row in agg.iterrows()
    ]
    print(f"  -> {len(result):,} {timeframe} bars")
    return result


# ── 2. Build NautilusTrader objects ─────────────────────────────────────

def build_instrument() -> CryptoPerpetual:
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol("BTC-USDT-SWAP"), Venue("OKX")),
        raw_symbol=Symbol("BTC-USDT-SWAP"),
        base_currency=BTC, quote_currency=USDT, settlement_currency=USDT,
        is_inverse=False, price_precision=1, size_precision=2,
        price_increment=Price(0.1, precision=1),
        size_increment=Quantity(0.01, precision=2),
        multiplier=Quantity(1, precision=0),
        lot_size=Quantity(0.01, precision=2),
        max_quantity=Quantity(10_000, precision=2),
        min_quantity=Quantity(0.01, precision=2),
        max_notional=None, min_notional=Money(1, USDT),
        max_price=Price(10_000_000, precision=1),
        min_price=Price(0.1, precision=1),
        margin_init=Decimal("0.02"), margin_maint=Decimal("0.01"),
        maker_fee=Decimal("0.0002"), taker_fee=Decimal("0.0005"),
        ts_event=0, ts_init=0,
    )


def build_bars(raw: list, inst: CryptoPerpetual) -> list[Bar]:
    step  = _tf["step"]
    nt_agg = BarAggregation.HOUR if _tf["nt"] == "HOUR" else BarAggregation.MINUTE
    bt = BarType(
        instrument_id=inst.id,
        bar_spec=BarSpecification(step, nt_agg, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    return [
        Bar(bar_type=bt,
            open=Price(float(r[1]), precision=1),
            high=Price(float(r[2]), precision=1),
            low=Price(float(r[3]), precision=1),
            close=Price(float(r[4]), precision=1),
            volume=Quantity(float(r[5]), precision=2),
            ts_event=int(r[0]) * 1_000_000,
            ts_init=int(r[0]) * 1_000_000)
        for r in raw
    ]


# ── 3. Run backtest ──────────────────────────────────────────────────────

def run_backtest(bars: list[Bar], inst: CryptoPerpetual) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("CHART-001"),
        logging=LoggingConfig(log_level="WARNING", log_colors=True),
    ))
    engine.add_venue(
        venue=Venue("OKX"), oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN, base_currency=USDT,
        starting_balances=[Money(START_BAL, USDT)],
        fill_model=FillModel(prob_fill_on_limit=0.95,
                             prob_slippage=0.3, random_seed=42),
        default_leverage=Decimal("10"),
    )
    engine.add_instrument(inst)
    engine.add_data(bars)
    bar_type_str = f"BTC-USDT-SWAP.OKX-{_tf['step']}-{_tf['nt']}-LAST-EXTERNAL"
    engine.add_strategy(EMACrossRR(config=EMACrossRRConfig(
        instrument_id=inst.id,
        bar_type=BarType.from_str(bar_type_str),
        trade_size=TRADE_SIZE,
        fast_ema_period=FAST_EMA,
        slow_ema_period=SLOW_EMA,
        stop_loss_pct=STOP_LOSS,
    )))
    t0 = time.time()
    engine.run()
    print(f"Backtest done in {time.time()-t0:.1f}s")
    return engine


# ── 4. Build chart data ──────────────────────────────────────────────────

def build_chart_data(raw: list, engine: BacktestEngine):
    # ── Build chart candles from the backtest bars (already TIMEFRAME) ──
    # For sub-hourly timeframes (1m/5m/15m) we still collapse to 1H for the browser.
    # For 1H / 4H we use the bars directly.
    chart_resample = None if _tf["resample"] in ("1h", "4h") else "1h"

    ncols = len(raw[0]) if raw else 6
    col_names = ["ts_ms","o","h","l","c","v","vc","vcq","confirm"][:ncols]
    df_raw = pd.DataFrame(raw, columns=col_names)
    df_raw["ts_ms"] = df_raw["ts_ms"].astype(int)
    for col in ["o","h","l","c","v"]:
        df_raw[col] = df_raw[col].astype(float)
    df_raw.index = pd.to_datetime(df_raw["ts_ms"], unit="ms", utc=True)

    if chart_resample:
        print(f"  Aggregating {len(raw):,} {TIMEFRAME} bars to 1H for chart …")
        chart_df = df_raw.resample(chart_resample).agg(
            o=("o","first"), h=("h","max"), l=("l","min"), c=("c","last")
        ).dropna(subset=["o"])
    else:
        chart_df = df_raw[["o","h","l","c"]].copy()

    candles = [
        {"time": int(idx.timestamp()),
         "open": round(row.o, 1), "high": round(row.h, 1),
         "low":  round(row.l, 1), "close": round(row.c, 1)}
        for idx, row in chart_df.iterrows()
    ]
    print(f"  {len(candles):,} chart candles  ({len(raw):,} {TIMEFRAME} backtest bars)")

    # ── EMAs on the backtest closes, sampled to chart timeframe ──
    closes = pd.Series(df_raw["c"].values, index=df_raw.index)
    ema_fast_s = closes.ewm(span=FAST_EMA, adjust=False).mean().round(1)
    ema_slow_s = closes.ewm(span=SLOW_EMA, adjust=False).mean().round(1)

    if chart_resample:
        ema_fast_h = ema_fast_s.resample(chart_resample).last().dropna()
        ema_slow_h = ema_slow_s.resample(chart_resample).last().dropna()
    else:
        ema_fast_h = ema_fast_s
        ema_slow_h = ema_slow_s

    ema_fast_data = [{"time": int(t.timestamp()), "value": float(v)}
                     for t, v in ema_fast_h.items()]
    ema_slow_data = [{"time": int(t.timestamp()), "value": float(v)}
                     for t, v in ema_slow_h.items()]

    # Positions
    pos_df   = engine.trader.generate_positions_report()
    fills_df = engine.trader.generate_order_fills_report()

    if "side" in pos_df.columns:
        closed = pos_df[pos_df["side"] == "FLAT"].copy()
        closed["pnl"] = closed["realized_pnl"].str.replace(" USDT", "").astype(float)
    else:
        closed = pd.DataFrame(columns=["pnl", "ts_opened", "ts_closed",
                                        "avg_px_open", "avg_px_close",
                                        "entry", "duration_ns",
                                        "closing_order_id"])

    # Build closing order -> type map from fills
    sl_order_ids = set()
    if not fills_df.empty and "tags" in fills_df.columns:
        for idx, row in fills_df.iterrows():
            tags = str(row.get("tags", ""))
            if "SL" in tags:
                sl_order_ids.add(idx)   # client_order_id is the index

    # Trade markers + table rows
    markers = []
    trades  = []

    # Snap markers to the chart candle boundary
    CANDLE_SEC = {"1m": 60, "5m": 300, "15m": 900, "1H": 3600, "4H": 14400}
    snap = CANDLE_SEC.get(TIMEFRAME, 3600)
    # For sub-hourly timeframes we display 1H candles, so always snap to 1H
    if snap < 3600:
        snap = 3600

    for i, (_, row) in enumerate(closed.iterrows(), 1):
        entry_ts_raw = int(pd.Timestamp(row["ts_opened"]).timestamp())
        exit_ts_raw  = int(pd.Timestamp(row["ts_closed"]).timestamp())
        entry_ts  = (entry_ts_raw // snap) * snap
        exit_ts   = (exit_ts_raw  // snap) * snap
        # If entry and exit land on same candle, push exit to next candle
        if exit_ts <= entry_ts:
            exit_ts = entry_ts + snap

        entry_px  = float(row["avg_px_open"])
        exit_px   = float(row["avg_px_close"])
        pnl       = row["pnl"]
        direction = row["entry"]   # "BUY" or "SELL"
        duration  = row["duration_ns"] / 1e9 / 60  # minutes
        is_long   = direction == "BUY"
        is_win    = pnl > 0

        # Determine exit type from closing order ID
        closing_id = str(row.get("closing_order_id", ""))
        exit_type = "SL" if closing_id in sl_order_ids else "EMA Exit"

        # Entry arrow
        markers.append({
            "time": entry_ts,
            "position": "belowBar" if is_long else "aboveBar",
            "color": "#26a69a" if is_long else "#ef5350",
            "shape": "arrowUp" if is_long else "arrowDown",
            "text": f"#{i} {'L' if is_long else 'S'} {entry_px:,.0f}",
            "size": 1,
        })

        # Exit dot
        markers.append({
            "time": exit_ts,
            "position": "aboveBar" if is_long else "belowBar",
            "color": "#26a69a" if is_win else "#ef5350",
            "shape": "circle",
            "text": f"{'WIN' if is_win else 'LOSS'} {pnl:+.2f}",
            "size": 1,
        })

        trades.append({
            "num":       i,
            "entry_ts":  str(row["ts_opened"])[:16].replace("T", " "),
            "exit_ts":   str(row["ts_closed"])[:16].replace("T", " "),
            "side":      direction,
            "entry_px":  entry_px,
            "exit_px":   exit_px,
            "pnl":       round(pnl, 4),
            "win":       is_win,
            "exit_type": exit_type,
            "duration":  f"{duration:.0f} min",
        })

    # Sort markers by time (required by Lightweight Charts)
    markers.sort(key=lambda m: m["time"])

    # Equity curve from account report
    acct = engine.trader.generate_account_report(Venue("OKX"))
    acct_ts  = [int(pd.Timestamp(str(t)).timestamp()) for t in acct.index]
    acct_bal = [round(float(v), 2) for v in acct["total"]]
    equity   = [{"time": t, "value": v} for t, v in zip(acct_ts, acct_bal)]

    # Stats
    pnls      = closed["pnl"] if len(closed) else pd.Series([], dtype=float)
    winners   = pnls[pnls > 0]
    losers    = pnls[pnls < 0]
    start_bal = float(acct["total"].iloc[0])
    end_bal   = float(acct["total"].iloc[-1])
    net_pnl   = end_bal - start_bal
    win_rate  = len(winners) / len(pnls) * 100 if len(pnls) else 0
    avg_win   = float(winners.mean()) if len(winners) else 0
    avg_loss  = float(losers.mean())  if len(losers)  else 0
    pf        = abs(winners.sum() / losers.sum()) if len(losers) and losers.sum() != 0 else 0

    stats = {
        "start_bal":   f"{start_bal:,.2f}",
        "end_bal":     f"{end_bal:,.2f}",
        "net_pnl":     f"{net_pnl:+,.2f}",
        "net_pnl_pct": f"{net_pnl/start_bal*100:+.2f}%",
        "net_pnl_pos": net_pnl >= 0,
        "total":       len(pnls),
        "winners":     len(winners),
        "losers":      len(losers),
        "win_rate":    f"{win_rate:.1f}%",
        "avg_win":     f"{avg_win:+.4f}",
        "avg_loss":    f"{avg_loss:+.4f}",
        "profit_factor": f"{pf:.2f}",
        "max_win":     f"{float(winners.max()):+.4f}" if len(winners) else "0",
        "max_loss":    f"{float(losers.min()):+.4f}"  if len(losers)  else "0",
        "fast_ema":    FAST_EMA,
        "slow_ema":    SLOW_EMA,
        "sl_pct":      f"{STOP_LOSS*100:.2f}%",
        "years":       YEARS,
        "tf":          TIMEFRAME,
        "tf_label":    _tf["label"],
        # Chip shown when chart display differs from backtest timeframe
        "chart_res_chip": (
            '<span class="chip" style="background:#1a2234;color:#94a3b8">'
            'chart 1H</span>'
            if TIMEFRAME not in ("1H", "4H") else ""
        ),
    }

    return candles, ema_fast_data, ema_slow_data, markers, equity, trades, stats


# ── 5. Generate HTML ──────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMA Cross Dashboard — BTC-USDT-SWAP</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{
  --bg:#0b0e14;--surface:#111827;--surface2:#1a2234;--border:#1f2d45;
  --text:#e2e8f0;--muted:#64748b;--accent:#3b82f6;
  --green:#10b981;--red:#ef4444;--yellow:#f59e0b;--purple:#8b5cf6;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow-x:hidden}
body{background:var(--bg);color:var(--text);
     font-family:'Inter','Segoe UI',system-ui,sans-serif;font-size:13px}

/* ── topbar ── */
#topbar{
  display:flex;align-items:center;gap:10px;
  padding:10px 16px;background:var(--surface);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
}
#topbar .logo{font-size:15px;font-weight:800;color:#fff;letter-spacing:-.3px}
#topbar .logo span{color:var(--accent)}
.chip{
  padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;
  letter-spacing:.5px;text-transform:uppercase;
  background:#1e3a5f;color:#60a5fa;border:1px solid #1e3a8a;
}
.chip.red{background:#3b0f0f;color:#f87171;border-color:#7f1d1d}
.chip.green{background:#052e16;color:#34d399;border-color:#065f46}
#topbar .spacer{flex:1}
#topbar .ts{font-size:11px;color:var(--muted)}
#fit-btn{
  padding:5px 12px;border-radius:6px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);cursor:pointer;
  font-size:11px;font-weight:600;transition:background .15s;
}
#fit-btn:hover{background:var(--border)}

/* ── stat strip ── */
#stats-strip{
  display:flex;gap:1px;background:var(--border);
  border-bottom:1px solid var(--border);overflow-x:auto;
}
#stats-strip::-webkit-scrollbar{height:3px}
.stat-card{
  flex:1;min-width:120px;padding:10px 16px;background:var(--surface);
  display:flex;flex-direction:column;gap:3px;
}
.stat-card .lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.stat-card .val{font-size:19px;font-weight:800;color:var(--text)}
.stat-card .sub{font-size:10px;color:var(--muted)}
.green{color:var(--green)!important}
.red  {color:var(--red)!important}
.blue {color:var(--accent)!important}

/* win/loss bar */
.wl-bar{height:3px;border-radius:2px;background:var(--border);margin-top:4px;overflow:hidden}
.wl-bar .fill{height:100%;background:var(--green);border-radius:2px;transition:width .4s}

/* ── layout ── */
#body{display:flex;height:calc(100vh - 96px)}
#left{flex:1;min-width:0;display:flex;flex-direction:column;border-right:1px solid var(--border)}
#right{width:340px;display:flex;flex-direction:column;background:var(--surface)}

/* ── chart containers ── */
.pane-header{
  display:flex;align-items:center;gap:12px;
  padding:7px 14px;background:var(--surface2);
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.pane-header .title{font-size:11px;font-weight:600;color:var(--text)}
.legend-item{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted)}
.legend-line{width:18px;height:2px;border-radius:1px}
#main-wrap{flex:1;min-height:0;position:relative}
#main-chart{position:absolute;inset:0}
#eq-wrap{height:130px;position:relative;flex-shrink:0}
#eq-chart{position:absolute;inset:0}

/* ── right panel: trade list ── */
.panel-hdr{
  padding:10px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.panel-hdr .title{font-size:12px;font-weight:700;color:var(--text)}
.panel-hdr .count{font-size:10px;color:var(--muted)}
#filters{
  display:flex;gap:4px;padding:8px 10px;flex-shrink:0;
  border-bottom:1px solid var(--border);background:var(--surface2);
}
.flt-btn{
  flex:1;padding:5px 0;border-radius:5px;border:1px solid var(--border);
  background:transparent;color:var(--muted);cursor:pointer;font-size:10px;
  font-weight:600;text-transform:uppercase;letter-spacing:.4px;transition:all .15s;
}
.flt-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.flt-btn:hover:not(.active){background:var(--border);color:var(--text)}
#trade-list{flex:1;overflow-y:auto;padding:4px}
#trade-list::-webkit-scrollbar{width:4px}
#trade-list::-webkit-scrollbar-track{background:transparent}
#trade-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.trade-card{
  border-radius:8px;border:1px solid var(--border);
  background:var(--surface2);margin-bottom:4px;
  padding:9px 12px;cursor:pointer;transition:border-color .15s;
}
.trade-card:hover{border-color:var(--accent)}
.trade-card.win-card{border-left:3px solid var(--green)}
.trade-card.loss-card{border-left:3px solid var(--red)}
.tc-row1{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.tc-num{font-size:10px;color:var(--muted)}
.tc-side{font-size:11px;font-weight:800;letter-spacing:.3px}
.tc-side.l{color:var(--green)}
.tc-side.s{color:var(--red)}
.tc-pnl{font-size:14px;font-weight:800}
.tc-row2{display:flex;gap:10px}
.tc-info{font-size:10px;color:var(--muted)}
.tc-info span{color:var(--text)}
.tc-exit{padding:1px 6px;border-radius:3px;font-size:9px;font-weight:700;margin-left:auto}
.tc-exit.sl {background:#3b0f0f;color:#f87171}
.tc-exit.ema{background:#1e3a5f;color:#60a5fa}

/* ── crosshair info box ── */
#info-box{
  position:absolute;top:8px;left:12px;z-index:50;
  background:#111827ee;border:1px solid var(--border);
  border-radius:7px;padding:8px 12px;font-size:11px;
  pointer-events:none;display:none;min-width:210px;
}
#info-box .ib-row{display:flex;justify-content:space-between;gap:16px;margin-bottom:2px}
#info-box .ib-lbl{color:var(--muted)}
#info-box .ib-val{font-weight:700;text-align:right}

/* scrollbar global */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <span class="logo">EMA<span>Cross</span></span>
  <span class="chip">BTC-USDT-SWAP</span>
  <span class="chip">__TF__ bars</span>
  <span class="chip">EMA __FAST__ / __SLOW__</span>
  <span class="chip red">SL __SL_PCT__</span>
  <span class="chip green">Trailing</span>
  __CHART_RES_CHIP__
  <span class="spacer"></span>
  <span class="ts">__YEARS__-year backtest</span>
  <button id="fit-btn">Fit All</button>
</div>

<!-- Stats strip -->
<div id="stats-strip">
  <div class="stat-card">
    <div class="lbl">Net P&amp;L</div>
    <div class="val __PNL_CLASS__">__NET_PNL__</div>
    <div class="sub">__NET_PCT__ return</div>
  </div>
  <div class="stat-card">
    <div class="lbl">End Balance</div>
    <div class="val blue">__END_BAL__</div>
    <div class="sub">started __START_BAL__ USDT</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Win Rate</div>
    <div class="val">__WIN_RATE__</div>
    <div class="wl-bar"><div class="fill" id="wr-fill"></div></div>
    <div class="sub">__WINNERS__ W &nbsp;/&nbsp; __LOSERS__ L</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Avg Win</div>
    <div class="val green">__AVG_WIN__</div>
    <div class="sub">USDT per winning trade</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Avg Loss</div>
    <div class="val red">__AVG_LOSS__</div>
    <div class="sub">USDT per losing trade</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Profit Factor</div>
    <div class="val __PF_CLASS__">__PF__</div>
    <div class="sub">&gt; 1.0 is profitable</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Best Trade</div>
    <div class="val green">__MAX_WIN__</div>
    <div class="sub">USDT</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Worst Trade</div>
    <div class="val red">__MAX_LOSS__</div>
    <div class="sub">USDT</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Total Trades</div>
    <div class="val blue">__TOTAL__</div>
    <div class="sub">__YEARS__ years</div>
  </div>
</div>

<!-- Body: chart left, trade list right -->
<div id="body">

  <!-- Left: charts -->
  <div id="left">
    <div class="pane-header">
      <span class="title">BTC-USDT-SWAP &bull; __TF_LABEL__</span>
      <span class="legend-item"><span class="legend-line" style="background:#2563eb"></span>EMA __FAST__</span>
      <span class="legend-item"><span class="legend-line" style="background:#f59e0b"></span>EMA __SLOW__</span>
      <span class="legend-item" style="margin-left:8px;font-size:10px;color:#10b981">&#9650; Long entry</span>
      <span class="legend-item" style="font-size:10px;color:#ef4444">&#9660; Short entry</span>
      <span class="legend-item" style="font-size:10px;color:#10b981">&#11044; Win exit</span>
      <span class="legend-item" style="font-size:10px;color:#ef4444">&#11044; Loss exit</span>
    </div>
    <div id="main-wrap">
      <div id="main-chart"></div>
      <div id="info-box">
        <div class="ib-row"><span class="ib-lbl">Time</span><span class="ib-val" id="ib-time">—</span></div>
        <div class="ib-row"><span class="ib-lbl">Open</span><span class="ib-val" id="ib-o">—</span></div>
        <div class="ib-row"><span class="ib-lbl">High</span><span class="ib-val green" id="ib-h">—</span></div>
        <div class="ib-row"><span class="ib-lbl">Low</span> <span class="ib-val red"   id="ib-l">—</span></div>
        <div class="ib-row"><span class="ib-lbl">Close</span><span class="ib-val" id="ib-c">—</span></div>
        <div class="ib-row"><span class="ib-lbl">EMA __FAST__</span><span class="ib-val blue" id="ib-ef">—</span></div>
        <div class="ib-row"><span class="ib-lbl">EMA __SLOW__</span><span class="ib-val" style="color:var(--yellow)" id="ib-es">—</span></div>
      </div>
    </div>

    <div class="pane-header" style="border-top:1px solid var(--border)">
      <span class="title">Equity Curve</span>
      <span class="legend-item"><span class="legend-line" style="background:#3b82f6"></span>Account Balance (USDT)</span>
    </div>
    <div id="eq-wrap">
      <div id="eq-chart"></div>
    </div>
  </div>

  <!-- Right: trade list -->
  <div id="right">
    <div class="panel-hdr">
      <span class="title">Trade Log</span>
      <span class="count" id="list-count">__TOTAL__ trades</span>
    </div>
    <div id="filters">
      <button class="flt-btn active" data-f="all">All</button>
      <button class="flt-btn" data-f="long">Long</button>
      <button class="flt-btn" data-f="short">Short</button>
      <button class="flt-btn" data-f="win">Wins</button>
      <button class="flt-btn" data-f="loss">Losses</button>
    </div>
    <div id="trade-list"></div>
  </div>

</div>

<script>
// ── Data injected by Python ────────────────────────────────────────────
const CANDLES  = __CANDLES__;
const EMA_FAST = __EMA_FAST__;
const EMA_SLOW = __EMA_SLOW__;
const MARKERS  = __MARKERS__;
const EQUITY   = __EQUITY__;
const TRADES   = __TRADES__;
const WIN_RATE_NUM = __WIN_RATE_NUM__;

// ── Win/loss bar fill ──────────────────────────────────────────────────
document.getElementById("wr-fill").style.width = WIN_RATE_NUM + "%";

// ── Profit factor colour ───────────────────────────────────────────────
// (already injected via __PF_CLASS__)

// ── Shared chart options ───────────────────────────────────────────────
const CHART_OPTS = {
  layout:{
    background:{type:LightweightCharts.ColorType.Solid, color:"#0b0e14"},
    textColor:"#64748b",
    fontSize:11,
  },
  grid:{vertLines:{color:"#1a2234"}, horzLines:{color:"#1a2234"}},
  crosshair:{
    mode:LightweightCharts.CrosshairMode.Normal,
    vertLine:{color:"#3b82f6",width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:"#1e3a5f"},
    horzLine:{color:"#3b82f6",width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:"#1e3a5f"},
  },
  rightPriceScale:{borderColor:"#1f2d45", scaleMargins:{top:.08,bottom:.08}},
  timeScale:{
    borderColor:"#1f2d45",timeVisible:true,secondsVisible:false,
    rightOffset:12, barSpacing:3, minBarSpacing:1,
    fixLeftEdge:false, fixRightEdge:false,
  },
  // autoSize = no manual resize needed → no blink
  autoSize:true,
};

// ── Main chart ──────────────────────────────────────────────────────────
const mainEl    = document.getElementById("main-chart");
const mainChart = LightweightCharts.createChart(mainEl, CHART_OPTS);

const candleSeries = mainChart.addCandlestickSeries({
  upColor:"#10b981",  downColor:"#ef4444",
  borderUpColor:"#10b981",  borderDownColor:"#ef4444",
  wickUpColor:"#10b981",    wickDownColor:"#ef4444",
});
candleSeries.setData(CANDLES);

const emaFastSeries = mainChart.addLineSeries({
  color:"#2563eb",lineWidth:1,
  priceLineVisible:false,lastValueVisible:true,
  crosshairMarkerVisible:false,
});
emaFastSeries.setData(EMA_FAST);

const emaSlowSeries = mainChart.addLineSeries({
  color:"#f59e0b",lineWidth:1,
  priceLineVisible:false,lastValueVisible:true,
  crosshairMarkerVisible:false,
});
emaSlowSeries.setData(EMA_SLOW);

candleSeries.setMarkers(MARKERS);

// ── Equity chart ────────────────────────────────────────────────────────
const eqEl    = document.getElementById("eq-chart");
const eqChart = LightweightCharts.createChart(eqEl, {
  ...CHART_OPTS,
  rightPriceScale:{...CHART_OPTS.rightPriceScale, scaleMargins:{top:.1,bottom:.1}},
});

const startBal = EQUITY.length ? EQUITY[0].value : 10000;
const eqSeries = eqChart.addAreaSeries({
  lineColor:"#3b82f6",
  topColor:"#3b82f620",
  bottomColor:"#3b82f600",
  lineWidth:2,
  priceLineVisible:false,
  lastValueVisible:true,
  crosshairMarkerVisible:true,
  crosshairMarkerRadius:4,
  crosshairMarkerBorderColor:"#3b82f6",
  crosshairMarkerBackgroundColor:"#0b0e14",
});
eqSeries.setData(EQUITY);

// Baseline (starting balance)
eqSeries.createPriceLine({
  price:startBal, color:"#64748b",
  lineWidth:1, lineStyle:LightweightCharts.LineStyle.Dashed,
  axisLabelVisible:false,
});

// ── Fit on load ─────────────────────────────────────────────────────────
mainChart.timeScale().fitContent();
eqChart.timeScale().fitContent();

// ── Sync time scales (mutex prevents infinite loop) ─────────────────────
let syncing = false;

mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
  if (syncing || !range) return;
  syncing = true;
  eqChart.timeScale().setVisibleLogicalRange(range);
  syncing = false;
});
eqChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
  if (syncing || !range) return;
  syncing = true;
  mainChart.timeScale().setVisibleLogicalRange(range);
  syncing = false;
});

// ── OHLC info box ────────────────────────────────────────────────────────
const infoBox = document.getElementById("info-box");
const fmt     = n => n == null ? "—" : n.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1});

// Build fast lookup maps for EMA values
const efMap = new Map(EMA_FAST.map(d => [d.time, d.value]));
const esMap = new Map(EMA_SLOW.map(d => [d.time, d.value]));

mainChart.subscribeCrosshairMove(param => {
  if (!param.time || !param.point) { infoBox.style.display = "none"; return; }
  const ohlc = param.seriesData.get(candleSeries);
  if (!ohlc) { infoBox.style.display = "none"; return; }
  infoBox.style.display = "block";
  const d = new Date(param.time * 1000);
  document.getElementById("ib-time").textContent =
    d.toUTCString().slice(5,22);
  document.getElementById("ib-o").textContent = fmt(ohlc.open);
  document.getElementById("ib-h").textContent = fmt(ohlc.high);
  document.getElementById("ib-l").textContent = fmt(ohlc.low);
  document.getElementById("ib-c").textContent = fmt(ohlc.close);
  document.getElementById("ib-ef").textContent = fmt(efMap.get(param.time) ?? null);
  document.getElementById("ib-es").textContent = fmt(esMap.get(param.time) ?? null);
});

// ── Fit button ───────────────────────────────────────────────────────────
document.getElementById("fit-btn").addEventListener("click", () => {
  mainChart.timeScale().fitContent();
  eqChart.timeScale().fitContent();
});

// ── Trade list (right panel) ─────────────────────────────────────────────
const listEl    = document.getElementById("trade-list");
const listCount = document.getElementById("list-count");

function renderList(filter) {
  listEl.innerHTML = "";
  const visible = TRADES.filter(t => {
    if (filter === "long")  return t.side === "BUY";
    if (filter === "short") return t.side === "SELL";
    if (filter === "win")   return t.win;
    if (filter === "loss")  return !t.win;
    return true;
  });
  listCount.textContent = visible.length + " trades";
  visible.forEach(t => {
    const card = document.createElement("div");
    card.className = "trade-card " + (t.win ? "win-card" : "loss-card");
    const pnlStr  = (t.pnl >= 0 ? "+" : "") + t.pnl.toFixed(4);
    const sideStr = t.side === "BUY" ? "LONG" : "SHORT";
    card.innerHTML = `
      <div class="tc-row1">
        <span class="tc-num">#${t.num}</span>
        <span class="tc-side ${t.side==='BUY'?'l':'s'}">${sideStr}</span>
        <span class="tc-pnl ${t.win?'green':'red'}">${pnlStr} <small style="font-weight:400;font-size:10px">USDT</small></span>
      </div>
      <div class="tc-row2">
        <span class="tc-info">In <span>${t.entry_ts.slice(5,16)}</span></span>
        <span class="tc-info">Out <span>${t.exit_ts.slice(5,16)}</span></span>
        <span class="tc-info">${t.duration}</span>
        <span class="tc-exit ${t.exit_type==='SL'?'sl':'ema'}">${t.exit_type}</span>
      </div>
      <div style="display:flex;gap:10px;margin-top:4px">
        <span class="tc-info">Entry <span style="font-family:monospace">${t.entry_px.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1})}</span></span>
        <span class="tc-info">Exit <span style="font-family:monospace">${t.exit_px.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1})}</span></span>
      </div>`;

    // Click: zoom chart to this trade
    card.addEventListener("click", () => {
      const t0 = Math.floor(new Date(t.entry_ts).getTime()/1000) - 300;
      const t1 = Math.floor(new Date(t.exit_ts ).getTime()/1000) + 300;
      mainChart.timeScale().setVisibleRange({from:t0, to:t1});
    });
    listEl.appendChild(card);
  });
}

renderList("all");

document.querySelectorAll(".flt-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".flt-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderList(btn.dataset.f);
  });
});
</script>
</body>
</html>
"""


def fill_template(template: str, candles, ema_fast, ema_slow,
                  markers, equity, trades, stats) -> str:
    html = template
    # JSON data
    html = html.replace("__CANDLES__",  json.dumps(candles))
    html = html.replace("__EMA_FAST__", json.dumps(ema_fast))
    html = html.replace("__EMA_SLOW__", json.dumps(ema_slow))
    html = html.replace("__MARKERS__",  json.dumps(markers))
    html = html.replace("__EQUITY__",   json.dumps(equity))
    html = html.replace("__TRADES__",   json.dumps(trades))
    # Numeric values for JS
    wr_num = float(stats["win_rate"].replace("%", ""))
    html = html.replace("__WIN_RATE_NUM__", str(round(wr_num, 1)))
    pf    = float(stats["profit_factor"])
    html = html.replace("__PF_CLASS__", "green" if pf >= 1 else "red")
    # Stats text (order matters — replace longer tokens first)
    html = html.replace("__NET_PCT__",   stats["net_pnl_pct"])
    html = html.replace("__NET_PNL__",   stats["net_pnl"])
    html = html.replace("__PNL_CLASS__", "green" if stats["net_pnl_pos"] else "red")
    html = html.replace("__END_BAL__",   stats["end_bal"])
    html = html.replace("__START_BAL__", stats["start_bal"])
    html = html.replace("__WINNERS__",   str(stats["winners"]))
    html = html.replace("__LOSERS__",    str(stats["losers"]))
    html = html.replace("__TOTAL__",     str(stats["total"]))
    html = html.replace("__WIN_RATE__",  stats["win_rate"])
    html = html.replace("__AVG_WIN__",   stats["avg_win"])
    html = html.replace("__AVG_LOSS__",  stats["avg_loss"])
    html = html.replace("__PF__",        stats["profit_factor"])
    html = html.replace("__MAX_WIN__",   stats["max_win"])
    html = html.replace("__MAX_LOSS__",  stats["max_loss"])
    html = html.replace("__FAST__",      str(stats["fast_ema"]))
    html = html.replace("__SLOW__",      str(stats["slow_ema"]))
    html = html.replace("__SL_PCT__",    stats["sl_pct"])
    html = html.replace("__YEARS__",     str(stats["years"]))
    html = html.replace("__TF_LABEL__",       stats["tf_label"])
    html = html.replace("__TF__",             stats["tf"])
    html = html.replace("__CHART_RES_CHIP__", stats["chart_res_chip"])
    return html


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EMA Cross — Professional Chart Dashboard")
    print(f"  EMA {FAST_EMA}/{SLOW_EMA}  |  SL {STOP_LOSS*100:.1f}%  |  {YEARS} years  |  {TIMEFRAME} bars")
    print("=" * 60 + "\n")

    raw_1m = fetch(YEARS)
    raw    = aggregate_raw(raw_1m, TIMEFRAME)
    inst   = build_instrument()
    bars   = build_bars(raw, inst)
    engine = run_backtest(bars, inst)

    print("Building chart data …")
    candles, ema_fast, ema_slow, markers, equity, trades, stats = \
        build_chart_data(raw, engine)

    html = fill_template(HTML_TEMPLATE, candles, ema_fast, ema_slow,
                         markers, equity, trades, stats)

    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Dashboard saved -> {OUTPUT_FILE}")
    print(f"Opening in browser …\n")
    webbrowser.open(OUTPUT_FILE.as_uri())

    engine.reset()
    engine.dispose()
