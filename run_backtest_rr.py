#!/usr/bin/env python3
"""
EMA Cross WITH Risk:Reward Rules — BTC-USDT-SWAP Backtest
==========================================================
Same 30-day OKX data as run_backtest_btc.py, but the strategy now uses
a proper bracket order on every trade:

    Stop-Loss   = entry ± STOP_LOSS_PCT          (default 0.5%)
    Take-Profit = entry ± STOP_LOSS_PCT × RR     (default 1.0% → 2:1 R:R)

Run:
    .venv\\Scripts\\python.exe run_backtest_rr.py

Adjust the settings below to experiment with different R:R values.
"""

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
    AccountType, AggregationSource, BarAggregation,
    OmsType, PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

from strategies.ema_cross_rr import EMACrossRR, EMACrossRRConfig

# ── ★  EDIT THESE  ★ ──────────────────────────────────────────────────
DAYS_HISTORY  = 30            # days of 1-minute bars to fetch

FAST_EMA      = 100
SLOW_EMA      = 200

TRADE_SIZE    = Decimal("0.01")   # 0.01 BTC per trade
START_BALANCE = 10_000            # USDT

# Risk management parameters — try different values here!
STOP_LOSS_PCT = 0.005   # 0.5% stop loss (e.g. $400 risk on a $80k BTC trade)
RISK_REWARD   = 2.0     # 2:1 reward:risk  →  TP = 1.0% away from entry

# Other R:R presets to try:
#   Conservative:  STOP_LOSS_PCT=0.003, RISK_REWARD=3.0  → tiny SL, big TP
#   Aggressive:    STOP_LOSS_PCT=0.010, RISK_REWARD=1.5  → wide SL, medium TP
#   Scalping:      STOP_LOSS_PCT=0.002, RISK_REWARD=1.0  → tight SL, quick TP
# ──────────────────────────────────────────────────────────────────────


# ── Step 1: Fetch OKX candles ─────────────────────────────────────────

def fetch_okx_candles(inst_id: str, bar: str, days: int) -> list[list]:
    url     = "https://www.okx.com/api/v5/market/history-candles"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    target  = days * 24 * 60
    all_bars: list[list] = []
    after   = ""

    print(f"Downloading {target:,} candles ({days} days of {bar} bars)...")
    with httpx.Client(headers=headers, timeout=15) as client:
        while len(all_bars) < target:
            params = {"instId": inst_id, "bar": bar, "limit": "100"}
            if after:
                params["after"] = after
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data["code"] != "0" or not data["data"]:
                break
            batch = data["data"]
            all_bars.extend(batch)
            after = batch[-1][0]
            print(f"  {len(all_bars):>6,} / {target:,}", end="\r")
            if len(batch) < 100:
                break
            time.sleep(0.05)

    all_bars.reverse()
    print(f"\nDownloaded {len(all_bars):,} candles")
    return all_bars[:target]


# ── Step 2: Build NautilusTrader objects ──────────────────────────────

def build_instrument() -> CryptoPerpetual:
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol("BTC-USDT-SWAP"), Venue("OKX")),
        raw_symbol=Symbol("BTC-USDT-SWAP"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=1,
        size_precision=2,
        price_increment=Price(0.1, precision=1),
        size_increment=Quantity(0.01, precision=2),
        multiplier=Quantity(1, precision=0),
        lot_size=Quantity(0.01, precision=2),
        max_quantity=Quantity(10_000, precision=2),
        min_quantity=Quantity(0.01, precision=2),
        max_notional=None,
        min_notional=Money(1, USDT),
        max_price=Price(10_000_000, precision=1),
        min_price=Price(0.1, precision=1),
        margin_init=Decimal("0.02"),
        margin_maint=Decimal("0.01"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
        ts_event=0,
        ts_init=0,
    )


def build_bars(raw: list[list], instrument: CryptoPerpetual) -> list[Bar]:
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    bars = []
    for row in raw:
        ts_ns = int(row[0]) * 1_000_000
        bars.append(Bar(
            bar_type=bar_type,
            open=Price(float(row[1]), precision=1),
            high=Price(float(row[2]), precision=1),
            low=Price(float(row[3]), precision=1),
            close=Price(float(row[4]), precision=1),
            volume=Quantity(float(row[5]), precision=2),
            ts_event=ts_ns,
            ts_init=ts_ns,
        ))
    return bars


# ── Step 3: Run backtest ──────────────────────────────────────────────

def run(bars: list[Bar], instrument: CryptoPerpetual) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("RR-BACKTESTER-001"),
        logging=LoggingConfig(log_level="WARNING", log_colors=True),  # quieter logs
    ))

    OKX_SIM = Venue("OKX")
    engine.add_venue(
        venue=OKX_SIM,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(START_BALANCE, USDT)],
        fill_model=FillModel(
            prob_fill_on_limit=0.95,   # TP limit orders almost always fill
            prob_slippage=0.3,
            random_seed=42,
        ),
        default_leverage=Decimal("10"),
    )

    engine.add_instrument(instrument)
    engine.add_data(bars)
    print(f"Loaded {len(bars):,} bars into backtest engine")

    bar_type_str = "BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL"
    strategy = EMACrossRR(config=EMACrossRRConfig(
        instrument_id=instrument.id,
        bar_type=BarType.from_str(bar_type_str),
        trade_size=TRADE_SIZE,
        fast_ema_period=FAST_EMA,
        slow_ema_period=SLOW_EMA,
        stop_loss_pct=STOP_LOSS_PCT,
        risk_reward=RISK_REWARD,
        subscribe_quote_ticks=False,
        subscribe_trade_ticks=False,
        request_bars=False,
        close_positions_on_stop=True,
    ))
    engine.add_strategy(strategy)

    print("\nRunning backtest...\n")
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0
    print(f"\nBacktest completed in {elapsed:.2f}s\n")
    return engine


# ── Step 4: Summarise results ─────────────────────────────────────────

def summarise(engine: BacktestEngine) -> None:
    OKX_SIM = Venue("OKX")

    # --- Account ---
    acct = engine.trader.generate_account_report(OKX_SIM)
    start_bal = float(acct["total"].iloc[0])
    end_bal   = float(acct["total"].iloc[-1])
    pnl       = end_bal - start_bal
    pct       = pnl / start_bal * 100

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Period        : {DAYS_HISTORY} days  (Apr 15 – May 15 2026)")
    print(f"  Strategy      : EMA({FAST_EMA}/{SLOW_EMA}) + SL/TP")
    print(f"  Stop-Loss     : {STOP_LOSS_PCT*100:.2f}%  per trade")
    print(f"  Take-Profit   : {STOP_LOSS_PCT*RISK_REWARD*100:.2f}%  per trade  (R:R = 1:{RISK_REWARD})")
    print(f"  Starting bal  : {start_bal:>12,.2f} USDT")
    print(f"  Ending bal    : {end_bal:>12,.2f} USDT")
    print(f"  Net P&L       : {pnl:>+12,.2f} USDT  ({pct:+.2f}%)")
    print("=" * 60)

    # --- Positions stats ---
    pos_df = engine.trader.generate_positions_report()
    if not pos_df.empty:
        closed = pos_df[pos_df["side"] == "FLAT"]
        winners = closed[closed["realized_pnl"].str.replace(" USDT","").astype(float) > 0]
        losers  = closed[closed["realized_pnl"].str.replace(" USDT","").astype(float) < 0]
        total_trades = len(closed)
        win_rate     = len(winners) / total_trades * 100 if total_trades else 0

        # Average win / loss
        avg_win  = closed.loc[winners.index, "realized_pnl"].str.replace(" USDT","").astype(float).mean() if len(winners) else 0
        avg_loss = closed.loc[losers.index,  "realized_pnl"].str.replace(" USDT","").astype(float).mean() if len(losers)  else 0

        print(f"  Total trades  : {total_trades}")
        print(f"  Winners       : {len(winners)}  ({win_rate:.1f}%)")
        print(f"  Losers        : {len(losers)}")
        print(f"  Avg win       : {avg_win:>+.4f} USDT")
        print(f"  Avg loss      : {avg_loss:>+.4f} USDT")
        if avg_loss != 0:
            actual_rr = abs(avg_win / avg_loss)
            print(f"  Actual R:R    : 1:{actual_rr:.2f}  (target 1:{RISK_REWARD})")
        print("=" * 60)

    # --- Compare to no-SL/TP (plain EMACross result) ---
    print()
    print("  COMPARISON (same period, EMA 10/20 no SL/TP):")
    print("  EMA(10/20) no SL/TP    : -1,772 USDT  (-17.7%)")
    print(f"  With SL/TP {STOP_LOSS_PCT*100:.1f}% / R:R {RISK_REWARD}: {pnl:>+,.0f} USDT  ({pct:+.1f}%)")
    print()

    # --- Verbose reports ---
    print("=" * 60)
    print("ORDER FILLS (last 10)")
    print("=" * 60)
    fills = engine.trader.generate_order_fills_report()
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(fills.tail(10).to_string())

    print("\n" + "=" * 60)
    print("POSITIONS (last 10)")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(pos_df.tail(10).to_string())

    # --- Tearsheet ---
    print("\n" + "=" * 60)
    print("Generating tearsheet...")
    output_path = Path("tearsheet_rr.html").resolve()
    try:
        from nautilus_trader.analysis import TearsheetConfig, create_tearsheet
        create_tearsheet(
            engine=engine,
            output_path=str(output_path),
            config=TearsheetConfig(theme="nautilus_dark"),
        )
        print(f"Tearsheet saved -> {output_path}")
        webbrowser.open(output_path.as_uri())
    except Exception as e:
        print(f"Tearsheet skipped: {e}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BTC-USDT-SWAP  EMA Cross + Risk:Reward Backtest")
    print(f"  Data      : {DAYS_HISTORY} days × 1-minute bars (OKX)")
    print(f"  EMA       : fast={FAST_EMA}  slow={SLOW_EMA}")
    print(f"  Trade size: {TRADE_SIZE} BTC")
    print(f"  Capital   : {START_BALANCE:,} USDT  (10x leverage)")
    print(f"  Stop-Loss : {STOP_LOSS_PCT*100:.2f}%")
    print(f"  Take-Prof : {STOP_LOSS_PCT*RISK_REWARD*100:.2f}%  (R:R = 1:{RISK_REWARD})")
    print("=" * 60 + "\n")

    raw        = fetch_okx_candles("BTC-USDT-SWAP", "1m", DAYS_HISTORY)
    instrument = build_instrument()
    bars       = build_bars(raw, instrument)
    engine     = run(bars, instrument)

    summarise(engine)

    engine.reset()
    engine.dispose()
