#!/usr/bin/env python3
"""
BTC-USDT-SWAP EMA Cross Backtest — Real OKX historical data
============================================================
Downloads 30 days of real 1-minute candles from OKX public API,
then backtests the exact same EMA(10/20) strategy running live.

Run:
    .venv\\Scripts\\python.exe run_backtest_btc.py
"""

import time
import webbrowser
from decimal import Decimal
from pathlib import Path

import httpx
import pandas as pd

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross
from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig
from nautilus_trader.model.currencies import BTC
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

# ── Settings ──────────────────────────────────────────────────────────
DAYS_HISTORY  = 30          # how many days of 1-minute data to download
FAST_EMA      = 10
SLOW_EMA      = 20
TRADE_SIZE    = Decimal("0.01")   # 0.01 BTC per trade
START_BALANCE = 10_000            # USDT
# ──────────────────────────────────────────────────────────────────────


# ── Step 1: Download historical candles from OKX ──────────────────────

def fetch_okx_candles(inst_id: str, bar: str, days: int) -> list[list]:
    """
    Fetch historical OHLCV from OKX public REST API.
    Returns list of [ts_ms, open, high, low, close, vol] sorted oldest→newest.
    """
    url = "https://www.okx.com/api/v5/market/history-candles"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    target = days * 24 * 60   # number of 1-minute bars
    all_bars: list[list] = []
    after = ""                # pagination cursor (oldest ts fetched so far)

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

            batch = data["data"]           # newest first
            all_bars.extend(batch)
            after = batch[-1][0]           # oldest ts in this batch → next page cursor

            fetched = len(all_bars)
            print(f"  {fetched:>6,} / {target:,} bars", end="\r")

            if len(batch) < 100:           # no more data
                break

            time.sleep(0.05)               # gentle rate limiting

    all_bars.reverse()                     # flip to oldest→newest
    print(f"\nDownloaded {len(all_bars):,} candles total")
    return all_bars[:target]              # trim to exactly what we need


# ── Step 2: Build NautilusTrader objects ──────────────────────────────

def build_instrument() -> CryptoPerpetual:
    """Build BTC-USDT-SWAP instrument spec matching OKX contract."""
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
    """Convert OKX raw candle rows to NautilusTrader Bar objects."""
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    bars = []
    for row in raw:
        ts_ns = int(row[0]) * 1_000_000   # ms → ns
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
        trader_id=TraderId("BTC-BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO", log_colors=True),
    ))

    OKX_SIM = Venue("OKX")
    engine.add_venue(
        venue=OKX_SIM,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(START_BALANCE, USDT)],
        fill_model=FillModel(
            prob_fill_on_limit=0.2,
            prob_slippage=0.5,
            random_seed=42,
        ),
        default_leverage=Decimal("10"),   # 10x — same as typical OKX default
    )

    engine.add_instrument(instrument)
    engine.add_data(bars)
    print(f"Loaded {len(bars):,} bars into backtest engine")

    bar_type_str = f"BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL"
    strategy = EMACross(config=EMACrossConfig(
        instrument_id=instrument.id,
        bar_type=BarType.from_str(bar_type_str),
        trade_size=TRADE_SIZE,
        fast_ema_period=FAST_EMA,
        slow_ema_period=SLOW_EMA,
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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BTC-USDT-SWAP EMA Cross Backtest")
    print(f"  Data      : {DAYS_HISTORY} days × 1-minute bars (OKX)")
    print(f"  EMA       : fast={FAST_EMA}  slow={SLOW_EMA}")
    print(f"  Trade size: {TRADE_SIZE} BTC per trade")
    print(f"  Capital   : {START_BALANCE:,} USDT  (10x leverage)")
    print("=" * 60 + "\n")

    # 1. Download data
    raw = fetch_okx_candles("BTC-USDT-SWAP", "1m", DAYS_HISTORY)

    # 2. Build objects
    instrument = build_instrument()
    bars = build_bars(raw, instrument)

    # 3. Run
    engine = run(bars, instrument)
    OKX_SIM = Venue("OKX")

    # 4. Reports
    print("=" * 60)
    print("ACCOUNT REPORT")
    print("=" * 60)
    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_account_report(OKX_SIM))

    print("\n" + "=" * 60)
    print("ORDER FILLS")
    print("=" * 60)
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_order_fills_report())

    print("\n" + "=" * 60)
    print("POSITIONS")
    print("=" * 60)
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_positions_report())

    # 5. Tearsheet
    print("\n" + "=" * 60)
    print("Generating tearsheet...")
    output_path = Path("tearsheet_btc.html").resolve()
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
    engine.reset()
    engine.dispose()
