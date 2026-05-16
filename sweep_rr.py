#!/usr/bin/env python3
"""
Parameter sweep: EMA + SL/TP across timeframes and R:R settings.
Run: python sweep_rr.py
"""
import sys, time
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import httpx
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AccountType, AggregationSource, BarAggregation, OmsType, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from strategies.ema_cross_rr import EMACrossRR, EMACrossRRConfig


def fetch_okx_candles(inst_id, bar_str, days):
    url = "https://www.okx.com/api/v5/market/history-candles"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    bars_per_day = {"1m": 1440, "5m": 288, "15m": 96}
    target = days * bars_per_day[bar_str]
    all_bars = []
    after = ""
    with httpx.Client(headers=headers, timeout=15) as client:
        while len(all_bars) < target:
            params = {"instId": inst_id, "bar": bar_str, "limit": "100"}
            if after:
                params["after"] = after
            resp = client.get(url, params=params)
            data = resp.json()
            if data["code"] != "0" or not data["data"]:
                break
            batch = data["data"]
            all_bars.extend(batch)
            after = batch[-1][0]
            if len(batch) < 100:
                break
            time.sleep(0.05)
    all_bars.reverse()
    return all_bars[:target]


def build_instrument():
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol("BTC-USDT-SWAP"), Venue("OKX")),
        raw_symbol=Symbol("BTC-USDT-SWAP"),
        base_currency=BTC, quote_currency=USDT, settlement_currency=USDT,
        is_inverse=False, price_precision=1, size_precision=2,
        price_increment=Price(0.1, precision=1), size_increment=Quantity(0.01, precision=2),
        multiplier=Quantity(1, precision=0), lot_size=Quantity(0.01, precision=2),
        max_quantity=Quantity(10_000, precision=2), min_quantity=Quantity(0.01, precision=2),
        max_notional=None, min_notional=Money(1, USDT),
        max_price=Price(10_000_000, precision=1), min_price=Price(0.1, precision=1),
        margin_init=Decimal("0.02"), margin_maint=Decimal("0.01"),
        maker_fee=Decimal("0.0002"), taker_fee=Decimal("0.0005"),
        ts_event=0, ts_init=0,
    )


def build_bars(raw, instrument, bar_step):
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(bar_step, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    bars = []
    for row in raw:
        ts_ns = int(row[0]) * 1_000_000
        bars.append(Bar(
            bar_type=bar_type,
            open=Price(float(row[1]), precision=1), high=Price(float(row[2]), precision=1),
            low=Price(float(row[3]), precision=1), close=Price(float(row[4]), precision=1),
            volume=Quantity(float(row[5]), precision=2), ts_event=ts_ns, ts_init=ts_ns,
        ))
    return bars


def run_test(label, bar_type_str, bars, instrument, fast, slow, sl_pct, rr):
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("TEST-001"),
        logging=LoggingConfig(log_level="ERROR", bypass_logging=True),
    ))
    engine.add_venue(
        venue=Venue("OKX"), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=USDT, starting_balances=[Money(10_000, USDT)],
        fill_model=FillModel(prob_fill_on_limit=0.95, prob_slippage=0.3, random_seed=42),
        default_leverage=Decimal("10"),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    strategy = EMACrossRR(config=EMACrossRRConfig(
        instrument_id=instrument.id,
        bar_type=BarType.from_str(bar_type_str),
        trade_size=Decimal("0.01"),
        fast_ema_period=fast, slow_ema_period=slow,
        stop_loss_pct=sl_pct, risk_reward=rr,
    ))
    engine.add_strategy(strategy)
    engine.run()

    acct = engine.trader.generate_account_report(Venue("OKX"))
    end_bal = float(acct["total"].iloc[-1])
    pnl = end_bal - 10_000
    pct = pnl / 10_000 * 100

    pos_df = engine.trader.generate_positions_report()
    closed = pos_df[pos_df["side"] == "FLAT"] if not pos_df.empty else pos_df
    total = len(closed)
    if total > 0:
        pnls = closed["realized_pnl"].str.replace(" USDT", "").astype(float)
        winners = int((pnls > 0).sum())
        win_rate = winners / total * 100
        avg_w = float(pnls[pnls > 0].mean()) if winners > 0 else 0.0
        avg_l = float(pnls[pnls < 0].mean()) if int((pnls < 0).sum()) > 0 else 0.0
    else:
        total, win_rate, avg_w, avg_l = 0, 0.0, 0.0, 0.0

    marker = " <-- BEST" if pnl > 0 else ""
    print(f"  {label:<42}  P&L: {pnl:>+8,.0f} USDT ({pct:>+6.1f}%)  "
          f"trades={total:>4}  WR={win_rate:>4.0f}%  "
          f"avgW={avg_w:>+6.2f}  avgL={avg_l:>+6.2f}{marker}")
    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    instrument = build_instrument()

    print("Fetching 1-minute bars (30 days)...")
    raw_1m  = fetch_okx_candles("BTC-USDT-SWAP", "1m",  30)
    bars_1m = build_bars(raw_1m, instrument, 1)
    print(f"  {len(bars_1m):,} bars")

    print("Fetching 5-minute bars (30 days)...")
    raw_5m  = fetch_okx_candles("BTC-USDT-SWAP", "5m",  30)
    bars_5m = build_bars(raw_5m, instrument, 5)
    print(f"  {len(bars_5m):,} bars")

    print("Fetching 15-minute bars (30 days)...")
    raw_15m  = fetch_okx_candles("BTC-USDT-SWAP", "15m", 30)
    bars_15m = build_bars(raw_15m, instrument, 15)
    print(f"  {len(bars_15m):,} bars\n")

    print("=" * 90)
    print("  PARAMETER SWEEP — EMA Cross + SL/TP  (30-day BTC-USDT-SWAP, 10,000 USDT, 10x)")
    print("=" * 90)

    # (label, bar_type_str, bars, fast, slow, sl_pct, rr)
    configs = [
        # --- 1-minute baseline ---
        ("1m  EMA(10/20) SL=0.5% RR=2.0 [baseline]",
         "BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL",  bars_1m,  10, 20, 0.005, 2.0),
        ("1m  EMA(10/20) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL",  bars_1m,  10, 20, 0.010, 2.0),
        ("1m  EMA(20/50) SL=0.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL",  bars_1m,  20, 50, 0.005, 2.0),
        ("1m  EMA(20/50) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL",  bars_1m,  20, 50, 0.010, 2.0),
        # --- 5-minute ---
        ("5m  EMA(10/20) SL=0.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  10, 20, 0.005, 2.0),
        ("5m  EMA(10/20) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  10, 20, 0.010, 2.0),
        ("5m  EMA(10/20) SL=1.0% RR=3.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  10, 20, 0.010, 3.0),
        ("5m  EMA(20/50) SL=0.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  20, 50, 0.005, 2.0),
        ("5m  EMA(20/50) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  20, 50, 0.010, 2.0),
        ("5m  EMA(20/50) SL=1.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-5-MINUTE-LAST-EXTERNAL",  bars_5m,  20, 50, 0.015, 2.0),
        # --- 15-minute ---
        ("15m EMA(10/20) SL=0.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 10, 20, 0.005, 2.0),
        ("15m EMA(10/20) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 10, 20, 0.010, 2.0),
        ("15m EMA(10/20) SL=1.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 10, 20, 0.015, 2.0),
        ("15m EMA(20/50) SL=1.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 20, 50, 0.010, 2.0),
        ("15m EMA(20/50) SL=1.0% RR=3.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 20, 50, 0.010, 3.0),
        ("15m EMA(20/50) SL=1.5% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 20, 50, 0.015, 2.0),
        ("15m EMA(20/50) SL=2.0% RR=2.0",
         "BTC-USDT-SWAP.OKX-15-MINUTE-LAST-EXTERNAL", bars_15m, 20, 50, 0.020, 2.0),
    ]

    for label, bar_type_str, bars, fast, slow, sl_pct, rr in configs:
        run_test(label, bar_type_str, bars, instrument, fast, slow, sl_pct, rr)

    print("=" * 90)
