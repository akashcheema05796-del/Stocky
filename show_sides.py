#!/usr/bin/env python3
"""Shows the long vs short trade breakdown from the last backtest config."""
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

# ── fetch data ──────────────────────────────────────────────────────────
url, headers = "https://www.okx.com/api/v5/market/history-candles", {"User-Agent": "Mozilla/5.0"}
raw, after = [], ""
print("Downloading candles...")
with httpx.Client(headers=headers, timeout=15) as c:
    while len(raw) < 43200:
        p = {"instId": "BTC-USDT-SWAP", "bar": "1m", "limit": "100"}
        if after:
            p["after"] = after
        d = c.get(url, params=p).json()
        if d["code"] != "0" or not d["data"]:
            break
        b = d["data"]; raw.extend(b); after = b[-1][0]
        if len(b) < 100:
            break
        time.sleep(0.05)
raw.reverse()

# ── build objects ────────────────────────────────────────────────────────
inst = CryptoPerpetual(
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

bt = BarType(
    instrument_id=inst.id,
    bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
    aggregation_source=AggregationSource.EXTERNAL,
)
bars = [
    Bar(bar_type=bt,
        open=Price(float(r[1]), precision=1), high=Price(float(r[2]), precision=1),
        low=Price(float(r[3]), precision=1),  close=Price(float(r[4]), precision=1),
        volume=Quantity(float(r[5]), precision=2),
        ts_event=int(r[0]) * 1_000_000, ts_init=int(r[0]) * 1_000_000)
    for r in raw[:43200]
]

# ── run backtest ─────────────────────────────────────────────────────────
engine = BacktestEngine(config=BacktestEngineConfig(
    trader_id=TraderId("T-001"),
    logging=LoggingConfig(log_level="ERROR"),
))
engine.add_venue(
    venue=Venue("OKX"), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
    base_currency=USDT, starting_balances=[Money(10_000, USDT)],
    fill_model=FillModel(prob_fill_on_limit=0.95, prob_slippage=0.3, random_seed=42),
    default_leverage=Decimal("10"),
)
engine.add_instrument(inst)
engine.add_data(bars)
engine.add_strategy(EMACrossRR(config=EMACrossRRConfig(
    instrument_id=inst.id,
    bar_type=BarType.from_str("BTC-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL"),
    trade_size=Decimal("0.01"),
    fast_ema_period=100, slow_ema_period=200,
    stop_loss_pct=0.005,
)))
engine.run()

# ── analyse ──────────────────────────────────────────────────────────────
pos = engine.trader.generate_positions_report()
closed = pos[pos["side"] == "FLAT"].copy()
closed["pnl"] = closed["realized_pnl"].str.replace(" USDT", "").astype(float)

longs  = closed[closed["entry"] == "BUY"]
shorts = closed[closed["entry"] == "SELL"]

def show(label, df):
    total   = len(df)
    winners = int((df["pnl"] > 0).sum())
    losers  = int((df["pnl"] < 0).sum())
    wr      = winners / total * 100 if total else 0
    avg     = float(df["pnl"].mean()) if total else 0
    net     = float(df["pnl"].sum())
    print(f"  Count    : {total}")
    print(f"  Winners  : {winners}  ({wr:.0f}%)")
    print(f"  Losers   : {losers}")
    print(f"  Avg P&L  : {avg:>+.4f} USDT per trade")
    print(f"  Net P&L  : {net:>+.2f} USDT")

print()
print("=" * 55)
print("  LONG  trades  (EMA100 crossed ABOVE EMA200 -> BUY)")
print("=" * 55)
show("LONG", longs)

print()
print("=" * 55)
print("  SHORT trades  (EMA100 crossed BELOW EMA200 -> SELL)")
print("=" * 55)
show("SHORT", shorts)

print()
print("=" * 55)
print(f"  COMBINED  :  {len(closed)} trades  "
      f"net {closed['pnl'].sum():>+.2f} USDT")
print("=" * 55)

engine.reset(); engine.dispose()
