#!/usr/bin/env python3
import sys
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd

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

# Load + resample once
print("Loading data from cache...")
raw = pd.read_csv("data/BTC_USDT_SWAP_1m_3y.csv.gz", compression="gzip", header=None)
raw.columns = range(len(raw.columns))
raw[0] = raw[0].astype(int)
for c in [1,2,3,4,5]:
    raw[c] = raw[c].astype(float)
raw.index = pd.to_datetime(raw[0], unit="ms", utc=True)
hourly = raw.resample("1h").agg({1:"first",2:"max",3:"min",4:"last",5:"sum"}).dropna(subset=[1])
print(f"  {len(hourly):,} 1H bars ready\n")

def build_engine(sl_pct, be_rr):
    inst = CryptoPerpetual(
        instrument_id=InstrumentId(Symbol("BTC-USDT-SWAP"), Venue("OKX")),
        raw_symbol=Symbol("BTC-USDT-SWAP"),
        base_currency=BTC, quote_currency=USDT, settlement_currency=USDT,
        is_inverse=False, price_precision=1, size_precision=2,
        price_increment=Price(0.1,precision=1), size_increment=Quantity(0.01,precision=2),
        multiplier=Quantity(1,precision=0), lot_size=Quantity(0.01,precision=2),
        max_quantity=Quantity(10_000,precision=2), min_quantity=Quantity(0.01,precision=2),
        max_notional=None, min_notional=Money(1,USDT),
        max_price=Price(10_000_000,precision=1), min_price=Price(0.1,precision=1),
        margin_init=Decimal("0.02"), margin_maint=Decimal("0.01"),
        maker_fee=Decimal("0.0002"), taker_fee=Decimal("0.0005"),
        ts_event=0, ts_init=0)

    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("T-001"),
        logging=LoggingConfig(log_level="ERROR", bypass_logging=True)))
    engine.add_venue(venue=Venue("OKX"), oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN, base_currency=USDT,
        starting_balances=[Money(10_000, USDT)],
        fill_model=FillModel(prob_fill_on_limit=0.95, prob_slippage=0.3, random_seed=42),
        default_leverage=Decimal("10"))
    engine.add_instrument(inst)

    bt = BarType(instrument_id=inst.id,
        bar_spec=BarSpecification(1, BarAggregation.HOUR, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL)
    bars = [Bar(bar_type=bt,
        open=Price(float(r[1]),precision=1), high=Price(float(r[2]),precision=1),
        low=Price(float(r[3]),precision=1),  close=Price(float(r[4]),precision=1),
        volume=Quantity(float(r[5]),precision=2),
        ts_event=int(idx.timestamp()*1e9), ts_init=int(idx.timestamp()*1e9))
        for idx, r in hourly.iterrows()]
    engine.add_data(bars)
    engine.add_strategy(EMACrossRR(config=EMACrossRRConfig(
        instrument_id=inst.id,
        bar_type=BarType.from_str("BTC-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"),
        trade_size=Decimal("0.01"),
        fast_ema_period=100, slow_ema_period=200,
        stop_loss_pct=sl_pct, breakeven_rr=be_rr)))
    engine.run()

    pos   = engine.trader.generate_positions_report()
    closed = pos[pos["side"] == "FLAT"].copy()
    closed["pnl"] = closed["realized_pnl"].str.replace(" USDT","").astype(float)
    acct  = engine.trader.generate_account_report(Venue("OKX"))
    start = float(acct["total"].iloc[0])
    end   = float(acct["total"].iloc[-1])
    w  = closed[closed["pnl"] > 0]
    l  = closed[closed["pnl"] < 0]
    pf = abs(w["pnl"].sum() / l["pnl"].sum()) if len(l) else float("inf")
    engine.reset(); engine.dispose()
    return dict(
        trades   = len(closed),
        winners  = len(w),
        losers   = len(l),
        wr       = len(w)/len(closed)*100 if len(closed) else 0,
        avg_win  = float(w["pnl"].mean()) if len(w) else 0,
        avg_loss = float(l["pnl"].mean()) if len(l) else 0,
        net      = end - start,
        pct      = (end - start) / start * 100,
        pf       = pf,
    )

configs = [
    ("SL 0.5% no BE",  0.005, 0.0),
    ("SL 0.5% BE@1:2", 0.005, 2.0),
    ("SL 2.0% no BE",  0.020, 0.0),
    ("SL 2.0% BE@1:2", 0.020, 2.0),
]

results = {}
for label, sl, be in configs:
    print(f"Running {label} ...")
    results[label] = build_engine(sl, be)

print()
print("=" * 80)
print(f"  {'':22}  {'SL 0.5%':>12}  {'SL 0.5% BE':>12}  {'SL 2.0%':>12}  {'SL 2.0% BE':>12}")
print("=" * 80)
keys = list(results.keys())
for metric, fmt in [
    ("trades",   "{:>12}"),
    ("winners",  "{:>12}"),
    ("losers",   "{:>12}"),
    ("wr",       "{:>11.1f}%"),
    ("avg_win",  "{:>+12.2f}"),
    ("avg_loss", "{:>+12.2f}"),
    ("pf",       "{:>12.2f}"),
    ("net",      "{:>+12.2f}"),
    ("pct",      "{:>+11.2f}%"),
]:
    labels = {
        "trades": "Trades", "winners": "Winners", "losers": "Losers",
        "wr": "Win Rate", "avg_win": "Avg Win (USDT)",
        "avg_loss": "Avg Loss (USDT)", "pf": "Profit Factor",
        "net": "Net PnL (USDT)", "pct": "Return",
    }
    row = f"  {labels[metric]:<22}"
    for k in keys:
        row += "  " + fmt.format(results[k][metric])
    print(row)
print("=" * 80)
