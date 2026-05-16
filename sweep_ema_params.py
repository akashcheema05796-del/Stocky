#!/usr/bin/env python3
"""
Sweep multiple EMA period combinations to find which generates enough trades
AND benefits most from the new filters.

We test:
  EMA pairs : (20,50)  (50,100)  (100,200)
  Filters   : baseline, RSI 65/35, ATR SL 2×, cooldown 5, combined best
"""
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

print("Loading 1H bars from cache...")
raw = pd.read_csv("data/BTC_USDT_SWAP_1m_3y.csv.gz", compression="gzip", header=None)
raw.columns = range(len(raw.columns))
raw[0] = raw[0].astype(int)
for c in [1,2,3,4,5]: raw[c] = raw[c].astype(float)
raw.index = pd.to_datetime(raw[0], unit="ms", utc=True)
hourly = raw.resample("1h").agg({1:"first",2:"max",3:"min",4:"last",5:"sum"}).dropna(subset=[1])
print(f"  {len(hourly):,} 1H bars ready\n")


def run(label, fast, slow, gap_pct=0.0, rsi_long_max=70.0, rsi_short_min=30.0,
        atr_mult=0.0, cooldown=0, sl_pct=0.02):
    inst = CryptoPerpetual(
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
        ts_event=0, ts_init=0)

    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("T-001"),
        logging=LoggingConfig(log_level="ERROR", bypass_logging=True)))
    engine.add_venue(
        venue=Venue("OKX"), oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN, base_currency=USDT,
        starting_balances=[Money(10_000, USDT)],
        fill_model=FillModel(prob_fill_on_limit=0.95, prob_slippage=0.3, random_seed=42),
        default_leverage=Decimal("10"))
    engine.add_instrument(inst)

    bt = BarType(instrument_id=inst.id,
        bar_spec=BarSpecification(1, BarAggregation.HOUR, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL)
    bars = [Bar(bar_type=bt,
        open=Price(float(r[1]), precision=1),
        high=Price(float(r[2]), precision=1),
        low=Price(float(r[3]), precision=1),
        close=Price(float(r[4]),precision=1),
        volume=Quantity(float(r[5]), precision=2),
        ts_event=int(idx.timestamp()*1e9),
        ts_init=int(idx.timestamp()*1e9))
        for idx, r in hourly.iterrows()]
    engine.add_data(bars)

    engine.add_strategy(EMACrossRR(config=EMACrossRRConfig(
        instrument_id=inst.id,
        bar_type=BarType.from_str("BTC-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"),
        trade_size=Decimal("0.01"),
        fast_ema_period=fast, slow_ema_period=slow,
        stop_loss_pct=sl_pct, breakeven_rr=0.0,
        min_ema_gap_pct=gap_pct,
        rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
        atr_sl_multiplier=atr_mult,
        sl_cooldown_bars=cooldown,
    )))
    engine.run()

    pos    = engine.trader.generate_positions_report()
    if "side" not in pos.columns:
        engine.reset(); engine.dispose()
        return dict(label=label, trades=0, winners=0, losers=0, wr=0.0,
                    avg_win=0.0, avg_loss=0.0, net=0.0, pct=0.0, pf=0.0)
    closed = pos[pos["side"] == "FLAT"].copy()
    closed["pnl"] = closed["realized_pnl"].str.replace(" USDT", "").astype(float)
    acct   = engine.trader.generate_account_report(Venue("OKX"))
    start  = float(acct["total"].iloc[0])
    end    = float(acct["total"].iloc[-1])
    w      = closed[closed["pnl"] > 0]
    l      = closed[closed["pnl"] < 0]
    pf     = abs(w["pnl"].sum() / l["pnl"].sum()) if len(l) and l["pnl"].sum() != 0 else float("inf")
    engine.reset(); engine.dispose()
    return dict(label=label, trades=len(closed), winners=len(w), losers=len(l),
                wr=len(w)/len(closed)*100 if len(closed) else 0,
                avg_win=float(w["pnl"].mean()) if len(w) else 0,
                avg_loss=float(l["pnl"].mean()) if len(l) else 0,
                net=end - start, pct=(end - start) / start * 100, pf=pf)


configs = []
for fast, slow in [(20, 50), (50, 100), (100, 200)]:
    tag = f"EMA({fast}/{slow})"
    configs += [
        dict(label=f"{tag} Baseline",           fast=fast, slow=slow),
        dict(label=f"{tag} RSI 65/35",          fast=fast, slow=slow, rsi_long_max=65.0, rsi_short_min=35.0),
        dict(label=f"{tag} ATR SL 2x",          fast=fast, slow=slow, atr_mult=2.0),
        dict(label=f"{tag} ATR SL 3x",          fast=fast, slow=slow, atr_mult=3.0),
        dict(label=f"{tag} Cooldown 5",         fast=fast, slow=slow, cooldown=5),
        dict(label=f"{tag} RSI+ATR2x+CD5",      fast=fast, slow=slow, rsi_long_max=65.0, rsi_short_min=35.0, atr_mult=2.0, cooldown=5),
    ]

print(f"Running {len(configs)} configurations...\n")
results = []
for cfg in configs:
    print(f"  [{len(results)+1:2d}/{len(configs)}] {cfg['label']} ...", end=" ", flush=True)
    r = run(**cfg)
    results.append(r)
    print(f"net={r['net']:+.2f} USDT ({r['pct']:+.2f}%)  trades={r['trades']}  WR={r['wr']:.0f}%  PF={r['pf']:.2f}")

# Print grouped by EMA pair
print()
for fast, slow in [(20, 50), (50, 100), (100, 200)]:
    tag = f"EMA({fast}/{slow})"
    group = [r for r in results if r["label"].startswith(tag)]
    best  = max(group, key=lambda x: x["net"])
    print(f"\n{'='*90}")
    print(f"  {tag}  (best: {best['label']}  net={best['net']:+.2f} USDT  {best['pct']:+.2f}%)")
    print(f"{'='*90}")
    print(f"  {'Label':<30}  {'T':>4}  {'W':>4}  {'L':>4}  {'WR%':>5}  "
          f"{'AvgW':>7}  {'AvgL':>7}  {'PF':>5}  {'Net':>9}  {'Ret':>7}")
    print(f"  {'-'*88}")
    for r in sorted(group, key=lambda x: x["net"], reverse=True):
        marker = " *" if r["label"] == best["label"] else "  "
        print(f"{marker} {r['label']:<30}  {r['trades']:>4}  {r['winners']:>4}  {r['losers']:>4}  "
              f"{r['wr']:>4.0f}%  {r['avg_win']:>+7.2f}  {r['avg_loss']:>+7.2f}  "
              f"{r['pf']:>5.2f}  {r['net']:>+9.2f}  {r['pct']:>+6.2f}%")

# Overall best
overall_best = max(results, key=lambda x: x["net"])
print(f"\n\nOverall best: {overall_best['label']}")
print(f"  Net: {overall_best['net']:+.2f} USDT ({overall_best['pct']:+.2f}%)  "
      f"Trades: {overall_best['trades']}  WR: {overall_best['wr']:.1f}%  "
      f"PF: {overall_best['pf']:.2f}")
