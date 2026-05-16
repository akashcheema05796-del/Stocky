#!/usr/bin/env python3
"""
Demo backtest: AUD/USD EMA Cross strategy with tearsheet output.
Uses built-in test data - no external downloads required.
"""

import time
import webbrowser
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.backtest.modules import FXRolloverInterestConfig
from nautilus_trader.backtest.modules import FXRolloverInterestModule
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross
from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NautilusTrader Demo Backtest")
    print("  Strategy : EMA Cross (fast=10, slow=20)")
    print("  Instrument: AUD/USD")
    print("  Data      : Built-in tick data (TrueFX)")
    print("  Capital   : $1,000,000 USD")
    print("="*60 + "\n")

    # --- Engine config with logging ---
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(
            log_level="INFO",
            log_colors=True,
        ),
    )

    engine = BacktestEngine(config=config)

    # --- Simulated venue ---
    fill_model = FillModel(
        prob_fill_on_limit=0.2,
        prob_slippage=0.5,
        random_seed=42,
    )

    provider = TestDataProvider()
    interest_rate_data = provider.read_csv("short-term-interest.csv")
    fx_rollover_interest = FXRolloverInterestModule(
        config=FXRolloverInterestConfig(interest_rate_data)
    )

    SIM = Venue("SIM")
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(1_000_000, USD)],
        fill_model=fill_model,
        modules=[fx_rollover_interest],
    )

    # --- Instrument + data ---
    AUDUSD_SIM = TestInstrumentProvider.default_fx_ccy("AUD/USD", SIM)
    engine.add_instrument(AUDUSD_SIM)

    wrangler = QuoteTickDataWrangler(instrument=AUDUSD_SIM)
    ticks = wrangler.process(provider.read_csv_ticks("truefx/audusd-ticks.csv"))
    engine.add_data(ticks)
    print(f"Loaded {len(ticks):,} ticks\n")

    # --- Strategy ---
    strategy = EMACross(config=EMACrossConfig(
        instrument_id=AUDUSD_SIM.id,
        bar_type=BarType.from_str("AUD/USD.SIM-100-TICK-MID-INTERNAL"),
        trade_size=Decimal(1_000_000),
        fast_ema_period=10,
        slow_ema_period=20,
        close_positions_on_stop=True,
    ))
    engine.add_strategy(strategy=strategy)

    # --- Run ---
    print("Running backtest...\n")
    start = time.time()
    engine.run()
    elapsed = time.time() - start
    print(f"\nBacktest completed in {elapsed:.2f}s\n")

    # --- Reports ---
    print("="*60)
    print("ACCOUNT REPORT")
    print("="*60)
    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_account_report(SIM))

    print("\n" + "="*60)
    print("ORDER FILLS")
    print("="*60)
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_order_fills_report())

    print("\n" + "="*60)
    print("POSITIONS")
    print("="*60)
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_positions_report())

    # --- Tearsheet ---
    print("\n" + "="*60)
    print("Generating tearsheet...")
    output_path = Path("tearsheet.html").resolve()

    try:
        from nautilus_trader.analysis import TearsheetConfig, create_tearsheet
        create_tearsheet(
            engine=engine,
            output_path=str(output_path),
            config=TearsheetConfig(theme="nautilus_dark"),
        )
        print(f"Tearsheet saved → {output_path}")
        print("Opening in browser...")
        webbrowser.open(output_path.as_uri())
    except Exception as e:
        print(f"Tearsheet generation skipped: {e}")

    print("="*60)

    engine.reset()
    engine.dispose()
