#!/usr/bin/env python3
"""
AI-Assisted EMA Cross Backtest
================================
Runs the AIEMACross strategy which gates EMA signals with Claude AI analysis.

Setup
-----
1. Install the anthropic package:
       pip install anthropic

2. Set your API key:
       $env:ANTHROPIC_API_KEY = "sk-ant-..."   (PowerShell)
       set ANTHROPIC_API_KEY=sk-ant-...         (CMD)

3. Run:
       python run_ai_demo.py

Notes
-----
- Without ANTHROPIC_API_KEY the strategy falls back to pure EMA (ai_fallback_on_error=True).
- ai_call_every_n_bars=5 means Claude is called once per 5 bars to keep API costs low.
- ai_confidence_threshold=0.65 requires Claude to be at least 65% confident before trading.
- Change ai_model to "claude-sonnet-4-6" for a stronger but slower/pricier model.
"""

import sys
import time
import webbrowser
from decimal import Decimal
from pathlib import Path

# Make sure the strategies folder is importable
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.backtest.modules import FXRolloverInterestConfig
from nautilus_trader.backtest.modules import FXRolloverInterestModule
from nautilus_trader.config import LoggingConfig
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

from strategies.ai_ema_cross import AIEMACross
from strategies.ai_ema_cross import AIEMACrossConfig


if __name__ == "__main__":
    import os
    ai_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))

    print("\n" + "=" * 60)
    print("  AI-Assisted EMA Cross Backtest")
    print(f"  AI Mode  : {'ENABLED (Claude)' if ai_key_set else 'DISABLED (pure EMA fallback)'}")
    print("  Strategy : EMA Cross + Claude bias filter")
    print("  Instrument: AUD/USD")
    print("  Capital   : $1,000,000 USD")
    print("=" * 60 + "\n")

    config = BacktestEngineConfig(
        trader_id=TraderId("AI-BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO", log_colors=True),
    )
    engine = BacktestEngine(config=config)

    fill_model = FillModel(prob_fill_on_limit=0.2, prob_slippage=0.5, random_seed=42)

    provider = TestDataProvider()
    interest_rate_data = provider.read_csv("short-term-interest.csv")
    fx_rollover = FXRolloverInterestModule(
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
        modules=[fx_rollover],
    )

    AUDUSD_SIM = TestInstrumentProvider.default_fx_ccy("AUD/USD", SIM)
    engine.add_instrument(AUDUSD_SIM)

    wrangler = QuoteTickDataWrangler(instrument=AUDUSD_SIM)
    ticks = wrangler.process(provider.read_csv_ticks("truefx/audusd-ticks.csv"))
    engine.add_data(ticks)
    print(f"Loaded {len(ticks):,} ticks\n")

    strategy = AIEMACross(
        config=AIEMACrossConfig(
            instrument_id=AUDUSD_SIM.id,
            bar_type=BarType.from_str("AUD/USD.SIM-100-TICK-MID-INTERNAL"),
            trade_size=Decimal(1_000_000),
            fast_ema_period=10,
            slow_ema_period=20,
            ai_model="claude-haiku-4-5-20251001",
            ai_confidence_threshold=0.65,
            ai_bars_context=20,
            ai_call_every_n_bars=5,
            ai_fallback_on_error=True,
            close_positions_on_stop=True,
        )
    )
    engine.add_strategy(strategy=strategy)

    print("Running backtest...\n")
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0
    print(f"\nBacktest completed in {elapsed:.2f}s\n")

    print("=" * 60)
    print("ACCOUNT REPORT")
    print("=" * 60)
    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
        print(engine.trader.generate_account_report(SIM))

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

    print("\n" + "=" * 60)
    print("Generating tearsheet...")
    output_path = Path("tearsheet_ai.html").resolve()
    try:
        from nautilus_trader.analysis import TearsheetConfig, create_tearsheet

        create_tearsheet(
            engine=engine,
            output_path=str(output_path),
            config=TearsheetConfig(theme="nautilus_dark"),
        )
        print(f"Tearsheet saved → {output_path}")
        webbrowser.open(output_path.as_uri())
    except Exception as e:
        print(f"Tearsheet skipped: {e}")

    print("=" * 60)
    engine.reset()
    engine.dispose()
