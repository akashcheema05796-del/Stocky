#!/usr/bin/env python3
"""
Sandbox Paper Trading
======================
Runs AIEMACross strategy with:
  - REAL live price data from Binance (free, no API key needed)
  - FAKE order execution (SimulatedExchange locally — zero real money)

=======================================================================
WHAT YOU NEED TO FILL IN
=======================================================================

REQUIRED (minimum to run):
  Nothing — market data is public, orders are simulated locally.

OPTIONAL (to enable AI gating):
  Set ANTHROPIC_API_KEY environment variable before running:
    PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
    CMD:         set ANTHROPIC_API_KEY=sk-ant-api03-...

HOW TO RUN:
  .venv\\Scripts\\python.exe run_sandbox.py

  Press CTRL+C to stop.
=======================================================================
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

from strategies.ai_ema_cross import AIEMACross
from strategies.ai_ema_cross import AIEMACrossConfig

# =======================================================================
# ★  EDIT THESE  ★
# =======================================================================

# The instrument to trade (Binance Spot format: SYMBOL.BINANCE)
SYMBOL = "BTCUSDT"
INSTRUMENT_ID = InstrumentId.from_str(f"{SYMBOL}.BINANCE")

# Bar type:
#   100-TICK-LAST-INTERNAL  → engine aggregates every 100 trade ticks into 1 bar
#   1-MINUTE-LAST-EXTERNAL  → use Binance's own 1-minute klines instead
BAR_TYPE = BarType.from_str(f"{SYMBOL}.BINANCE-100-TICK-LAST-INTERNAL")

# Paper trading account size (fake money)
STARTING_BALANCE = "10000 USDT"

# Order size — for BTC this means 0.001 BTC per trade (~$60–90 at typical prices)
TRADE_SIZE = Decimal("0.001")

# EMA settings
FAST_EMA = 10
SLOW_EMA  = 20

# AI settings (only used if ANTHROPIC_API_KEY is set)
AI_MODEL              = "claude-haiku-4-5-20251001"   # fast + cheap
AI_CONFIDENCE_MIN     = 0.65   # 0.0 = always trade, 1.0 = only very confident
AI_CALL_EVERY_N_BARS  = 5      # call Claude every 5 bars to save API cost

# =======================================================================

SANDBOX_VENUE = "SANDBOX"


def build_node() -> TradingNode:
    config_node = TradingNodeConfig(
        trader_id=TraderId("SANDBOX-TRADER-001"),
        logging=LoggingConfig(
            log_level="INFO",
            log_colors=True,
        ),
        data_engine=LiveDataEngineConfig(
            time_bars_build_with_no_updates=False,
        ),
        exec_engine=LiveExecEngineConfig(
            reconciliation=False,
        ),
        data_clients={
            # -------------------------------------------------------
            # Binance public WebSocket — NO API KEY NEEDED
            # Streams real live trade ticks for the chosen symbol
            # -------------------------------------------------------
            BINANCE: BinanceDataClientConfig(
                account_type=BinanceAccountType.SPOT,
                api_key=None,    # ← leave None for public data
                api_secret=None, # ← leave None for public data
            ),
        },
        exec_clients={
            # -------------------------------------------------------
            # Sandbox execution — 100% local, zero real orders sent
            # Uses the same SimulatedExchange as the backtest engine
            # Fills are simulated against the live Binance prices
            # -------------------------------------------------------
            SANDBOX_VENUE: SandboxExecutionClientConfig(
                venue=SANDBOX_VENUE,
                starting_balances=[STARTING_BALANCE],
                base_currency="USDT",
                oms_type="NETTING",
                account_type="CASH",
            ),
        },
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )

    node = TradingNode(config=config_node)

    # Wire up the strategy
    strategy = AIEMACross(
        config=AIEMACrossConfig(
            instrument_id=INSTRUMENT_ID,
            bar_type=BAR_TYPE,
            trade_size=TRADE_SIZE,
            fast_ema_period=FAST_EMA,
            slow_ema_period=SLOW_EMA,
            ai_model=AI_MODEL,
            ai_confidence_threshold=AI_CONFIDENCE_MIN,
            ai_call_every_n_bars=AI_CALL_EVERY_N_BARS,
            ai_fallback_on_error=True,
            close_positions_on_stop=True,
        )
    )
    node.trader.add_strategy(strategy)

    # Register the data client factory (Binance WebSocket)
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)

    # Register the sandbox execution factory (local simulation)
    node.add_exec_client_factory(SANDBOX_VENUE, SandboxLiveExecClientFactory)

    node.build()
    return node


if __name__ == "__main__":
    import os

    print("\n" + "=" * 60)
    print("  Sandbox Paper Trading")
    print(f"  Instrument : {INSTRUMENT_ID}")
    print(f"  Bar type   : {BAR_TYPE}")
    print(f"  Balance    : {STARTING_BALANCE} (fake)")
    print(f"  Trade size : {TRADE_SIZE}")
    print(f"  AI mode    : {'ENABLED' if os.environ.get('ANTHROPIC_API_KEY') else 'DISABLED (pure EMA)'}")
    print("=" * 60)
    print("\nConnecting to Binance live feed...")
    print("Press CTRL+C to stop.\n")

    node = build_node()

    try:
        node.run()   # blocks until CTRL+C
    finally:
        node.dispose()
        print("\nSandbox session ended.")
