#!/usr/bin/env python3
"""
Pure EMA Cross — Sandbox Paper Trading (OKX live feed)
Real OKX prices, zero real money. Press CTRL+C to stop.
"""

from decimal import Decimal

from nautilus_trader.adapters.okx import OKX
from nautilus_trader.adapters.okx import OKXDataClientConfig
from nautilus_trader.adapters.okx import OKXLiveDataClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.nautilus_pyo3 import OKXContractType
from nautilus_trader.core.nautilus_pyo3 import OKXEnvironment
from nautilus_trader.core.nautilus_pyo3 import OKXInstrumentType
from nautilus_trader.examples.strategies.ema_cross import EMACross
from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

# ── Settings ─────────────────────────────────────────────────────────
SYMBOL        = "BTC-USDT-SWAP"          # OKX perpetual futures
INSTRUMENT_ID = InstrumentId.from_str(f"{SYMBOL}.{OKX}")
BAR_TYPE      = BarType.from_str(f"{SYMBOL}.{OKX}-1-MINUTE-LAST-EXTERNAL")
STARTING_BALANCE = "10000 USDT"
TRADE_SIZE    = Decimal("0.01")          # OKX SWAP: 0.01 lots
FAST_EMA      = 10
SLOW_EMA      = 20
SANDBOX_VENUE = "OKX"   # sandbox impersonates OKX so order routing works
# ─────────────────────────────────────────────────────────────────────

config = TradingNodeConfig(
    trader_id=TraderId("EMA-SANDBOX-001"),
    logging=LoggingConfig(log_level="INFO", log_colors=True),
    data_engine=LiveDataEngineConfig(time_bars_build_with_no_updates=False),
    exec_engine=LiveExecEngineConfig(reconciliation=False),
    data_clients={
        OKX: OKXDataClientConfig(
            environment=OKXEnvironment.LIVE,
            api_key=None,
            api_secret=None,
            api_passphrase=None,
            instrument_types=(OKXInstrumentType.SWAP,),
            contract_types=(OKXContractType.LINEAR,),
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset([INSTRUMENT_ID]),
            ),
        ),
    },
    exec_clients={
        SANDBOX_VENUE: SandboxExecutionClientConfig(
            venue=SANDBOX_VENUE,          # "OKX" — intercepts all OKX venue orders
            starting_balances=[STARTING_BALANCE],
            base_currency="USDT",
            oms_type="NETTING",
            account_type="MARGIN",
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=10.0,
    timeout_post_stop=5.0,
)

node = TradingNode(config=config)

strategy = EMACross(config=EMACrossConfig(
    instrument_id=INSTRUMENT_ID,
    bar_type=BAR_TYPE,
    trade_size=TRADE_SIZE,
    fast_ema_period=FAST_EMA,
    slow_ema_period=SLOW_EMA,
    subscribe_quote_ticks=True,   # sandbox matching engine needs tick prices
    subscribe_trade_ticks=True,
    request_bars=True,
    close_positions_on_stop=True,
))

node.trader.add_strategy(strategy)
node.add_data_client_factory(OKX, OKXLiveDataClientFactory)
node.add_exec_client_factory(SANDBOX_VENUE, SandboxLiveExecClientFactory)
node.build()

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  EMA Cross — Sandbox Paper Trading (OKX)")
    print(f"  Instrument : {INSTRUMENT_ID}")
    print(f"  Bars       : 1-minute klines from OKX")
    print(f"  EMA        : fast={FAST_EMA}  slow={SLOW_EMA}")
    print(f"  Trade size : {TRADE_SIZE} lots  (fake money)")
    print(f"  Balance    : {STARTING_BALANCE}  (fake)")
    print("=" * 55)
    print("\nConnecting to OKX live feed...")
    print("Warming up EMAs — first trade after 20 bars (~20 min).")
    print("Press CTRL+C to stop.\n")
    try:
        node.run()
    finally:
        node.dispose()
        print("\nSession ended.")
