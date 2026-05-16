"""
AI-Assisted EMA Cross Strategy
================================
Combines classic EMA crossover signals with Claude AI market analysis.

Architecture
------------
- Fast path  : EMA cross generates a directional signal on every bar (microseconds)
- Slow path  : Claude API called in a ThreadPoolExecutor so it never blocks the event loop
- Gate logic : Only enter a trade when BOTH the EMA signal AND Claude's bias agree
               AND Claude's confidence is above `ai_confidence_threshold`
- Fallback   : If Claude is unavailable, falls back to pure EMA (degraded mode, logged)

Usage
-----
Set ANTHROPIC_API_KEY in your environment before running.
See run_ai_demo.py for a full backtest example.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from decimal import Decimal

import pandas as pd

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


_SYSTEM_PROMPT = """\
You are a quantitative FX market analyst. You receive recent OHLC bar data and EMA values.
Respond ONLY with a single JSON object — no markdown, no commentary. Schema:
{
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0-1.0>,
  "reason": "<one short sentence>"
}
"""

_USER_TEMPLATE = """\
Instrument: {instrument}
Timeframe: {bar_type}
Recent bars (oldest→newest, OHLC + volume):
{bars_csv}

Current indicators:
  fast_ema ({fast_period}): {fast_ema:.6f}
  slow_ema ({slow_period}): {slow_ema:.6f}
  ema_spread_pct: {spread_pct:+.4f}%

Current position: {position}

Given this data, what is your short-term directional bias?
"""


class AIEMACrossConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``AIEMACross`` instances.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    bar_type : BarType
        The bar type for the strategy.
    trade_size : Decimal
        The position size per trade.
    fast_ema_period : int, default 10
        The fast EMA period.
    slow_ema_period : int, default 20
        The slow EMA period.
    ai_model : str, default "claude-haiku-4-5-20251001"
        Anthropic model to use for analysis. Haiku is fast and cheap for this.
    ai_confidence_threshold : float, default 0.6
        Minimum Claude confidence to allow a trade (0.0 = ignore AI, 1.0 = max filter).
    ai_bars_context : int, default 20
        Number of recent bars to send to Claude in each request.
    ai_call_every_n_bars : int, default 5
        Call Claude every N bars (reduces API costs; last result used between calls).
    ai_fallback_on_error : bool, default True
        If True, fall back to pure EMA when Claude API fails. If False, skip the trade.
    close_positions_on_stop : bool, default True
        If all open positions should be closed on strategy stop.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    ai_model: str = "claude-haiku-4-5-20251001"
    ai_confidence_threshold: float = 0.6
    ai_bars_context: int = 20
    ai_call_every_n_bars: int = 5
    ai_fallback_on_error: bool = True
    close_positions_on_stop: bool = True


class AIEMACross(Strategy):
    """
    EMA Cross strategy gated by Claude AI directional bias.

    The EMA cross generates trade signals; Claude provides a confidence-weighted
    sanity check. A trade is only executed when both agree.

    Parameters
    ----------
    config : AIEMACrossConfig
        The configuration for the instance.
    """

    def __init__(self, config: AIEMACrossConfig) -> None:
        PyCondition.is_true(
            config.fast_ema_period < config.slow_ema_period,
            f"{config.fast_ema_period=} must be less than {config.slow_ema_period=}",
        )
        super().__init__(config)

        self.instrument: Instrument | None = None
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

        # Rolling bar buffer for AI context
        self._bar_buffer: deque[Bar] = deque(maxlen=config.ai_bars_context)

        # AI state (written from background thread, read from event loop)
        self._ai_bias: str = "NEUTRAL"
        self._ai_confidence: float = 0.0
        self._ai_reason: str = "not yet queried"
        self._ai_available: bool = False
        self._ai_bar_counter: int = 0
        self._ai_lock = threading.Lock()

        # Anthropic client (lazy init so missing key just logs a warning)
        self._anthropic_client = None
        self._init_anthropic()

    def _init_anthropic(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.log.warning(
                "ANTHROPIC_API_KEY not set — AI gating disabled, pure EMA mode",
                color=LogColor.YELLOW,
            )
            return
        try:
            import anthropic  # noqa: PLC0415

            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            self._ai_available = True
            self.log.info(
                f"Claude AI enabled — model={self.config.ai_model}, "
                f"threshold={self.config.ai_confidence_threshold}",
                color=LogColor.GREEN,
            )
        except ImportError:
            self.log.warning(
                "anthropic package not installed. Run: pip install anthropic",
                color=LogColor.YELLOW,
            )

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)

        self.request_bars(
            self.config.bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),
        )
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._bar_buffer.append(bar)

        if not self.indicators_initialized():
            self.log.info(
                f"Warming up indicators [{self.cache.bar_count(self.config.bar_type)}]",
                color=LogColor.BLUE,
            )
            return

        if bar.is_single_price():
            return

        # Trigger async Claude call every N bars
        self._ai_bar_counter += 1
        if self._ai_available and self._ai_bar_counter >= self.config.ai_call_every_n_bars:
            self._ai_bar_counter = 0
            self._request_ai_analysis_async()

        # --- EMA signal ---
        ema_signal = self._get_ema_signal()
        if ema_signal is None:
            return  # No crossover

        # --- AI gate ---
        ai_bias, ai_confidence, ai_reason = self._get_ai_state()
        ai_gate_passes = self._evaluate_ai_gate(ema_signal, ai_bias, ai_confidence)

        self.log.info(
            f"Bar signal={ema_signal} | AI bias={ai_bias} conf={ai_confidence:.2f} "
            f"({ai_reason}) | gate={'PASS' if ai_gate_passes else 'BLOCK'}",
            color=LogColor.CYAN,
        )

        if not ai_gate_passes:
            return

        # --- Execute ---
        if ema_signal == "BUY":
            if self.portfolio.is_flat(self.config.instrument_id):
                self._buy()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._buy()
        elif ema_signal == "SELL":
            if self.portfolio.is_flat(self.config.instrument_id):
                self._sell()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._sell()

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()
        self._bar_buffer.clear()
        with self._ai_lock:
            self._ai_bias = "NEUTRAL"
            self._ai_confidence = 0.0
            self._ai_reason = "reset"
        self._ai_bar_counter = 0

    # ------------------------------------------------------------------
    # EMA signal logic
    # ------------------------------------------------------------------

    def _get_ema_signal(self) -> str | None:
        """Return 'BUY', 'SELL', or None (no actionable crossover condition)."""
        if self.fast_ema.value >= self.slow_ema.value:
            return "BUY"
        return "SELL"

    # ------------------------------------------------------------------
    # AI gate
    # ------------------------------------------------------------------

    def _evaluate_ai_gate(self, ema_signal: str, ai_bias: str, ai_confidence: float) -> bool:
        if not self._ai_available:
            return self.config.ai_fallback_on_error

        if ai_confidence < self.config.ai_confidence_threshold:
            return False  # Claude not confident enough

        if ai_bias == "NEUTRAL":
            return False  # Claude says stand aside

        # Signal must match AI bias
        if ema_signal == "BUY" and ai_bias == "BULLISH":
            return True
        if ema_signal == "SELL" and ai_bias == "BEARISH":
            return True
        return False

    def _get_ai_state(self) -> tuple[str, float, str]:
        with self._ai_lock:
            return self._ai_bias, self._ai_confidence, self._ai_reason

    # ------------------------------------------------------------------
    # Async Claude API call (background thread)
    # ------------------------------------------------------------------

    def _request_ai_analysis_async(self) -> None:
        bars_snapshot = list(self._bar_buffer)
        fast_val = self.fast_ema.value
        slow_val = self.slow_ema.value
        position_str = self._describe_position()

        thread = threading.Thread(
            target=self._call_claude,
            args=(bars_snapshot, fast_val, slow_val, position_str),
            daemon=True,
        )
        thread.start()

    def _call_claude(
        self,
        bars: list[Bar],
        fast_val: float,
        slow_val: float,
        position_str: str,
    ) -> None:
        try:
            bars_csv = self._bars_to_csv(bars)
            spread_pct = ((fast_val - slow_val) / slow_val) * 100 if slow_val else 0.0

            user_msg = _USER_TEMPLATE.format(
                instrument=str(self.config.instrument_id),
                bar_type=str(self.config.bar_type),
                bars_csv=bars_csv,
                fast_period=self.config.fast_ema_period,
                fast_ema=fast_val,
                slow_period=self.config.slow_ema_period,
                slow_ema=slow_val,
                spread_pct=spread_pct,
                position=position_str,
            )

            response = self._anthropic_client.messages.create(
                model=self.config.ai_model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            raw = response.content[0].text.strip()
            parsed = json.loads(raw)

            bias = parsed.get("bias", "NEUTRAL").upper()
            confidence = float(parsed.get("confidence", 0.0))
            reason = parsed.get("reason", "")

            if bias not in ("BULLISH", "BEARISH", "NEUTRAL"):
                bias = "NEUTRAL"
            confidence = max(0.0, min(1.0, confidence))

            with self._ai_lock:
                self._ai_bias = bias
                self._ai_confidence = confidence
                self._ai_reason = reason

        except Exception as e:
            error_msg = f"Claude API error: {e}"
            with self._ai_lock:
                self._ai_reason = error_msg
                if not self.config.ai_fallback_on_error:
                    self._ai_bias = "NEUTRAL"
                    self._ai_confidence = 0.0

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _buy(self) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def _sell(self) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _describe_position(self) -> str:
        if self.portfolio.is_flat(self.config.instrument_id):
            return "FLAT"
        if self.portfolio.is_net_long(self.config.instrument_id):
            return "LONG"
        return "SHORT"

    @staticmethod
    def _bars_to_csv(bars: list[Bar]) -> str:
        if not bars:
            return "(no bars)"
        lines = ["timestamp,open,high,low,close,volume"]
        for b in bars:
            lines.append(
                f"{b.ts_event},"
                f"{b.open},{b.high},{b.low},{b.close},{b.volume}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Required overrides (no-op)
    # ------------------------------------------------------------------

    def on_data(self, data: Data) -> None:
        pass

    def on_event(self, event: Event) -> None:
        pass

    def on_save(self) -> dict[str, bytes]:
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        pass

    def on_dispose(self) -> None:
        pass
