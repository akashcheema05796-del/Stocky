#!/usr/bin/env python3
"""
EMA Cross — Stop-Loss + Breakeven management
=============================================
Entry : Market order on EMA(fast) × EMA(slow) crossover.
Exit  :
  ① Stop-Loss hits        → loss capped at stop_loss_pct
  ② 1:2 R:R reached       → SL moved to entry price (breakeven — zero risk)
  ③ EMA crosses back      → cancel SL, close at market (ride the full trend)

Once the trade moves 2× the initial risk in our favour the worst outcome
becomes breakeven — unlimited upside remains open.
"""

from decimal import Decimal

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators import ExponentialMovingAverage


class EMACrossRRConfig(StrategyConfig, frozen=True):
    """
    Configuration for EMACrossRR.

    Parameters
    ----------
    instrument_id : InstrumentId
        Instrument to trade.
    bar_type : BarType
        Bar series to drive signals.
    trade_size : Decimal
        Fixed lot size per trade (e.g. 0.01 BTC).
    fast_ema_period : int, default 100
        Fast EMA period.
    slow_ema_period : int, default 200
        Slow EMA period.
    stop_loss_pct : float, default 0.005
        Stop-loss distance as fraction of entry price (0.005 = 0.5%).
        Profits are uncapped — SL is the only hard exit.
    subscribe_quote_ticks : bool, default False
        Subscribe to live quote ticks (needed for sandbox paper trading).
    subscribe_trade_ticks : bool, default False
        Subscribe to live trade ticks (needed for sandbox paper trading).
    request_bars : bool, default False
        Request historical bars on start (for live/paper trading warm-up).
    close_positions_on_stop : bool, default True
        Close open positions when strategy is stopped.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 100
    slow_ema_period: PositiveInt = 200
    stop_loss_pct: float = 0.005        # 0.5% hard stop
    breakeven_rr: float = 2.0          # move SL to entry when profit = rr × risk
                                        # e.g. 2.0 → move BE at 1% profit (2 × 0.5%)
                                        # set to 0.0 to disable
    risk_reward: float = 2.0            # kept for config compatibility, not used
    subscribe_quote_ticks: bool = False
    subscribe_trade_ticks: bool = False
    request_bars: bool = False
    close_positions_on_stop: bool = True


class EMACrossRR(Strategy):
    """
    EMA Cross — unlimited upside, capped downside.

    Entry : market order when EMA(fast) crosses EMA(slow).
    Exit  :
        • SL stop-market fires  → loss <= stop_loss_pct
        • EMA crosses back      → close at market (take whatever profit is there)
    """

    def __init__(self, config: EMACrossRRConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

        # Track the pending entry order and active SL order
        self._entry_order_id: ClientOrderId | None = None
        self._sl_order_id:    ClientOrderId | None = None
        self._entry_side:     OrderSide | None = None
        self._entry_px:       float = 0.0

        # Breakeven management
        self._be_target_px:  float = 0.0   # price at which we move SL to entry
        self._at_breakeven:  bool  = False  # True once SL has been moved to entry

        # Previous-bar EMA relationship — None until warm-up completes.
        # We store a bool: True = fast was above slow, False = fast was below.
        # A trade is only triggered when this VALUE CHANGES (actual crossover).
        self._prev_bullish: bool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)
        self.subscribe_bars(self.config.bar_type)

        if self.config.subscribe_quote_ticks:
            self.subscribe_quote_ticks(self.config.instrument_id)
        if self.config.subscribe_trade_ticks:
            self.subscribe_trade_ticks(self.config.instrument_id)

        if self.config.request_bars:
            import pandas as pd
            self.request_bars(
                self.config.bar_type,
                start=self._clock.utc_now() - pd.Timedelta(days=1),
            )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
        if self.config.subscribe_quote_ticks:
            self.unsubscribe_quote_ticks(self.config.instrument_id)
        if self.config.subscribe_trade_ticks:
            self.unsubscribe_trade_ticks(self.config.instrument_id)

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()
        self._entry_order_id = None
        self._sl_order_id    = None
        self._entry_side     = None
        self._entry_px       = 0.0
        self._be_target_px   = 0.0
        self._at_breakeven   = False
        self._prev_bullish   = None

    # ------------------------------------------------------------------
    # Event handler — place SL after entry fills
    # ------------------------------------------------------------------

    def on_event(self, event) -> None:
        if not isinstance(event, OrderFilled):
            return

        # If the entry order just filled → place the SL stop-market
        if (self._entry_order_id is not None
                and event.client_order_id == self._entry_order_id):
            self._entry_px       = float(event.last_px)
            self._entry_order_id = None          # entry is done
            self._place_sl(self._entry_side, self._entry_px)

        # If the SL order just filled → clear state (position is flat)
        elif (self._sl_order_id is not None
                and event.client_order_id == self._sl_order_id):
            be_tag = " [BREAKEVEN]" if self._at_breakeven else ""
            self.log.info(
                f"SL hit{be_tag} @ {event.last_px}  entry was {self._entry_px:.1f}",
                color=LogColor.YELLOW if self._at_breakeven else LogColor.RED,
            )
            self._sl_order_id  = None
            self._entry_side   = None
            self._be_target_px = 0.0
            self._at_breakeven = False

    # ------------------------------------------------------------------
    # Bar handler — EMA cross signals
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized():
            self.log.info(
                f"Warming up [{self.cache.bar_count(self.config.bar_type)}/"
                f"{self.config.slow_ema_period}]",
                color=LogColor.BLUE,
            )
            return

        if bar.is_single_price():
            return

        curr_bullish = self.fast_ema.value >= self.slow_ema.value

        # ── Breakeven check (runs every bar while in a position) ───────
        # Once the trade moves breakeven_rr × risk in our favour,
        # move the SL to entry price → worst outcome is now 0.
        if (not self._at_breakeven
                and self._sl_order_id is not None
                and self._entry_side is not None
                and self._be_target_px > 0.0
                and self.config.breakeven_rr > 0.0):
            if self._entry_side == OrderSide.BUY:
                triggered = float(bar.high) >= self._be_target_px
            else:
                triggered = float(bar.low)  <= self._be_target_px
            if triggered:
                self._move_sl_to_breakeven()

        # First bar after warm-up: record state but don't trade.
        # We need TWO bars to confirm a crossover direction.
        if self._prev_bullish is None:
            self._prev_bullish = curr_bullish
            return

        # ── Crossover detection ────────────────────────────────────────
        # crossed_up   : fast was BELOW slow, now ABOVE  → BUY signal
        # crossed_down : fast was ABOVE slow, now BELOW  → SELL signal
        crossed_up   = (not self._prev_bullish) and curr_bullish
        crossed_down = self._prev_bullish and (not curr_bullish)

        # Always update state for next bar
        self._prev_bullish = curr_bullish

        # No crossover this bar — nothing to do
        if not (crossed_up or crossed_down):
            return

        is_flat  = self.portfolio.is_flat(self.config.instrument_id)
        is_long  = self.portfolio.is_net_long(self.config.instrument_id)
        is_short = self.portfolio.is_net_short(self.config.instrument_id)

        if crossed_up:
            if is_flat:
                self._enter(OrderSide.BUY, bar.close)
            elif is_short:
                # Trend crossed back bullish — exit short, go long
                self._exit_and_reverse(OrderSide.BUY, bar.close)
        else:  # crossed_down
            if is_flat:
                self._enter(OrderSide.SELL, bar.close)
            elif is_long:
                # Trend crossed back bearish — exit long, go short
                self._exit_and_reverse(OrderSide.SELL, bar.close)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(self, side: OrderSide, ref_price) -> None:
        """Submit a market entry order. SL is placed in on_event after fill."""
        qty = self.instrument.make_qty(self.config.trade_size)

        entry = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
            tags=["ENTRY"],
        )

        self._entry_order_id = entry.client_order_id
        self._entry_side     = side
        self.submit_order(entry)

        self.log.info(
            f"{'BUY ' if side == OrderSide.BUY else 'SELL'} entry  "
            f"ref={float(ref_price):.1f}  "
            f"SL will be placed {self.config.stop_loss_pct*100:.2f}% away after fill",
            color=LogColor.GREEN if side == OrderSide.BUY else LogColor.RED,
        )

    def _place_sl(self, side: OrderSide, entry_px: float) -> None:
        """Submit a stop-market SL after the entry fills."""
        sl_pct = self.config.stop_loss_pct
        be_rr  = self.config.breakeven_rr
        pp     = self.instrument.price_precision
        qty    = self.instrument.make_qty(self.config.trade_size)

        if side == OrderSide.BUY:
            sl_price         = Price(entry_px * (1.0 - sl_pct), precision=pp)
            exit_side        = OrderSide.SELL
            self._be_target_px = entry_px * (1.0 + sl_pct * be_rr) if be_rr > 0 else 0.0
        else:
            sl_price         = Price(entry_px * (1.0 + sl_pct), precision=pp)
            exit_side        = OrderSide.BUY
            self._be_target_px = entry_px * (1.0 - sl_pct * be_rr) if be_rr > 0 else 0.0

        self._at_breakeven = False

        sl_order = self.order_factory.stop_market(
            instrument_id=self.config.instrument_id,
            order_side=exit_side,
            quantity=qty,
            trigger_price=sl_price,
            trigger_type=TriggerType.LAST_PRICE,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            tags=["SL"],
        )

        self._sl_order_id = sl_order.client_order_id
        self.submit_order(sl_order)

        be_info = (f"  BE target={self._be_target_px:.1f}"
                   f" (+{sl_pct*be_rr*100:.2f}%)" if be_rr > 0 else "")
        self.log.info(
            f"SL placed @ {sl_price}  (entry={entry_px:.1f}  "
            f"risk={sl_pct*100:.2f}%{be_info})",
            color=LogColor.YELLOW,
        )

    def _move_sl_to_breakeven(self) -> None:
        """Cancel the current SL and replace it at entry price (zero risk)."""
        sl_order = self.cache.order(self._sl_order_id)
        if sl_order is None or not sl_order.is_open:
            return

        # Cancel old SL
        self.cancel_order(sl_order)
        self._sl_order_id = None

        # New SL at entry price
        pp        = self.instrument.price_precision
        qty       = self.instrument.make_qty(self.config.trade_size)
        be_price  = Price(self._entry_px, precision=pp)
        exit_side = OrderSide.SELL if self._entry_side == OrderSide.BUY else OrderSide.BUY

        new_sl = self.order_factory.stop_market(
            instrument_id=self.config.instrument_id,
            order_side=exit_side,
            quantity=qty,
            trigger_price=be_price,
            trigger_type=TriggerType.LAST_PRICE,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            tags=["SL", "BE"],
        )

        self._sl_order_id  = new_sl.client_order_id
        self._at_breakeven = True
        self.submit_order(new_sl)

        self.log.info(
            f"BREAKEVEN: SL moved to entry @ {be_price}  "
            f"(1:{self.config.breakeven_rr:.0f} R:R reached — "
            f"trade can no longer lose)",
            color=LogColor.GREEN,
        )

    def _exit_and_reverse(self, new_side: OrderSide, ref_price) -> None:
        """EMA reversal — cancel SL, close current position, open opposite."""
        # Cancel the standing SL first
        if self._sl_order_id is not None:
            sl_order = self.cache.order(self._sl_order_id)
            if sl_order is not None and sl_order.is_open:
                self.cancel_order(sl_order)
            self._sl_order_id = None

        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

        # Clear all state
        self._entry_side   = None
        self._be_target_px = 0.0
        self._at_breakeven = False

        # Enter in the new direction
        self._enter(new_side, ref_price)
