#!/usr/bin/env python3
"""
EMA Cross — Enhanced Strategy (v2)
====================================
Improvements over v1
--------------------
1. **EMA gap filter**   — only enter when |fast − slow| / slow ≥ min_ema_gap_pct.
                          Weak / brush crossovers in ranging markets are skipped.
2. **RSI filter**       — skip longs when RSI(14) > rsi_long_max (overbought),
                          skip shorts when RSI(14) < rsi_short_min (oversold).
3. **ATR-based SL**     — when atr_sl_multiplier > 0, SL = entry ± ATR(14) × mult
                          instead of a fixed percentage.  Adapts to volatility.
4. **Cooldown**         — after an SL hit, wait sl_cooldown_bars bars before
                          accepting a new entry.  Avoids whipsaw re-entries.

Exit logic (unchanged from v1)
-------------------------------
  • Initial SL = entry ± stop_loss_pct  (or ATR-based if configured)
  • Trail activates once price reaches 1:1 R:R (1× SL distance in profit).
  • SL then trails stop_loss_pct below the rolling peak.
  • EMA crosses back → cancel SL, close at market.
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
from nautilus_trader.model.events import OrderRejected
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import AverageTrueRange


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
        Ignored when atr_sl_multiplier > 0.
    breakeven_rr : float, default 0.0
        Move SL to entry when profit = breakeven_rr × risk.  0 = disabled.
    risk_reward : float, default 2.0
        Kept for config compatibility, not used in signal logic.

    --- Improvement knobs ---
    min_ema_gap_pct : float, default 0.0
        Minimum separation between fast and slow EMA, as a fraction of the
        slow EMA value (e.g. 0.002 = 0.2%).  0 = disabled (all crossovers).
    rsi_period : PositiveInt, default 14
        Period for the RSI filter.
    rsi_long_max : float, default 70.0
        Do not open longs when RSI > this value.  70 = disabled (effectively).
    rsi_short_min : float, default 30.0
        Do not open shorts when RSI < this value.  30 = disabled (effectively).
    atr_period : PositiveInt, default 14
        Period for the ATR-based stop calculation.
    atr_sl_multiplier : float, default 0.0
        If > 0, overrides stop_loss_pct: SL = entry ± ATR × atr_sl_multiplier.
    sl_cooldown_bars : int, default 0
        Number of bars to skip after an SL hit before accepting a new entry.
        0 = no cooldown.

    --- Live / paper trading ---
    subscribe_quote_ticks : bool, default False
    subscribe_trade_ticks : bool, default False
    request_bars : bool, default False
    close_positions_on_stop : bool, default True
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 100
    slow_ema_period: PositiveInt = 200
    stop_loss_pct: float = 0.005
    breakeven_rr: float = 0.0
    risk_reward: float = 2.0

    # ── Improvement knobs ─────────────────────────────────────────────
    min_ema_gap_pct: float = 0.0          # 0 = disabled
    rsi_period: PositiveInt = 14
    rsi_long_max: float = 70.0            # skip long if RSI > this
    rsi_short_min: float = 30.0           # skip short if RSI < this
    atr_period: PositiveInt = 14
    atr_sl_multiplier: float = 0.0        # 0 = use stop_loss_pct instead
    sl_cooldown_bars: int = 0             # 0 = no cooldown

    # ── Live / paper trading ──────────────────────────────────────────
    subscribe_quote_ticks: bool = False
    subscribe_trade_ticks: bool = False
    request_bars: bool = False
    close_positions_on_stop: bool = True


class EMACrossRR(Strategy):
    """
    EMA Cross — enhanced with RSI filter, EMA gap filter, ATR-based SL,
    and post-SL cooldown.
    """

    def __init__(self, config: EMACrossRRConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None

        # ── Indicators ────────────────────────────────────────────────
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.rsi      = RelativeStrengthIndex(config.rsi_period)
        self.atr      = AverageTrueRange(config.atr_period)

        # ── Order tracking ────────────────────────────────────────────
        self._entry_order_id: ClientOrderId | None = None
        self._sl_order_id:    ClientOrderId | None = None
        self._entry_side:     OrderSide | None = None
        self._entry_px:       float = 0.0

        # ── Trailing SL state ─────────────────────────────────────────
        self._peak_px:      float = 0.0
        self._trail_px:     float = 0.0
        self._trail_active: bool  = False
        self._sl_distance:  float = 0.0   # actual SL distance used (ATR or pct)

        # ── Breakeven state ───────────────────────────────────────────
        self._be_target_px:  float = 0.0
        self._at_breakeven:  bool  = False

        # ── Crossover memory ──────────────────────────────────────────
        self._prev_bullish: bool | None = None

        # ── Cooldown counter ─────────────────────────────────────────
        self._cooldown_bars_left: int = 0

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
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
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
        self.rsi.reset()
        self.atr.reset()
        self._entry_order_id     = None
        self._sl_order_id        = None
        self._entry_side         = None
        self._entry_px           = 0.0
        self._peak_px            = 0.0
        self._trail_px           = 0.0
        self._trail_active       = False
        self._sl_distance        = 0.0
        self._be_target_px       = 0.0
        self._at_breakeven       = False
        self._prev_bullish       = None
        self._cooldown_bars_left = 0

    # ------------------------------------------------------------------
    # Event handler — place SL after entry fills
    # ------------------------------------------------------------------

    def on_event(self, event) -> None:
        # ── SL order rejected (price gapped past stop level) ──────────
        # This can happen on fast-moving bars (common on 1-minute data).
        # Treat it like an immediate SL fill: close the position at market.
        if isinstance(event, OrderRejected):
            if (self._sl_order_id is not None
                    and event.client_order_id == self._sl_order_id):
                self.log.warning(
                    f"SL order REJECTED (price already past stop) — "
                    f"closing position at market",
                )
                self._sl_order_id = None
                self.cancel_all_orders(self.config.instrument_id)
                self.close_all_positions(self.config.instrument_id)
                # State will be cleaned up when the close fill arrives
                self._entry_side   = None
                self._peak_px      = 0.0
                self._trail_px     = 0.0
                self._trail_active = False
                self._sl_distance  = 0.0
                self._be_target_px = 0.0
                self._at_breakeven = False
                if self.config.sl_cooldown_bars > 0:
                    self._cooldown_bars_left = self.config.sl_cooldown_bars
            return

        if not isinstance(event, OrderFilled):
            return

        # Entry fill → place SL
        if (self._entry_order_id is not None
                and event.client_order_id == self._entry_order_id):
            self._entry_px       = float(event.last_px)
            self._entry_order_id = None
            self._place_sl(self._entry_side, self._entry_px)

        # SL fill → clear state, start cooldown
        elif (self._sl_order_id is not None
                and event.client_order_id == self._sl_order_id):
            be_tag = " [BREAKEVEN]" if self._at_breakeven else ""
            self.log.info(
                f"SL hit{be_tag} @ {event.last_px}  entry was {self._entry_px:.1f}",
                color=LogColor.YELLOW if self._at_breakeven else LogColor.RED,
            )
            self._sl_order_id        = None
            self._entry_side         = None
            self._peak_px            = 0.0
            self._trail_px           = 0.0
            self._trail_active       = False
            self._sl_distance        = 0.0
            self._be_target_px       = 0.0
            self._at_breakeven       = False
            # ── Cooldown: ignore new signals for N bars ────────────────
            if self.config.sl_cooldown_bars > 0:
                self._cooldown_bars_left = self.config.sl_cooldown_bars
                self.log.info(
                    f"Cooldown: skipping next {self._cooldown_bars_left} bar(s) "
                    f"after SL hit",
                    color=LogColor.BLUE,
                )

    # ------------------------------------------------------------------
    # Bar handler — EMA cross signals + trailing SL update
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

        # ── Tick down cooldown counter ────────────────────────────────
        if self._cooldown_bars_left > 0:
            self._cooldown_bars_left -= 1

        curr_bullish = self.fast_ema.value >= self.slow_ema.value

        # ── Trailing SL update (runs every bar while in a position) ───
        if (self._sl_order_id is not None
                and self._entry_side is not None
                and self._peak_px > 0.0):
            # Use the actual SL distance stored at entry time
            sl_dist = self._sl_distance   # absolute price distance

            # 1:1 trigger = entry + SL distance in the profit direction
            if self._entry_side == OrderSide.BUY:
                trigger_1to1 = self._entry_px + sl_dist
            else:
                trigger_1to1 = self._entry_px - sl_dist

            if self._entry_side == OrderSide.BUY:
                if not self._trail_active and float(bar.high) >= trigger_1to1:
                    self._trail_active = True
                    self.log.info(
                        f"Trail SL ACTIVATED at 1:1  "
                        f"(price={float(bar.high):.1f}  "
                        f"threshold={trigger_1to1:.1f})",
                        color=LogColor.CYAN,
                    )
                if self._trail_active:
                    new_peak  = max(self._peak_px, float(bar.high))
                    new_trail = new_peak - sl_dist
                    if new_trail > self._trail_px:
                        self._peak_px = new_peak
                        self._update_trailing_sl(new_trail)
            else:  # SHORT
                if not self._trail_active and float(bar.low) <= trigger_1to1:
                    self._trail_active = True
                    self.log.info(
                        f"Trail SL ACTIVATED at 1:1  "
                        f"(price={float(bar.low):.1f}  "
                        f"threshold={trigger_1to1:.1f})",
                        color=LogColor.CYAN,
                    )
                if self._trail_active:
                    new_peak  = min(self._peak_px, float(bar.low))
                    new_trail = new_peak + sl_dist
                    if new_trail < self._trail_px:
                        self._peak_px = new_peak
                        self._update_trailing_sl(new_trail)

        # First bar after warm-up: record state, don't trade
        if self._prev_bullish is None:
            self._prev_bullish = curr_bullish
            return

        # ── Crossover detection ───────────────────────────────────────
        crossed_up   = (not self._prev_bullish) and curr_bullish
        crossed_down = self._prev_bullish and (not curr_bullish)
        self._prev_bullish = curr_bullish

        if not (crossed_up or crossed_down):
            return

        # ── EMA gap filter ────────────────────────────────────────────
        if self.config.min_ema_gap_pct > 0.0:
            gap_pct = abs(self.fast_ema.value - self.slow_ema.value) / self.slow_ema.value
            if gap_pct < self.config.min_ema_gap_pct:
                self.log.info(
                    f"EMA gap filter: gap={gap_pct*100:.3f}% < "
                    f"min={self.config.min_ema_gap_pct*100:.3f}%  — skipping",
                    color=LogColor.BLUE,
                )
                return

        # ── RSI filter ────────────────────────────────────────────────
        if self.rsi.initialized:
            rsi_val = self.rsi.value
            if crossed_up and rsi_val > self.config.rsi_long_max:
                self.log.info(
                    f"RSI filter: RSI={rsi_val:.1f} > {self.config.rsi_long_max} "
                    f"(overbought) — skipping LONG",
                    color=LogColor.BLUE,
                )
                return
            if crossed_down and rsi_val < self.config.rsi_short_min:
                self.log.info(
                    f"RSI filter: RSI={rsi_val:.1f} < {self.config.rsi_short_min} "
                    f"(oversold) — skipping SHORT",
                    color=LogColor.BLUE,
                )
                return

        # ── Cooldown filter ───────────────────────────────────────────
        if self._cooldown_bars_left > 0:
            self.log.info(
                f"Cooldown: {self._cooldown_bars_left} bar(s) remaining — skipping",
                color=LogColor.BLUE,
            )
            return

        # ── Position management ───────────────────────────────────────
        is_flat  = self.portfolio.is_flat(self.config.instrument_id)
        is_long  = self.portfolio.is_net_long(self.config.instrument_id)
        is_short = self.portfolio.is_net_short(self.config.instrument_id)

        if crossed_up:
            if is_flat:
                self._enter(OrderSide.BUY, bar.close)
            elif is_short:
                self._exit_and_reverse(OrderSide.BUY, bar.close)
        else:  # crossed_down
            if is_flat:
                self._enter(OrderSide.SELL, bar.close)
            elif is_long:
                self._exit_and_reverse(OrderSide.SELL, bar.close)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(self, side: OrderSide, ref_price) -> None:
        """Submit a market entry order."""
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
            f"RSI={self.rsi.value:.1f}  "
            f"ATR={self.atr.value:.1f}",
            color=LogColor.GREEN if side == OrderSide.BUY else LogColor.RED,
        )

    def _sl_price_distance(self, entry_px: float) -> float:
        """
        Return the absolute price distance from entry to the initial SL.

        Uses ATR × multiplier when configured; otherwise falls back to
        stop_loss_pct × entry_px.
        """
        if self.config.atr_sl_multiplier > 0.0 and self.atr.initialized:
            dist = self.atr.value * self.config.atr_sl_multiplier
            # Sanity floor: never tighter than 0.1% of entry
            dist = max(dist, entry_px * 0.001)
            return dist
        return entry_px * self.config.stop_loss_pct

    def _place_sl(self, side: OrderSide, entry_px: float) -> None:
        """Submit a stop-market SL after the entry fills."""
        sl_dist = self._sl_price_distance(entry_px)
        be_rr   = self.config.breakeven_rr
        pp      = self.instrument.price_precision
        qty     = self.instrument.make_qty(self.config.trade_size)

        if side == OrderSide.BUY:
            raw_sl_price       = entry_px - sl_dist
            # Sanity: SL must be strictly below entry for a long
            if raw_sl_price >= entry_px:
                raw_sl_price = entry_px * (1.0 - self.config.stop_loss_pct)
            sl_price           = Price(raw_sl_price, precision=pp)
            exit_side          = OrderSide.SELL
            self._be_target_px = entry_px + sl_dist * be_rr if be_rr > 0 else 0.0
        else:
            raw_sl_price       = entry_px + sl_dist
            # Sanity: SL must be strictly above entry for a short
            if raw_sl_price <= entry_px:
                raw_sl_price = entry_px * (1.0 + self.config.stop_loss_pct)
            sl_price           = Price(raw_sl_price, precision=pp)
            exit_side          = OrderSide.BUY
            self._be_target_px = entry_px - sl_dist * be_rr if be_rr > 0 else 0.0

        self._sl_distance  = sl_dist
        self._at_breakeven = False
        self._peak_px      = entry_px
        self._trail_px     = float(sl_price)
        self._trail_active = False

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

        sl_pct_actual = sl_dist / entry_px * 100
        be_info = (f"  BE target={self._be_target_px:.1f}" if be_rr > 0 else "")
        self.log.info(
            f"SL placed @ {sl_price}  "
            f"(entry={entry_px:.1f}  dist={sl_dist:.1f}  "
            f"={sl_pct_actual:.2f}%{be_info})",
            color=LogColor.YELLOW,
        )

    def _move_sl_to_breakeven(self) -> None:
        """Cancel SL and replace it at entry price (zero risk)."""
        sl_order = self.cache.order(self._sl_order_id)
        if sl_order is None or not sl_order.is_open:
            return

        self.cancel_order(sl_order)
        self._sl_order_id = None

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
            f"(1:{self.config.breakeven_rr:.0f} R:R reached)",
            color=LogColor.GREEN,
        )

    def _update_trailing_sl(self, new_trail_px: float) -> None:
        """Cancel current SL and replace with a tighter trailing stop."""
        sl_order = self.cache.order(self._sl_order_id)
        if sl_order is None or not sl_order.is_open:
            return

        self.cancel_order(sl_order)
        self._sl_order_id = None

        pp        = self.instrument.price_precision
        qty       = self.instrument.make_qty(self.config.trade_size)
        trail_px  = Price(new_trail_px, precision=pp)
        exit_side = OrderSide.SELL if self._entry_side == OrderSide.BUY else OrderSide.BUY

        new_sl = self.order_factory.stop_market(
            instrument_id=self.config.instrument_id,
            order_side=exit_side,
            quantity=qty,
            trigger_price=trail_px,
            trigger_type=TriggerType.LAST_PRICE,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            tags=["SL", "TRAIL"],
        )
        self._sl_order_id = new_sl.client_order_id
        self._trail_px    = new_trail_px
        self.submit_order(new_sl)

        self.log.info(
            f"Trail SL -> {trail_px}  "
            f"(peak={self._peak_px:.1f}  "
            f"lock={self._sl_distance:.1f} from peak)",
            color=LogColor.CYAN,
        )

    def _exit_and_reverse(self, new_side: OrderSide, ref_price) -> None:
        """EMA reversal — cancel SL, close current position, open opposite."""
        if self._sl_order_id is not None:
            sl_order = self.cache.order(self._sl_order_id)
            if sl_order is not None and sl_order.is_open:
                self.cancel_order(sl_order)
            self._sl_order_id = None

        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

        self._entry_side         = None
        self._peak_px            = 0.0
        self._trail_px           = 0.0
        self._trail_active       = False
        self._sl_distance        = 0.0
        self._be_target_px       = 0.0
        self._at_breakeven       = False

        self._enter(new_side, ref_price)
