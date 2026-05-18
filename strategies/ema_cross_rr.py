#!/usr/bin/env python3
"""
EMA Cross Strategy — Pure Two-EMA Crossover
=============================================
Entry  : Market order on true EMA crossover (state-flip, not level).
Exit 1 : Two-phase trailing stop-loss.
           Phase 1 — fixed SL placed immediately after fill.
           Phase 2 — trail activates once price reaches 1:1 R:R.
Exit 2 : EMA crosses back → cancel SL, close at market, reverse.
Guard  : If SL order is rejected (price gaps past stop), position is
         closed at market immediately.
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
    fast_ema_period : int, default 9
        Fast EMA period.
    slow_ema_period : int, default 14
        Slow EMA period.
    stop_loss_pct : float, default 0.005
        Stop-loss distance as fraction of entry price (0.005 = 0.5%).
    close_positions_on_stop : bool, default True
        Close open positions when the strategy is stopped.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 9
    slow_ema_period: PositiveInt = 14
    stop_loss_pct: float = 0.005

    # live / paper trading helpers
    subscribe_quote_ticks: bool = False
    subscribe_trade_ticks: bool = False
    request_bars: bool = False
    close_positions_on_stop: bool = True


class EMACrossRR(Strategy):
    """Pure EMA(9/14) crossover strategy with two-phase trailing SL."""

    def __init__(self, config: EMACrossRRConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None

        # Indicators
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

        # Order tracking
        self._entry_order_id: ClientOrderId | None = None
        self._sl_order_id:    ClientOrderId | None = None
        self._entry_side:     OrderSide | None = None
        self._entry_px:       float = 0.0

        # Trailing SL state
        self._peak_px:      float = 0.0
        self._trail_px:     float = 0.0
        self._trail_active: bool  = False
        self._sl_distance:  float = 0.0   # absolute price distance at entry

        # Crossover memory — compare current bar to previous bar
        self._prev_bullish: bool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.config.instrument_id}")
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
        self._peak_px        = 0.0
        self._trail_px       = 0.0
        self._trail_active   = False
        self._sl_distance    = 0.0
        self._prev_bullish   = None

    # ------------------------------------------------------------------
    # Events — place SL after entry fill; handle SL fill / rejection
    # ------------------------------------------------------------------

    def on_event(self, event) -> None:
        # SL rejected (price gapped through stop) → close at market
        if isinstance(event, OrderRejected):
            if (self._sl_order_id is not None
                    and event.client_order_id == self._sl_order_id):
                self.log.warning("SL order REJECTED — closing position at market")
                self._sl_order_id = None
                self.cancel_all_orders(self.config.instrument_id)
                self.close_all_positions(self.config.instrument_id)
                self._entry_side   = None
                self._peak_px      = 0.0
                self._trail_px     = 0.0
                self._trail_active = False
                self._sl_distance  = 0.0
            return

        if not isinstance(event, OrderFilled):
            return

        # Entry fill → place initial SL
        if (self._entry_order_id is not None
                and event.client_order_id == self._entry_order_id):
            self._entry_px       = float(event.last_px)
            self._entry_order_id = None
            self._place_sl(self._entry_side, self._entry_px)

        # SL fill → clear all state
        elif (self._sl_order_id is not None
                and event.client_order_id == self._sl_order_id):
            self.log.info(
                f"SL hit @ {event.last_px}  entry was {self._entry_px:.1f}",
                color=LogColor.RED,
            )
            self._sl_order_id  = None
            self._entry_side   = None
            self._peak_px      = 0.0
            self._trail_px     = 0.0
            self._trail_active = False
            self._sl_distance  = 0.0

    # ------------------------------------------------------------------
    # Bar handler — EMA crossover signals + trailing SL update
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

        # ── Trailing SL update (every bar while in a position) ────────
        if (self._sl_order_id is not None
                and self._entry_side is not None
                and self._peak_px > 0.0):
            sl_dist = self._sl_distance

            if self._entry_side == OrderSide.BUY:
                trigger_1to1 = self._entry_px + sl_dist
                if not self._trail_active and float(bar.high) >= trigger_1to1:
                    self._trail_active = True
                    self.log.info(
                        f"Trail activated at 1:1  (price={float(bar.high):.1f}  "
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
                trigger_1to1 = self._entry_px - sl_dist
                if not self._trail_active and float(bar.low) <= trigger_1to1:
                    self._trail_active = True
                    self.log.info(
                        f"Trail activated at 1:1  (price={float(bar.low):.1f}  "
                        f"threshold={trigger_1to1:.1f})",
                        color=LogColor.CYAN,
                    )
                if self._trail_active:
                    new_peak  = min(self._peak_px, float(bar.low))
                    new_trail = new_peak + sl_dist
                    if new_trail < self._trail_px:
                        self._peak_px = new_peak
                        self._update_trailing_sl(new_trail)

        # First bar after warm-up — record state, skip trading
        if self._prev_bullish is None:
            self._prev_bullish = curr_bullish
            return

        # ── True crossover detection (state flip) ─────────────────────
        crossed_up   = (not self._prev_bullish) and curr_bullish
        crossed_down = self._prev_bullish and (not curr_bullish)
        self._prev_bullish = curr_bullish

        if not (crossed_up or crossed_down):
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
        else:
            if is_flat:
                self._enter(OrderSide.SELL, bar.close)
            elif is_long:
                self._exit_and_reverse(OrderSide.SELL, bar.close)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(self, side: OrderSide, ref_price) -> None:
        qty   = self.instrument.make_qty(self.config.trade_size)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
            tags=["ENTRY"],
        )
        self._entry_order_id = order.client_order_id
        self._entry_side     = side
        self.submit_order(order)
        self.log.info(
            f"{'BUY ' if side == OrderSide.BUY else 'SELL'} entry  "
            f"ref={float(ref_price):.1f}  "
            f"fast_ema={self.fast_ema.value:.1f}  "
            f"slow_ema={self.slow_ema.value:.1f}",
            color=LogColor.GREEN if side == OrderSide.BUY else LogColor.RED,
        )

    def _place_sl(self, side: OrderSide, entry_px: float) -> None:
        sl_dist   = entry_px * self.config.stop_loss_pct
        pp        = self.instrument.price_precision
        qty       = self.instrument.make_qty(self.config.trade_size)

        if side == OrderSide.BUY:
            sl_price  = Price(entry_px - sl_dist, precision=pp)
            exit_side = OrderSide.SELL
        else:
            sl_price  = Price(entry_px + sl_dist, precision=pp)
            exit_side = OrderSide.BUY

        self._sl_distance  = sl_dist
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
        self.log.info(
            f"SL placed @ {sl_price}  "
            f"(entry={entry_px:.1f}  dist={sl_dist:.1f}  "
            f"={self.config.stop_loss_pct*100:.2f}%)",
            color=LogColor.YELLOW,
        )

    def _update_trailing_sl(self, new_trail_px: float) -> None:
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
            f"Trail SL -> {trail_px}  (peak={self._peak_px:.1f})",
            color=LogColor.CYAN,
        )

    def _exit_and_reverse(self, new_side: OrderSide, ref_price) -> None:
        """EMA crossed back — cancel SL, close position, open opposite."""
        if self._sl_order_id is not None:
            sl_order = self.cache.order(self._sl_order_id)
            if sl_order is not None and sl_order.is_open:
                self.cancel_order(sl_order)
            self._sl_order_id = None

        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

        self._entry_side   = None
        self._peak_px      = 0.0
        self._trail_px     = 0.0
        self._trail_active = False
        self._sl_distance  = 0.0

        self._enter(new_side, ref_price)
