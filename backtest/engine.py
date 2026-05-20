"""
Unified Backtest Engine
========================
Single parameterized engine that replaces the previous inheritance chain:
  MultiTimeframeEngine → Round2Engine → RegimeEngine
  → IntradayAdaptiveEngine / CooldownEngine / ParamExploreEngine

All behavior is controlled via constructor parameters — no subclassing needed.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import research_config as config
from indicators import (
    prepare_indicators,
    check_exit_signal,
    calc_auto_lot_size,
    get_orb_strategy,
)
import indicators as signals_mod


# ═══════════════════════════════════════════════════════════════
# Data types (shared across the package)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    strategy: str
    direction: str        # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    lots: float
    sl_distance: float
    tp_distance: float
    sl_price: float = 0.0
    tp_price: float = 0.0
    extreme_price: float = 0.0
    trailing_stop_price: float = 0.0
    bars_held: int = 0
    entry_atr: float = 0.0

    def __post_init__(self):
        if self.direction == 'BUY':
            self.sl_price = self.entry_price - self.sl_distance
            self.tp_price = self.entry_price + self.tp_distance
            self.extreme_price = self.entry_price
        else:
            self.sl_price = self.entry_price + self.sl_distance
            self.tp_price = self.entry_price - self.tp_distance
            self.extreme_price = self.entry_price


@dataclass
class TradeRecord:
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lots: float
    pnl: float
    exit_reason: str
    bars_held: int


# ═══════════════════════════════════════════════════════════════
# Unified Engine
# ═══════════════════════════════════════════════════════════════

class BacktestEngine:
    """Dual-timeframe (M15 primary + H1) backtest engine.

    All previously separate engine behaviors are controlled by parameters:
      - trailing_activate_atr / trailing_distance_atr: trailing stop tuning
      - sl_atr_mult / tp_atr_mult: override signal SL/TP with ATR multiples
      - keltner_adx_threshold: override Keltner ADX entry filter
      - max_positions: override config.MAX_POSITIONS
      - cooldown_hours: override config.COOLDOWN_MINUTES (in hours)
      - regime_config: ATR-percentile based parameter adaptation
      - intraday_adaptive / choppy_threshold / kc_only_threshold: trend gating
      - min_entry_gap_hours: global entry cooldown
      - rsi_adx_filter / rsi_atr_pct_filter / etc.: M15 RSI filters
      - rsi_buy_threshold / rsi_sell_threshold: custom RSI thresholds
      - spread_cost: per-trade transaction cost
      - atr_regime_lots: ATR-based lot sizing
    """

    M15_WINDOW = 150
    H1_WINDOW = 150

    def __init__(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        *,
        # Trailing stop
        trailing_activate_atr: float = 0,
        trailing_distance_atr: float = 0,
        # SL/TP overrides
        sl_atr_mult: float = 0,
        tp_atr_mult: float = 0,
        # Keltner ADX threshold override
        keltner_adx_threshold: float = 0,
        # R178: Session-based Keltner ADX — {session_name: (start_hour, end_hour, adx_threshold)}
        keltner_session_adx: Optional[Dict[str, tuple]] = None,
        # Position / cooldown overrides
        max_positions: int = 0,
        cooldown_hours: float = 0,
        # Regime-adaptive (replaces RegimeEngine)
        regime_config: Optional[Dict] = None,
        # Intraday trend gating (replaces IntradayAdaptiveEngine)
        intraday_adaptive: bool = False,
        choppy_threshold: float = 0.35,
        kc_only_threshold: float = 0.60,
        # Global entry gap (replaces CooldownEngine)
        min_entry_gap_hours: float = 0,
        # M15 RSI filters
        rsi_adx_filter: float = 0,
        rsi_atr_pct_filter: float = 0,
        rsi_sell_enabled: bool = True,
        rsi_atr_pct_min_filter: float = 0,
        # Custom RSI thresholds (replaces ParamExploreEngine)
        rsi_buy_threshold: float = 0,
        rsi_sell_threshold: float = 0,
        # ORB
        orb_max_hold_m15: int = 0,
        # M15 RSI max hold override (in M15 bars, 0 = use default 15)
        rsi_max_hold_m15: int = 0,
        # Keltner max hold override (in M15 bars, 0 = use config default)
        keltner_max_hold_m15: int = 0,
        # KC bandwidth expanding filter: only enter Keltner when bandwidth is expanding
        kc_bw_filter_bars: int = 0,  # 0=disabled, N=require bw(now)>bw(N bars ago)
        # Session filter for H1 entries (UTC hours). Empty = no filter.
        h1_allowed_sessions: Optional[List[int]] = None,  # e.g. [7,8,9,...,20]
        # EMA slope filter: block BUY when EMA100 slope < 0 over N bars
        block_buy_ema_slope: int = 0,
        # Lot sizing
        atr_regime_lots: bool = False,
        # R157: Session-based lot sizing — map UTC hour -> lot size
        session_lot_map: Optional[Dict[int, float]] = None,
        # Transaction cost
        spread_cost: float = 0.0,
        spread_model: str = "fixed",        # "fixed" | "atr_scaled" | "session_aware" | "historical" | "realistic"
        spread_base: float = 0.30,          # base spread for dynamic models
        spread_max: float = 3.0,            # max spread cap
        spread_series: Optional[pd.Series] = None,  # historical spread indexed by timestamp(ms)
        # Entry slippage model based on 91 real trades from live EA
        # "none" = no slippage (default), "fixed" = fixed per-direction,
        # "empirical" = sample from real distribution, "realistic" = fixed mean values
        slippage_model: str = "none",
        slippage_buy: float = 0.67,         # mean BUY entry slippage from live data ($)
        slippage_sell: float = 0.17,         # mean SELL entry slippage from live data ($)
        slippage_seed: int = 42,             # RNG seed for empirical sampling
        # Macro regime (P4)
        macro_df: Optional[pd.DataFrame] = None,
        macro_regime_enabled: bool = False,
        # ATR spike protection: tighten trailing when intra-trade ATR spikes
        atr_spike_protection: bool = False,
        atr_spike_threshold: float = 1.5,   # ATR > entry_atr * threshold triggers
        atr_spike_trail_mult: float = 0.7,  # multiply trail_distance by this when spiked
        # Time-decay TP: gradually lower profit lock threshold over time
        time_decay_tp: bool = False,
        time_decay_start_hour: float = 1.0,   # hours before decay kicks in
        time_decay_atr_start: float = 0.30,   # min profit at start (ATR mult)
        time_decay_atr_step: float = 0.10,    # decay per hour (ATR mult)
        # Entry quality filters (anti false-breakout)
        min_h1_bars_today: int = 0,           # require N H1 bars before allowing entry (0=disabled)
        adx_gray_zone: float = 0,             # ADX in [threshold, threshold+gray_zone) requires higher trend_score
        adx_gray_zone_min_score: float = 0.50,  # min trend_score required in ADX gray zone
        escalating_cooldown: bool = False,     # double cooldown after 2nd same-day loss
        escalating_cooldown_mult: float = 4.0, # multiplier for escalated cooldown
        # Live parity: use rolling-50 ATR rank instead of precomputed rolling-500
        live_atr_percentile: bool = False,
        # Time-adaptive trailing: tighten trail as bars_held increases
        time_adaptive_trail: bool = False,
        time_adaptive_trail_start: int = 4,      # start tightening after N bars
        time_adaptive_trail_decay: float = 0.95,  # multiply dist by this per bar after start
        time_adaptive_trail_floor: float = 0.005,  # minimum trail_dist floor
        # Breakeven stop: move SL to entry after float profit exceeds threshold
        breakeven_after_atr: float = 0.0,  # 0=disabled; e.g. 0.5 = move SL to entry when profit >= 0.5*ATR
        # Pinbar confirmation: require Pinbar on preceding H1 bar aligned with entry direction
        pinbar_confirmation: bool = False,
        # Support/Resistance proximity filter: skip BUY near resistance, skip SELL near support
        sr_filter_atr: float = 0.0,  # 0=disabled; e.g. 1.5 = skip entry if within 1.5*ATR of S/R level
        # Pinbar standalone: enter on Pinbar at S/R zones (independent strategy mode)
        pinbar_sr_strategy: bool = False,
        pinbar_sr_atr_zone: float = 2.0,  # proximity to S/R in ATR units to trigger Pinbar entry
        # Fractal confirmation: require top/bottom fractal aligned with entry direction
        fractal_confirmation: bool = False,
        # Inside Bar confirmation: require inside-bar breakout aligned with entry direction
        inside_bar_confirmation: bool = False,
        # Engulfing/2B confirmation: require engulfing pattern aligned with entry direction
        engulf_confirmation: bool = False,
        # Any PA confirmation: require at least one price-action pattern (pinbar/fractal/inside/engulf)
        any_pa_confirmation: bool = False,
        # PA confluence filter: require N+ patterns in the same direction
        pa_confluence_min: int = 0,  # 0=disabled; 2=require at least 2 PA signals aligned
        # Daily range filter ($15 rule): skip entry if daily range already exceeded threshold
        daily_range_filter: float = 0.0,  # 0=disabled; e.g. 15.0 = skip if daily move > $15
        # S/R "Rule of Three" decay: ignore S/R level if touched 3+ times
        sr_touch_decay: int = 0,  # 0=disabled; 3=invalidate after 3 touches
        # Fractal + S/R standalone strategy: enter on fractal at S/R zones
        fractal_sr_strategy: bool = False,
        # Inside Bar + S/R standalone strategy
        inside_bar_sr_strategy: bool = False,
        # Engulfing + S/R standalone strategy
        engulf_sr_strategy: bool = False,
        # R12: Squeeze detector — only enter KC breakout when coming out of squeeze
        squeeze_filter: bool = False,
        squeeze_lookback: int = 20,  # bars to check for squeeze condition
        # R12: Consecutive bars outside KC — require N bars closing outside channel
        consecutive_outside_bars: int = 0,  # 0=disabled; 2=require 2 consecutive bars outside KC
        # R12: Session quality filter — restrict H1 entries to specific session hours (UTC)
        # Unlike h1_allowed_sessions (which blocks all), this records session for analysis
        entry_session_tag: bool = False,
        # R12: Partial TP — close half position at 1 ATR profit, trail the rest
        partial_tp_atr: float = 0.0,  # 0=disabled; e.g. 1.0 = close 50% at 1*ATR profit
        # R12: Profit drawdown exit — close if unrealized profit drops X% from peak
        profit_drawdown_pct: float = 0.0,  # 0=disabled; e.g. 0.50 = exit if profit retraces 50% from peak
        # R12: Adaptive max hold — shorten max hold if no profit after N bars
        adaptive_max_hold: bool = False,
        adaptive_max_hold_profit_bars: int = 4,  # if no profit after N bars, halve remaining hold time
        # R13: KC parameter override for grid search
        kc_ema_override: int = 0,
        kc_mult_override: float = 0.0,
        # R13: Dual KC — fast+slow Keltner in parallel
        dual_kc_mode: str = "",
        dual_kc_fast_ema: int = 15,
        dual_kc_fast_mult: float = 0.8,
        dual_kc_slow_ema: int = 35,
        dual_kc_slow_mult: float = 1.6,
        # R13: KC midline MA type for runner (ema / hma / kama)
        kc_ma_type: str = "ema",
        # R13: Gold-Silver ratio factor
        gsr_filter_enabled: bool = False,
        gsr_high_threshold: float = 80.0,
        gsr_low_threshold: float = 65.0,
        gsr_series: Optional[pd.Series] = None,
        # R13: Purged walk-forward embargo (harness only)
        purge_embargo_bars: int = 0,
        # R14: Capital & position sizing overrides (0 = use config defaults)
        risk_per_trade: float = 0.0,
        min_lot_size: float = 0.0,
        max_lot_size: float = 0.0,
        compounding: bool = False,
        initial_capital: float = 0.0,
        # R14: Progressive SL tightening (0 = disabled)
        progressive_sl_start_bar: int = 0,
        progressive_sl_target_mult: float = 0.0,
        progressive_sl_steps: int = 0,
        # R16: Timeout Sniper — dynamic timeout exit strategies
        timeout_profit_lock_atr: float = 0.0,  # lock profit at N*ATR before timeout (0=disabled)
        timeout_profit_lock_bar: int = 0,       # activate profit lock after N bars (0=use max_hold-2)
        timeout_adverse_exit: bool = False,      # exit early if adverse signal detected near timeout
        timeout_adverse_bar: int = 0,            # start checking adverse signals after N bars (0=max_hold//2)
        timeout_momentum_exit: bool = False,     # exit if momentum decays (close crosses EMA mid)
        timeout_momentum_bar: int = 0,           # start checking momentum after N bars (0=max_hold//2)
        timeout_dynamic: bool = False,           # dynamic max hold based on profit trajectory
        timeout_extend_bars: int = 4,            # extend max hold by N bars if profitable
        timeout_cut_bars: int = 4,               # cut max hold by N bars if adverse
        timeout_cut_threshold_atr: float = -0.5, # adverse threshold to cut hold (ATR mult, negative)
        timeout_extend_threshold_atr: float = 0.3,  # profit threshold to extend hold (ATR mult)
        # R17: Capital Curve Engineering — money management
        kelly_fraction: float = 0.0,       # Kelly sizing fraction (0=disabled; 0.25=quarter Kelly)
        drawdown_protection: bool = False,  # enable equity drawdown protection
        drawdown_max_pct: float = 0.10,    # pause trading if drawdown from peak > X%
        drawdown_reduce_pct: float = 0.05, # reduce lot size when drawdown > X%
        drawdown_reduce_factor: float = 0.5,  # multiply lots by this factor during drawdown
        anti_martingale: bool = False,       # increase size after wins, decrease after losses
        anti_martingale_win_mult: float = 1.2,  # lot multiplier after consecutive wins
        anti_martingale_loss_mult: float = 0.8, # lot multiplier after consecutive losses
        anti_martingale_max_streak: int = 3,    # cap streak effect at N consecutive
        profit_reinvest_pct: float = 0.0,  # fraction of profits to reinvest into capital (0-1)
        equity_curve_filter: bool = False,  # only trade when equity > its own MA
        equity_ma_period: int = 50,         # MA period for equity curve filter (in trades)
        # R53: MaxLoss Cap — per-trade floating loss hard limit (0 = disabled)
        maxloss_cap: float = 0,
        # R53B: Dynamic ATR Cap — maxloss = N * ATR * lots * POINT_VALUE (0 = use fixed cap)
        maxloss_cap_atr_mult: float = 0,
        # Performance: when no positions and not H1 boundary, skip the bar entirely.
        # Gives ~1.6x speedup by avoiding H1 window lookup + M15 window slicing.
        # Trade-off: M15-only signals (RSI) on non-H1 bars are not checked when flat.
        skip_non_h1_bars: bool = True,
        # M30 strategy integration (R236)
        m30_df: Optional[pd.DataFrame] = None,
        m30_enabled: bool = False,
        m30_signal_func: Optional[Callable] = None,
        m30_strategy_name: str = "m30_rsi14",
        m30_sl_atr_mult: float = 8.0,
        m30_tp_atr_mult: float = 8.0,
        m30_trail_activate_atr: float = 0.50,
        m30_trail_distance_atr: float = 0.15,
        m30_max_hold_m15: int = 96,  # 48 M30 bars = 96 M15 bars
        m30_cooldown_bars: int = 8,  # M30 bars cooldown (= 16 M15 bars)
        # Label
        label: str = "",
    ):
        self.m15_df = m15_df
        self.h1_df = h1_df
        self.h1_lookup = self._build_h1_lookup(h1_df)
        self.label = label

        # Trailing stop params
        self._trail_act = trailing_activate_atr
        self._trail_dist = trailing_distance_atr
        # Originals saved for regime reset
        self._trail_act_base = trailing_activate_atr
        self._trail_dist_base = trailing_distance_atr

        # SL/TP
        self._sl_atr_mult = sl_atr_mult
        self._sl_atr_mult_base = sl_atr_mult
        self._tp_atr_mult = tp_atr_mult

        # Keltner
        self._kc_adx_threshold = keltner_adx_threshold
        self._kc_session_adx = keltner_session_adx

        # Positions / cooldown
        self._max_pos = max_positions or config.MAX_POSITIONS
        self._cooldown_hours_override = cooldown_hours

        # Regime
        self._regime_config = regime_config

        # Intraday adaptive
        self._intraday_adaptive = intraday_adaptive
        self._choppy_threshold = choppy_threshold
        self._kc_only_threshold = kc_only_threshold
        self._current_score = 0.5
        self._current_regime = 'neutral'
        self._cached_date = None
        self._cached_h1_count = 0
        self._h1_date_map: Dict = {}
        if intraday_adaptive:
            self._precompute_h1_dates()

        # Global entry gap
        self._min_entry_gap_hours = min_entry_gap_hours
        self._last_entry_time = None

        # RSI filters
        self._rsi_adx_filter = rsi_adx_filter
        self._rsi_atr_pct_filter = rsi_atr_pct_filter
        self._rsi_sell_enabled = rsi_sell_enabled
        self._rsi_atr_pct_min_filter = rsi_atr_pct_min_filter
        self._rsi_buy_threshold = rsi_buy_threshold
        self._rsi_sell_threshold = rsi_sell_threshold

        # ORB
        self._orb_max_hold_m15 = orb_max_hold_m15
        self._rsi_max_hold_m15 = rsi_max_hold_m15
        self._keltner_max_hold_m15 = keltner_max_hold_m15
        self._kc_bw_filter_bars = kc_bw_filter_bars
        self._h1_allowed_sessions = set(h1_allowed_sessions) if h1_allowed_sessions else None
        self._block_buy_ema_slope = block_buy_ema_slope
        self.skipped_kc_bw = 0
        self.skipped_session = 0

        # Lots
        self._atr_regime_lots = atr_regime_lots
        self._session_lot_map = session_lot_map

        # Cost
        self._spread_cost = spread_cost
        self._spread_model = spread_model
        self._spread_base = spread_base
        self._spread_max = spread_max
        self._spread_series = spread_series

        # Entry slippage (from 91 real EA trades)
        self._slippage_model = slippage_model
        self._slippage_buy = slippage_buy
        self._slippage_sell = slippage_sell
        self._slippage_rng = np.random.RandomState(slippage_seed)
        # Empirical distributions from live trading (sorted, for resampling)
        self._empirical_buy_slips = np.array([
            -3.43, -1.29, -1.17, -0.82, -0.59, -0.53, -0.28, -0.22,
            0.18, 0.19, 0.23, 0.24, 0.26, 0.26, 0.29, 0.30, 0.32, 0.34,
            0.35, 0.41, 0.41, 0.49, 0.51, 0.61, 0.62, 0.62, 0.64, 0.64,
            0.67, 0.69, 0.73, 0.81, 0.82, 0.83, 0.87, 0.87, 0.87, 1.08,
            1.08, 1.12, 1.20, 1.23, 1.33, 1.86, 1.91, 2.26, 2.97, 3.02,
            3.20, 4.33,
        ])
        self._empirical_sell_slips = np.array([
            -2.59, -2.04, -1.61, -0.97, -0.78, -0.63, -0.61, -0.60,
            -0.51, -0.25, -0.24, -0.06, -0.05, -0.05, -0.04, -0.04, -0.04,
            0.00, 0.00, 0.17, 0.19, 0.26, 0.28, 0.28, 0.32, 0.40, 0.43,
            0.46, 0.47, 0.53, 0.58, 0.61, 0.82, 0.91, 0.97, 0.99, 1.04,
            1.10, 1.55, 2.33, 3.19,
        ])
        self.total_entry_slippage = 0.0
        self.slippage_count = 0

        # Macro regime
        self._macro_df = macro_df
        self._macro_regime_enabled = macro_regime_enabled
        self._macro_regime_detector = None
        if macro_regime_enabled and macro_df is not None:
            try:
                from macro.regime_detector import MacroRegimeDetector
                self._macro_regime_detector = MacroRegimeDetector()
            except ImportError:
                pass

        # ATR spike protection
        self._atr_spike_protection = atr_spike_protection
        self._atr_spike_threshold = atr_spike_threshold
        self._atr_spike_trail_mult = atr_spike_trail_mult

        # Time-decay TP
        self._time_decay_tp = time_decay_tp
        self._td_start_bars = int(time_decay_start_hour * 4)  # convert hours to M15 bars
        self._td_atr_start = time_decay_atr_start
        self._td_atr_step_per_bar = time_decay_atr_step / 4   # convert per-hour to per-M15-bar

        # Entry quality filters
        self._min_h1_bars_today = min_h1_bars_today
        self._adx_gray_zone = adx_gray_zone
        self._adx_gray_zone_min_score = adx_gray_zone_min_score
        self._escalating_cooldown = escalating_cooldown
        self._escalating_cooldown_mult = escalating_cooldown_mult
        self._live_atr_pct = live_atr_percentile

        # Time-adaptive trailing
        self._time_adaptive_trail = time_adaptive_trail
        self._ta_trail_start = time_adaptive_trail_start
        self._ta_trail_decay = time_adaptive_trail_decay
        self._ta_trail_floor = time_adaptive_trail_floor

        # Breakeven stop
        self._breakeven_after_atr = breakeven_after_atr
        self.breakeven_triggered = 0

        # Pinbar / S/R filters
        self._pinbar_confirmation = pinbar_confirmation
        self._sr_filter_atr = sr_filter_atr
        self._pinbar_sr_strategy = pinbar_sr_strategy
        self._pinbar_sr_atr_zone = pinbar_sr_atr_zone
        self.skipped_pinbar = 0
        self.skipped_sr = 0
        self.pinbar_sr_entries = 0

        # New PA filters
        self._fractal_confirmation = fractal_confirmation
        self._inside_bar_confirmation = inside_bar_confirmation
        self._engulf_confirmation = engulf_confirmation
        self._any_pa_confirmation = any_pa_confirmation
        self._pa_confluence_min = pa_confluence_min
        self._daily_range_filter = daily_range_filter
        self._sr_touch_decay = sr_touch_decay
        self._fractal_sr_strategy = fractal_sr_strategy
        self._inside_bar_sr_strategy = inside_bar_sr_strategy
        self._engulf_sr_strategy = engulf_sr_strategy
        self.skipped_fractal = 0
        self.skipped_inside_bar = 0
        self.skipped_engulf = 0
        self.skipped_pa_confluence = 0
        self.skipped_daily_range = 0
        self.fractal_sr_entries = 0
        self.inside_bar_sr_entries = 0
        self.engulf_sr_entries = 0

        # R12 features
        self._squeeze_filter = squeeze_filter
        self._squeeze_lookback = squeeze_lookback
        self._consecutive_outside_bars = consecutive_outside_bars
        self._entry_session_tag = entry_session_tag
        self._partial_tp_atr = partial_tp_atr
        self._profit_drawdown_pct = profit_drawdown_pct
        self._adaptive_max_hold = adaptive_max_hold
        self._adaptive_max_hold_profit_bars = adaptive_max_hold_profit_bars
        self.skipped_squeeze = 0
        self.skipped_consecutive = 0
        self.partial_tp_count = 0
        self.profit_dd_exit_count = 0
        self.adaptive_hold_triggered = 0
        self.session_entry_counts: Dict[str, int] = {}  # UTC hour -> count

        self.kc_ema_override = kc_ema_override
        self.kc_mult_override = kc_mult_override
        self._dual_kc_mode = dual_kc_mode
        self._dual_kc_fast_ema = dual_kc_fast_ema
        self._dual_kc_fast_mult = dual_kc_fast_mult
        self._dual_kc_slow_ema = dual_kc_slow_ema
        self._dual_kc_slow_mult = dual_kc_slow_mult
        self.kc_ma_type = kc_ma_type
        self._gsr_filter_enabled = gsr_filter_enabled
        self._gsr_high_threshold = gsr_high_threshold
        self._gsr_low_threshold = gsr_low_threshold
        self._gsr_series = gsr_series
        self.purge_embargo_bars = purge_embargo_bars
        self.dual_kc_filtered = 0
        self.dual_kc_added = 0
        self.skipped_gsr = 0

        # R14: Capital & sizing overrides
        self._risk_per_trade = risk_per_trade if risk_per_trade > 0 else config.RISK_PER_TRADE
        self._min_lot = min_lot_size if min_lot_size > 0 else config.MIN_LOT_SIZE
        self._max_lot = max_lot_size if max_lot_size > 0 else config.MAX_LOT_SIZE
        self._compounding = compounding
        self._initial_capital = initial_capital if initial_capital > 0 else config.CAPITAL
        self._current_capital = self._initial_capital
        self._realized_pnl = 0.0

        # R14: Progressive SL tightening
        self._prog_sl_start = progressive_sl_start_bar
        self._prog_sl_target = progressive_sl_target_mult
        self._prog_sl_steps = progressive_sl_steps

        # R16: Timeout Sniper
        self._timeout_profit_lock_atr = timeout_profit_lock_atr
        self._timeout_profit_lock_bar = timeout_profit_lock_bar
        self._timeout_adverse_exit = timeout_adverse_exit
        self._timeout_adverse_bar = timeout_adverse_bar
        self._timeout_momentum_exit = timeout_momentum_exit
        self._timeout_momentum_bar = timeout_momentum_bar
        self._timeout_dynamic = timeout_dynamic
        self._timeout_extend_bars = timeout_extend_bars
        self._timeout_cut_bars = timeout_cut_bars
        self._timeout_cut_threshold_atr = timeout_cut_threshold_atr
        self._timeout_extend_threshold_atr = timeout_extend_threshold_atr
        self.timeout_profit_lock_count = 0
        self.timeout_adverse_exit_count = 0
        self.timeout_momentum_exit_count = 0
        self.timeout_dynamic_extend_count = 0
        self.timeout_dynamic_cut_count = 0

        # R17: Capital Curve Engineering
        self._kelly_fraction = kelly_fraction
        self._drawdown_protection = drawdown_protection
        self._drawdown_max_pct = drawdown_max_pct
        self._drawdown_reduce_pct = drawdown_reduce_pct
        self._drawdown_reduce_factor = drawdown_reduce_factor
        self._anti_martingale = anti_martingale
        self._anti_martingale_win_mult = anti_martingale_win_mult
        self._anti_martingale_loss_mult = anti_martingale_loss_mult
        self._anti_martingale_max_streak = anti_martingale_max_streak
        self._profit_reinvest_pct = profit_reinvest_pct
        self._equity_curve_filter = equity_curve_filter
        self._equity_ma_period = equity_ma_period
        self._maxloss_cap = maxloss_cap
        self._maxloss_cap_atr_mult = maxloss_cap_atr_mult
        self.maxloss_cap_count = 0
        self._skip_non_h1_bars = skip_non_h1_bars
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._equity_peak = self._initial_capital
        self._trade_equity_history: List[float] = []
        self._trading_paused_dd = False
        self.dd_pause_count = 0
        self.dd_reduce_count = 0
        self.equity_filter_skip_count = 0

        # State
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.daily_loss_count = 0
        self.current_date = None
        self.cooldown_until: Dict[str, datetime] = {}

        # Pending entries: signals generated on bar[i] execute on bar[i+1].Open
        # to avoid using the current bar's Close as entry price (look-ahead)
        self._pending_signals: List[tuple] = []  # [(signals, source), ...]

        # Counters
        self.rsi_filtered_count = 0
        self.rsi_total_signals = 0
        self.h1_entry_count = 0
        self.m15_entry_count = 0
        self.skipped_choppy = 0
        self.skipped_neutral_m15 = 0
        self.skipped_ema_slope = 0
        self.atr_spike_tighten_count = 0
        self.time_decay_tp_count = 0
        self.skipped_min_bars = 0
        self.skipped_adx_gray = 0
        self.escalated_cooldowns = 0

        # M30 strategy (R236 integration)
        self._m30_enabled = m30_enabled
        self._m30_df = m30_df
        self._m30_signal_func = m30_signal_func
        self._m30_strategy_name = m30_strategy_name
        self._m30_sl_atr_mult = m30_sl_atr_mult
        self._m30_tp_atr_mult = m30_tp_atr_mult
        self._m30_trail_act = m30_trail_activate_atr
        self._m30_trail_dist = m30_trail_distance_atr
        self._m30_max_hold_m15 = m30_max_hold_m15
        self._m30_cooldown_m15 = m30_cooldown_bars * 2
        self._m30_last_entry_bar = -9999
        self.m30_total_signals = 0
        self.m30_entries = 0
        self.m30_skipped_choppy = 0
        self.m30_skipped_atr = 0
        self.m30_skipped_slot = 0
        self.m30_skipped_cooldown = 0
        if m30_enabled and m30_df is not None:
            self._m30_lookup = self._build_m30_lookup(m30_df)
        else:
            self._m30_lookup = {}

        # ── P1: H1 column pre-extraction ──────────────────────────
        # Profile showed 52,723 calls to fast_xs (h1_window.iloc[-1])
        # accounting for ~3.94s self time. Pre-extracting columns as
        # numpy arrays lets us use O(1) array[h1_idx] access instead.
        self._build_h1_arrays()

    def _build_h1_arrays(self):
        """Pre-extract hot H1 columns as numpy arrays for O(1) row access."""
        h1_df = self.h1_df
        n = len(h1_df)
        self._h1_n = n

        def _arr(col, default):
            if col in h1_df.columns:
                return np.ascontiguousarray(h1_df[col].to_numpy(dtype=np.float64))
            return np.full(n, default, dtype=np.float64)

        self._h1_atr_arr = _arr('ATR', 0.0)
        self._h1_atr_pct_arr = _arr('atr_percentile', 0.5)
        self._h1_adx_arr = _arr('ADX', 0.0)
        self._h1_ema100_arr = _arr('EMA100', np.nan)
        self._h1_kc_upper_arr = _arr('KC_upper', np.nan)
        self._h1_kc_lower_arr = _arr('KC_lower', np.nan)
        self._h1_kc_mid_arr = _arr('KC_mid', np.nan)
        self._h1_close_arr = _arr('Close', np.nan)

        # Pre-compute live atr percentile (rolling-50 rank) so
        # _get_atr_percentile_at() with live_atr_percentile=True is O(1)
        # instead of doing dropna + rolling mean on every call.
        if self._live_atr_pct:
            self._h1_atr_pct_live_arr = self._compute_live_atr_pct_array()
        else:
            self._h1_atr_pct_live_arr = None

        # Pre-compute "today's bar count up to and including h1_idx" so the
        # _check_h1_entries `min_h1_bars_today` filter is O(1).
        # Also start_of_day index and per-h1 score/regime for the
        # _update_intraday_score replacement (was 28% of run time).
        self._h1_start_of_day_arr = self._compute_start_of_day()
        self._h1_today_count_arr = (
            np.arange(n) - self._h1_start_of_day_arr + 1
        ).astype(np.int64)
        if self._intraday_adaptive:
            self._build_intraday_score_arrays()
        else:
            self._h1_score_arr = None
            self._h1_regime_arr = None

    def _compute_start_of_day(self) -> np.ndarray:
        """For each H1 bar, return the first index of the same UTC date."""
        h1 = self.h1_df
        n = len(h1)
        out = np.empty(n, dtype=np.int64)
        if n == 0:
            return out
        dates = h1.index.date
        prev_date = None
        cur_start = 0
        for i in range(n):
            d = dates[i]
            if d != prev_date:
                cur_start = i
                prev_date = d
            out[i] = cur_start
        return out

    def _build_intraday_score_arrays(self):
        """Vectorize _update_intraday_score: pre-compute score & regime per H1 bar.

        The legacy logic in _update_intraday_score / _calc_realtime_score is a
        pure function of H1 bars on the same date up to (and including) h1_idx.
        We compute it once at init so the runtime path becomes
        ``score, regime = self._h1_score_arr[i], self._h1_regime_arr[i]`` (O(1)).

        Profile (post-P1) attributed 12.5s / 28% of total run time to this
        path.  After this pre-computation runtime should drop to ~0.
        """
        h1 = self.h1_df
        n = self._h1_n
        if n == 0:
            self._h1_score_arr = np.full(0, 0.5, dtype=np.float64)
            self._h1_regime_arr = np.empty(0, dtype=object)
            return

        adx = self._h1_adx_arr
        close = self._h1_close_arr
        kc_u = self._h1_kc_upper_arr
        kc_l = self._h1_kc_lower_arr

        def _arr2(col):
            return (np.ascontiguousarray(h1[col].to_numpy(dtype=np.float64))
                    if col in h1.columns
                    else np.full(n, np.nan, dtype=np.float64))

        open_arr = _arr2('Open')
        high_arr = _arr2('High')
        low_arr = _arr2('Low')
        ema9_arr = _arr2('EMA9')
        ema21_arr = _arr2('EMA21')
        ema100_arr = self._h1_ema100_arr

        # KC break boolean -> float (0.0 / 1.0); NaN-safe -> 0.0
        valid_kc = ~(np.isnan(close) | np.isnan(kc_u) | np.isnan(kc_l))
        kc_break = np.where(
            valid_kc,
            ((close > kc_u) | (close < kc_l)).astype(np.float64),
            0.0,
        )

        valid_ema = ~(np.isnan(ema9_arr) | np.isnan(ema21_arr) | np.isnan(ema100_arr))
        ema_aligned = np.where(
            valid_ema,
            (((ema9_arr > ema21_arr) & (ema21_arr > ema100_arr))
             | ((ema9_arr < ema21_arr) & (ema21_arr < ema100_arr))).astype(np.float64),
            0.0,
        )

        cum_kc = np.zeros(n + 1, dtype=np.float64)
        cum_kc[1:] = np.cumsum(kc_break)
        cum_ema = np.zeros(n + 1, dtype=np.float64)
        cum_ema[1:] = np.cumsum(ema_aligned)

        sod = self._h1_start_of_day_arr
        n_bars = self._h1_today_count_arr

        adx_safe = np.where(np.isnan(adx), 20.0, adx)
        adx_score = np.minimum(adx_safe / 40.0, 1.0)

        # Range-sum / n_bars (kc_score, ema_score) — all vectorized
        denom = np.maximum(n_bars, 1).astype(np.float64)
        kc_score = np.minimum((cum_kc[np.arange(n) + 1] - cum_kc[sod]) / denom, 1.0)
        ema_score = (cum_ema[np.arange(n) + 1] - cum_ema[sod]) / denom

        # day_high / day_low / day_open via per-bar slice (Python loop, n=11k -> fast)
        score_arr = np.full(n, 0.5, dtype=np.float64)
        regime_arr = np.empty(n, dtype=object)
        regime_arr[:] = 'neutral'

        for i in range(n):
            if n_bars[i] < 2:
                score_arr[i] = 0.5
                regime_arr[i] = 'neutral'
                continue
            s = sod[i]
            day_open = open_arr[s]
            day_close = close[i]
            day_high = np.nanmax(high_arr[s:i + 1])
            day_low = np.nanmin(low_arr[s:i + 1])
            day_range = day_high - day_low
            if (not np.isnan(day_open) and not np.isnan(day_close)
                    and not np.isnan(day_range) and day_range > 0.01):
                ti = abs(day_close - day_open) / day_range
            else:
                ti = 0.0
            score = round(
                0.30 * adx_score[i] + 0.25 * kc_score[i]
                + 0.25 * ema_score[i] + 0.20 * ti,
                3,
            )
            score_arr[i] = score
            if score >= self._kc_only_threshold:
                regime_arr[i] = 'trending'
            elif score >= self._choppy_threshold:
                regime_arr[i] = 'neutral'
            else:
                regime_arr[i] = 'choppy'

        self._h1_score_arr = score_arr
        self._h1_regime_arr = regime_arr

    def _compute_live_atr_pct_array(self) -> np.ndarray:
        """Vectorize the live rolling-50 ATR percentile rank.

        Replicates the legacy logic in _get_atr_percentile():
            atr_series = h1_window['ATR'].dropna()
            if len(atr_series) >= 50:
                cur = atr_series.iloc[-1]
                return (atr_series.iloc[-50:] < cur).mean()
            return 0.5
        """
        atr = self._h1_atr_arr
        n = len(atr)
        out = np.full(n, 0.5, dtype=np.float64)
        nan_mask = np.isnan(atr)

        # Build "non-NaN" rolling buffer of 50 most recent values for each i.
        # For each H1 bar i, scan back to collect 50 non-NaN ATR values
        # ending at i.  ATR has 14-bar warmup so non-NaN density is ~100%
        # after that; scanning back at most 200 bars is safe.
        max_lookback = 200
        for i in range(n):
            if nan_mask[i]:
                continue
            start = max(0, i - max_lookback + 1)
            window = atr[start:i + 1]
            wmask = ~np.isnan(window)
            non_nan = window[wmask]
            if len(non_nan) >= 50:
                cur = non_nan[-1]
                last50 = non_nan[-50:]
                # Match legacy semantics: strict less-than count / 50
                out[i] = float((last50 < cur).sum()) / 50.0
        return out

    # ── P1: H1 array fast accessors ─────────────────────────────

    def _get_h1_atr_at(self, h1_idx: Optional[int]) -> float:
        """O(1) replacement for _get_h1_atr(h1_window) using pre-extracted arrays."""
        if h1_idx is None or h1_idx < 0 or h1_idx >= self._h1_n:
            return 0.0
        val = self._h1_atr_arr[h1_idx]
        return 0.0 if np.isnan(val) else float(val)

    def _get_atr_percentile_at(self, h1_idx: Optional[int]) -> float:
        """O(1) replacement for _get_atr_percentile(h1_window)."""
        if h1_idx is None or h1_idx < 0 or h1_idx >= self._h1_n:
            return 0.5
        if self._live_atr_pct:
            return float(self._h1_atr_pct_live_arr[h1_idx])
        val = self._h1_atr_pct_arr[h1_idx]
        return 0.5 if np.isnan(val) else float(val)

    def _get_kc_adx_for_hour(self, utc_hour: int) -> Optional[float]:
        """R178: Return Keltner ADX threshold for the given UTC hour.
        Uses session_adx map if configured, falls back to _kc_adx_threshold.
        Returns None to block entry (hour not in any session)."""
        if self._kc_session_adx:
            for _name, (start, end, adx_thr) in self._kc_session_adx.items():
                if start <= utc_hour <= end:
                    return float(adx_thr)
            return None
        if self._kc_adx_threshold > 0:
            return float(self._kc_adx_threshold)
        return float(signals_mod.ADX_TREND_THRESHOLD)

    # ── Main loop ─────────────────────────────────────────────

    def run(self) -> List[TradeRecord]:
        self._reset_global_state()
        total_bars = len(self.m15_df)
        lookback = self.M15_WINDOW
        max_pos = self._max_pos

        m15_start = self.m15_df.index[lookback]
        m15_end = self.m15_df.index[-1]
        print(f"  Backtest: {m15_start.strftime('%Y-%m-%d')} -> {m15_end.strftime('%Y-%m-%d')}")
        print(f"  M15 bars: {total_bars}, H1 bars: {len(self.h1_df)}")

        m15_index = self.m15_df.index
        m15_close_arr = self.m15_df['Close'].values.astype(np.float64)
        m15_open_arr = self.m15_df['Open'].values.astype(np.float64)
        m15_high_arr = self.m15_df['High'].values.astype(np.float64)
        m15_low_arr = self.m15_df['Low'].values.astype(np.float64)
        m15_is_flat_arr = self.m15_df['is_flat'].values if 'is_flat' in self.m15_df.columns else np.zeros(total_bars, dtype=bool)
        m15_minute_arr = m15_index.minute

        m15_dates = m15_index.date

        base_capital = config.CAPITAL
        self.skipped_no_signal = 0
        last_pct = 0

        for i in range(lookback, total_bars):
            pct = int((i - lookback) / (total_bars - lookback) * 100) // 10 * 10
            if pct > last_pct:
                print(f"    {pct}%...", end='', flush=True)
                last_pct = pct

            bar_time = m15_index[i]
            bar_date = m15_dates[i]

            if bar_date != self.current_date:
                self.current_date = bar_date
                self.daily_loss_count = 0

            if m15_is_flat_arr[i]:
                unrealized = self._calc_unrealized(m15_close_arr[i])
                self.equity_curve.append(base_capital + self._realized_pnl + unrealized)
                continue

            has_positions = len(self.positions) > 0
            has_pending = len(self._pending_signals) > 0
            is_h1_boundary = (m15_minute_arr[i] == 0)

            is_m30_boundary = self._m30_enabled and m15_minute_arr[i] in (15, 45)
            if (self._skip_non_h1_bars
                    and not has_positions and not has_pending
                    and not is_h1_boundary and not is_m30_boundary):
                self.equity_curve.append(base_capital + self._realized_pnl)
                self.skipped_no_signal += 1
                continue

            h1_idx, h1_window = self._get_h1_window_with_idx(bar_time)

            need_entries = (self.daily_loss_count < config.DAILY_MAX_LOSSES
                           and len(self.positions) < max_pos)

            if need_entries or has_positions or has_pending:
                m15_start_idx = max(0, i - self.M15_WINDOW + 1)
                m15_window = self.m15_df.iloc[m15_start_idx:i + 1]
            else:
                m15_window = None

            if has_pending:
                bar_open = m15_open_arr[i]
                for pending_sigs, pending_source in self._pending_signals:
                    self._process_signals(pending_sigs, bar_time, pending_source,
                                          entry_price_override=bar_open,
                                          h1_idx=h1_idx)
                self._pending_signals.clear()

            if has_positions and m15_window is not None:
                self._check_exits(m15_window, h1_window,
                                  self.m15_df.iloc[i], bar_time, h1_idx=h1_idx)

            if need_entries and m15_window is not None:
                h1_idx_closed = h1_idx - 1 if h1_idx is not None and h1_idx > 0 else None
                h1_window_closed = self._h1_window_from_idx(h1_idx_closed) if h1_idx_closed is not None else None

                if is_h1_boundary and h1_window_closed is not None and len(h1_window_closed) >= 50:
                    self._check_h1_entries(h1_window_closed, bar_time, h1_idx=h1_idx_closed)

                if len(m15_window) >= 105:
                    self._check_m15_entries(m15_window, h1_window_closed, bar_time, h1_idx=h1_idx_closed)

                # M30 signal check on M30 boundaries (when M15 minute is 15 or 45,
                # the previous M30 bar just closed at :00 or :30)
                if self._m30_enabled and m15_minute_arr[i] in (15, 45):
                    self._check_m30_entries(bar_time, i, h1_idx=h1_idx_closed)

            unrealized = self._calc_unrealized(m15_close_arr[i])
            self.equity_curve.append(base_capital + self._realized_pnl + unrealized)

        if self.positions:
            last_close = m15_close_arr[-1]
            last_time = m15_index[-1]
            for pos in list(self.positions):
                self._close_position(pos, last_close, last_time, "backtest_end")

        print(f" done! (skipped {self.skipped_no_signal} no-signal bars)")
        return self.trades

    # ── Exits ─────────────────────────────────────────────────

    def _check_exits(self, m15_window, h1_window, bar, bar_time, *,
                     h1_idx: Optional[int] = None):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        # Apply regime-adaptive parameters if configured
        if self._regime_config and h1_idx is not None and h1_idx >= 0:
            atr_pct = self._get_atr_percentile_at(h1_idx)
            regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
            rc = self._regime_config.get(regime, {})
            self._trail_act = rc.get('trail_act', self._trail_act_base)
            self._trail_dist = rc.get('trail_dist', self._trail_dist_base)
            self._sl_atr_mult = rc.get('sl', self._sl_atr_mult_base)

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            # 1. SL/TP
            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = "SL"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = "TP"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = "SL"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = "TP"
                    exit_price = pos.tp_price

            # 1a. MaxLoss Cap — per-trade floating loss hard limit
            cap_limit = self._maxloss_cap
            if self._maxloss_cap_atr_mult > 0:
                atr_cap = self._get_h1_atr_at(h1_idx)
                if atr_cap > 0:
                    cap_limit = self._maxloss_cap_atr_mult * atr_cap * pos.lots * config.POINT_VALUE_PER_LOT
            if not reason and cap_limit > 0:
                if pos.direction == 'BUY':
                    float_pnl = (close - pos.entry_price) * pos.lots * config.POINT_VALUE_PER_LOT
                else:
                    float_pnl = (pos.entry_price - close) * pos.lots * config.POINT_VALUE_PER_LOT
                if float_pnl < -cap_limit:
                    reason = "MaxLossCap"
                    exit_price = close
                    self.maxloss_cap_count += 1

            # 1b. Breakeven stop: move SL to entry when profit exceeds threshold
            if not reason and self._breakeven_after_atr > 0 and pos.strategy == 'keltner':
                atr_be = self._get_h1_atr_at(h1_idx)
                if atr_be > 0:
                    be_threshold = atr_be * self._breakeven_after_atr
                    if pos.direction == 'BUY':
                        if high - pos.entry_price >= be_threshold and pos.sl_price < pos.entry_price:
                            pos.sl_price = pos.entry_price
                            self.breakeven_triggered += 1
                    else:
                        if pos.entry_price - low >= be_threshold and pos.sl_price > pos.entry_price:
                            pos.sl_price = pos.entry_price
                            self.breakeven_triggered += 1

            # 1c. Progressive SL tightening: reduce SL distance as bars_held increases
            if (not reason and self._prog_sl_start > 0 and self._prog_sl_steps > 0
                    and pos.bars_held >= self._prog_sl_start and pos.strategy == 'keltner'):
                atr_prog = self._get_h1_atr_at(h1_idx)
                if atr_prog > 0:
                    bars_past = pos.bars_held - self._prog_sl_start
                    step_size = (self._sl_atr_mult - self._prog_sl_target) / self._prog_sl_steps
                    new_mult = max(self._prog_sl_target,
                                   self._sl_atr_mult - step_size * min(bars_past, self._prog_sl_steps))
                    new_sl_dist = round(atr_prog * new_mult, 2)
                    new_sl_dist = max(signals_mod.ATR_SL_MIN, new_sl_dist)
                    if pos.direction == 'BUY':
                        new_sl = pos.entry_price - new_sl_dist
                        if new_sl > pos.sl_price:
                            pos.sl_price = new_sl
                    else:
                        new_sl = pos.entry_price + new_sl_dist
                        if new_sl < pos.sl_price:
                            pos.sl_price = new_sl

            # 2. Keltner trailing stop
            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self._trail_act or config.TRAILING_ACTIVATE_ATR
                dist_atr = self._trail_dist or config.TRAILING_DISTANCE_ATR
                atr = self._get_h1_atr_at(h1_idx)
                if atr > 0:
                    # ATR spike protection: tighten trailing when volatility surges
                    if (self._atr_spike_protection
                            and pos.entry_atr > 0
                            and atr > pos.entry_atr * self._atr_spike_threshold):
                        dist_atr = dist_atr * self._atr_spike_trail_mult
                        self.atr_spike_tighten_count += 1

                    # Time-adaptive trailing: tighten as position ages
                    if self._time_adaptive_trail and pos.bars_held > self._ta_trail_start:
                        decay_steps = pos.bars_held - self._ta_trail_start
                        dist_atr = max(self._ta_trail_floor,
                                       dist_atr * (self._ta_trail_decay ** decay_steps))

                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    if float_profit >= atr * act_atr:
                        trail_distance = atr * dist_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = "Trailing"
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = "Trailing"
                                exit_price = pos.trailing_stop_price

            # 3. Signal exit
            if not reason and pos.strategy == 'keltner':
                pass
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                if pos.bars_held > 1:
                    exit_sig = check_exit_signal(m15_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close
            elif not reason and pos.strategy not in ('keltner',):
                if h1_window is not None and len(h1_window) > 2:
                    exit_sig = check_exit_signal(h1_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close

            # 3b. Time-decay TP: shrink profit target for stalled positions
            if (not reason
                    and self._time_decay_tp
                    and pos.strategy == 'keltner'
                    and pos.bars_held >= self._td_start_bars):
                trailing_activated = (pos.trailing_stop_price > 0) if pos.direction == 'BUY' else (pos.trailing_stop_price > 0)
                if not trailing_activated:
                    atr_td = self._get_h1_atr_at(h1_idx)
                    if atr_td > 0:
                        decay_bars = pos.bars_held - self._td_start_bars
                        min_profit_atr = max(0.0, self._td_atr_start - decay_bars * self._td_atr_step_per_bar)
                        min_profit = atr_td * min_profit_atr
                        if pos.direction == 'BUY':
                            float_pnl = close - pos.entry_price
                        else:
                            float_pnl = pos.entry_price - close
                        if float_pnl >= min_profit and float_pnl > 0:
                            reason = "TimeDecayTP"
                            exit_price = close
                            self.time_decay_tp_count += 1

            # R12: Profit drawdown exit — close if unrealized profit retraces X% from peak
            if not reason and self._profit_drawdown_pct > 0 and pos.bars_held >= 2:
                if pos.direction == 'BUY':
                    peak_profit = pos.extreme_price - pos.entry_price
                    current_profit = close - pos.entry_price
                else:
                    peak_profit = pos.entry_price - pos.extreme_price
                    current_profit = pos.entry_price - close
                if peak_profit > 0 and current_profit > 0:
                    retrace = 1.0 - (current_profit / peak_profit)
                    if retrace >= self._profit_drawdown_pct:
                        reason = f"ProfitDD:{retrace:.0%}"
                        exit_price = close
                        self.profit_dd_exit_count += 1

            # 2b. M30 trailing stop
            if not reason and pos.strategy == self._m30_strategy_name and self._m30_enabled:
                act_atr = self._m30_trail_act
                dist_atr = self._m30_trail_dist
                atr = self._get_h1_atr_at(h1_idx)
                if atr > 0 and act_atr > 0 and dist_atr > 0:
                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    if float_profit >= atr * act_atr:
                        trail_distance = atr * dist_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = "Trailing"
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = "Trailing"
                                exit_price = pos.trailing_stop_price

            # 4. Time stop
            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = self._rsi_max_hold_m15 if self._rsi_max_hold_m15 > 0 else 15
                elif pos.strategy == 'orb' and self._orb_max_hold_m15 > 0:
                    max_hold = self._orb_max_hold_m15
                elif pos.strategy == 'keltner' and self._keltner_max_hold_m15 > 0:
                    max_hold = self._keltner_max_hold_m15
                elif pos.strategy == self._m30_strategy_name and self._m30_enabled:
                    max_hold = self._m30_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4

                # R12: Adaptive max hold — halve remaining time if no profit after N bars
                if self._adaptive_max_hold and pos.strategy == 'keltner':
                    check_bar = self._adaptive_max_hold_profit_bars
                    if pos.bars_held == check_bar:
                        if pos.direction == 'BUY':
                            profit = close - pos.entry_price
                        else:
                            profit = pos.entry_price - close
                        if profit <= 0:
                            remaining = max_hold - check_bar
                            max_hold = check_bar + remaining // 2
                            self.adaptive_hold_triggered += 1

                # R16: Dynamic timeout — extend if profitable, cut if adverse
                if self._timeout_dynamic and pos.strategy == 'keltner':
                    atr_dyn = self._get_h1_atr_at(h1_idx)
                    if atr_dyn > 0:
                        if pos.direction == 'BUY':
                            pnl_atr = (close - pos.entry_price) / atr_dyn
                        else:
                            pnl_atr = (pos.entry_price - close) / atr_dyn
                        if pnl_atr >= self._timeout_extend_threshold_atr:
                            max_hold += self._timeout_extend_bars
                            self.timeout_dynamic_extend_count += 1
                        elif pnl_atr <= self._timeout_cut_threshold_atr:
                            max_hold = max(pos.bars_held + 1, max_hold - self._timeout_cut_bars)
                            self.timeout_dynamic_cut_count += 1

                # R16: Pre-timeout profit lock
                if (not reason and self._timeout_profit_lock_atr > 0
                        and pos.strategy == 'keltner'):
                    lock_bar = self._timeout_profit_lock_bar if self._timeout_profit_lock_bar > 0 else max(1, max_hold - 2)
                    if pos.bars_held >= lock_bar:
                        atr_lock = self._get_h1_atr_at(h1_idx)
                        if atr_lock > 0:
                            if pos.direction == 'BUY':
                                float_pnl = close - pos.entry_price
                            else:
                                float_pnl = pos.entry_price - close
                            if float_pnl >= atr_lock * self._timeout_profit_lock_atr:
                                reason = "TimeoutProfitLock"
                                exit_price = close
                                self.timeout_profit_lock_count += 1

                # R16: Adverse signal exit near timeout
                if (not reason and self._timeout_adverse_exit
                        and pos.strategy == 'keltner' and h1_window is not None and len(h1_window) >= 2):
                    adv_bar = self._timeout_adverse_bar if self._timeout_adverse_bar > 0 else max(1, max_hold // 2)
                    if pos.bars_held >= adv_bar:
                        last_h1 = h1_window.iloc[-1]
                        kc_mid = float(last_h1.get('KC_mid', 0))
                        if kc_mid > 0:
                            if pos.direction == 'BUY' and close < kc_mid:
                                reason = "TimeoutAdverse"
                                exit_price = close
                                self.timeout_adverse_exit_count += 1
                            elif pos.direction == 'SELL' and close > kc_mid:
                                reason = "TimeoutAdverse"
                                exit_price = close
                                self.timeout_adverse_exit_count += 1

                # R16: Momentum decay exit
                if (not reason and self._timeout_momentum_exit
                        and pos.strategy == 'keltner' and h1_window is not None and len(h1_window) >= 2):
                    mom_bar = self._timeout_momentum_bar if self._timeout_momentum_bar > 0 else max(1, max_hold // 2)
                    if pos.bars_held >= mom_bar:
                        last_h1 = h1_window.iloc[-1]
                        ema100 = float(last_h1.get('EMA100', 0))
                        if ema100 > 0:
                            if pos.direction == 'BUY' and close < ema100:
                                reason = "MomentumDecay"
                                exit_price = close
                                self.timeout_momentum_exit_count += 1
                            elif pos.direction == 'SELL' and close > ema100:
                                reason = "MomentumDecay"
                                exit_price = close
                                self.timeout_momentum_exit_count += 1

                if pos.bars_held >= max_hold:
                    reason = f"Timeout:{pos.bars_held}>={max_hold}"
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason, h1_idx=h1_idx)

    def _dual_kc_levels_last(self, h1_window: pd.DataFrame, utc_hour: int = -1) -> Optional[Dict]:
        need = max(self._dual_kc_fast_ema, self._dual_kc_slow_ema) + 2
        if h1_window is None or len(h1_window) < need:
            return None
        if 'ATR' not in h1_window.columns or 'Close' not in h1_window.columns:
            return None
        close_s = h1_window['Close'].astype(float)
        atr_s = h1_window['ATR'].astype(float)
        fm = close_s.ewm(span=self._dual_kc_fast_ema, adjust=False).mean()
        sm = close_s.ewm(span=self._dual_kc_slow_ema, adjust=False).mean()
        fu = fm + self._dual_kc_fast_mult * atr_s
        fl = fm - self._dual_kc_fast_mult * atr_s
        su = sm + self._dual_kc_slow_mult * atr_s
        sl_ = sm - self._dual_kc_slow_mult * atr_s
        row = h1_window.iloc[-1]
        close = float(row['Close'])
        ema100 = float(row.get('EMA100', np.nan))
        adx = float(row.get('ADX', np.nan))
        if any(pd.isna(x) for x in [close, ema100, adx]):
            return None
        fu_v = float(fu.iloc[-1])
        fl_v = float(fl.iloc[-1])
        su_v = float(su.iloc[-1])
        sl_v = float(sl_.iloc[-1])
        sm_v = float(sm.iloc[-1])
        if any(pd.isna(v) for v in [fu_v, fl_v, su_v, sl_v, sm_v]):
            return None
        if utc_hour >= 0 and self._kc_session_adx:
            adx_th = self._get_kc_adx_for_hour(utc_hour)
            if adx_th is None:
                return None
        elif self._kc_adx_threshold > 0:
            adx_th = self._kc_adx_threshold
        else:
            adx_th = signals_mod.ADX_TREND_THRESHOLD
        if adx < adx_th:
            return None
        fast_buy = close > fu_v and close > ema100
        fast_sell = close < fl_v and close < ema100
        slow_buy = close > su_v and close > ema100
        slow_sell = close < sl_v and close < ema100
        return {
            'close': close,
            'ema100': ema100,
            'slow_mid': sm_v,
            'fast_buy': fast_buy,
            'fast_sell': fast_sell,
            'slow_buy': slow_buy,
            'slow_sell': slow_sell,
        }

    def _gsr_value_at_bar(self, bar_time) -> Optional[float]:
        if self._gsr_series is None or len(self._gsr_series) == 0:
            return None
        s = self._gsr_series.sort_index()
        ts = pd.Timestamp(bar_time)
        try:
            idx = int(s.index.searchsorted(ts, side='right')) - 1
        except (TypeError, ValueError):
            return None
        if idx < 0:
            return None
        v = s.iloc[idx]
        if pd.isna(v):
            return None
        return float(v)

    # ── H1 Entries ────────────────────────────────────────────

    def _check_h1_entries(self, h1_window, bar_time, *, h1_idx: Optional[int] = None):
        if len(self.positions) >= self._max_pos:
            return

        # Session filter
        if self._h1_allowed_sessions is not None:
            hour = pd.Timestamp(bar_time).hour
            if hour not in self._h1_allowed_sessions:
                self.skipped_session += 1
                return

        # Intraday trend gating
        if self._intraday_adaptive:
            self._update_intraday_score(h1_window, bar_time)
            if self._current_regime == 'choppy':
                self.skipped_choppy += 1
                return

        # Min H1 bars today: don't trade before enough intraday data
        if self._min_h1_bars_today > 0 and self._intraday_adaptive:
            if h1_idx is not None and 0 <= h1_idx < self._h1_n:
                today_count = int(self._h1_today_count_arr[h1_idx])
            else:
                bar_date = pd.Timestamp(bar_time).date()
                indices = self._h1_date_map.get(bar_date, [])
                h1_time = pd.Timestamp(bar_time).floor('h')
                max_idx = self.h1_lookup.get(h1_time, -1)
                today_count = len([i for i in indices if i <= max_idx]) if max_idx >= 0 else 0
            if today_count < self._min_h1_bars_today:
                self.skipped_min_bars += 1
                return

        # Resolve UTC hour once for session ADX
        _bar_utc_hour = pd.Timestamp(bar_time).hour

        # ADX gray zone: require higher trend_score when ADX is marginal
        if self._adx_gray_zone > 0 and self._intraday_adaptive and h1_idx is not None and h1_idx >= 0:
            adx_val = self._h1_adx_arr[h1_idx]
            if np.isnan(adx_val):
                adx_val = 0.0
            adx_threshold = self._get_kc_adx_for_hour(_bar_utc_hour) or signals_mod.ADX_TREND_THRESHOLD
            if adx_threshold <= adx_val < adx_threshold + self._adx_gray_zone:
                if self._current_score < self._adx_gray_zone_min_score:
                    self.skipped_adx_gray += 1
                    return

        # Macro regime gating
        if self._macro_regime_enabled and self._macro_regime_detector and self._macro_df is not None:
            try:
                bar_date = pd.Timestamp(bar_time).normalize()
                if bar_date.tz is not None:
                    bar_date = bar_date.tz_localize(None)
                if bar_date in self._macro_df.index:
                    macro_row = self._macro_df.loc[bar_date]
                    m_regime = self._macro_regime_detector.detect_from_row(macro_row)
                    m_weights = self._macro_regime_detector.get_strategy_weights(m_regime)
                    if not m_weights.get('allow_trading', True):
                        self.skipped_choppy += 1
                        return
            except Exception:
                pass

        # Regime-based disable
        if self._regime_config and h1_idx is not None and h1_idx >= 0:
            atr_pct = self._get_atr_percentile_at(h1_idx)
            regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
            rc = self._regime_config.get(regime, {})
            if rc.get('disable_keltner', False):
                return
            if rc.get('keltner_adx', 0) > 0:
                self._kc_adx_threshold = rc['keltner_adx']

        # Keltner ADX threshold override (R178: session-aware)
        _effective_adx = self._get_kc_adx_for_hour(_bar_utc_hour)
        if _effective_adx is None:
            return  # hour not in any session → block Keltner entry
        if _effective_adx != signals_mod.ADX_TREND_THRESHOLD:
            old_threshold = signals_mod.ADX_TREND_THRESHOLD
            signals_mod.ADX_TREND_THRESHOLD = _effective_adx
            signals = signals_mod.scan_all_signals(h1_window, 'H1')
            signals_mod.ADX_TREND_THRESHOLD = old_threshold
        else:
            signals = signals_mod.scan_all_signals(h1_window, 'H1')

        if not signals:
            return

        # KC bandwidth expanding filter: block keltner entries when bandwidth is contracting
        if (self._kc_bw_filter_bars > 0 and h1_idx is not None
                and h1_idx - self._kc_bw_filter_bars >= 0):
            prev_idx = h1_idx - self._kc_bw_filter_bars
            kc_u = self._h1_kc_upper_arr[h1_idx]
            kc_l = self._h1_kc_lower_arr[h1_idx]
            kc_m = self._h1_kc_mid_arr[h1_idx]
            kc_u_prev = self._h1_kc_upper_arr[prev_idx]
            kc_l_prev = self._h1_kc_lower_arr[prev_idx]
            kc_m_prev = self._h1_kc_mid_arr[prev_idx]
            if (not (np.isnan(kc_u) or np.isnan(kc_l) or np.isnan(kc_m)
                     or np.isnan(kc_u_prev) or np.isnan(kc_l_prev) or np.isnan(kc_m_prev))
                    and kc_m > 0 and kc_m_prev > 0):
                bw_now = (kc_u - kc_l) / kc_m
                bw_prev = (kc_u_prev - kc_l_prev) / kc_m_prev
                if bw_now <= bw_prev:
                    signals = [s for s in signals if s.get('strategy') != 'keltner']
                    if not signals:
                        self.skipped_kc_bw += 1
                        return

        if self._dual_kc_mode and h1_window is not None and len(h1_window) > 0:
            dm = self._dual_kc_mode.strip().lower()
            lv = self._dual_kc_levels_last(h1_window, utc_hour=pd.Timestamp(bar_time).hour)
            if lv is not None:
                filtered = []
                for sig in signals:
                    if sig.get('strategy') != 'keltner':
                        filtered.append(sig)
                        continue
                    d = sig.get('signal', '')
                    ok = True
                    if dm == 'union':
                        ok = (d == 'BUY' and (lv['fast_buy'] or lv['slow_buy'])) or (
                            d == 'SELL' and (lv['fast_sell'] or lv['slow_sell']))
                    elif dm == 'intersect':
                        ok = (d == 'BUY' and lv['fast_buy'] and lv['slow_buy']) or (
                            d == 'SELL' and lv['fast_sell'] and lv['slow_sell'])
                    elif dm == 'fast_confirmed':
                        ok = (d == 'BUY' and lv['close'] > lv['slow_mid']) or (
                            d == 'SELL' and lv['close'] < lv['slow_mid'])
                    if ok:
                        filtered.append(sig)
                    else:
                        self.dual_kc_filtered += 1
                signals = filtered
                if not signals:
                    return

        if self._gsr_filter_enabled and self._gsr_series is not None and len(self._gsr_series) > 0:
            gsr_val = self._gsr_value_at_bar(bar_time)
            if gsr_val is not None and not pd.isna(gsr_val):
                filtered_gsr = []
                for sig in signals:
                    d = sig.get('signal', '')
                    if d == 'BUY' and gsr_val > self._gsr_high_threshold:
                        self.skipped_gsr += 1
                    elif d == 'SELL' and gsr_val < self._gsr_low_threshold:
                        self.skipped_gsr += 1
                    else:
                        filtered_gsr.append(sig)
                signals = filtered_gsr
                if not signals:
                    return

        # EMA slope filter: block BUY when EMA100 is declining
        if (self._block_buy_ema_slope > 0 and h1_idx is not None
                and h1_idx - self._block_buy_ema_slope + 1 >= 0):
            ema_now = self._h1_ema100_arr[h1_idx]
            ema_prev = self._h1_ema100_arr[h1_idx - self._block_buy_ema_slope + 1]
            if not (np.isnan(ema_now) or np.isnan(ema_prev)) and ema_now < ema_prev:
                signals = [s for s in signals if s.get('direction') != 'BUY']
                if not signals:
                    self.skipped_ema_slope += 1
                    return

        # Pinbar confirmation filter: require aligned Pinbar on prior H1 bar
        if self._pinbar_confirmation and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY' and float(prev.get('pinbar_bull', 0)) > 0:
                    filtered_sigs.append(sig)
                elif d == 'SELL' and float(prev.get('pinbar_bear', 0)) > 0:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_pinbar += 1
            signals = filtered_sigs
            if not signals:
                return

        # S/R proximity filter: skip BUY near resistance, skip SELL near support
        if self._sr_filter_atr > 0 and h1_window is not None and len(h1_window) >= 1:
            row = h1_window.iloc[-1]
            dist_r = float(row.get('dist_to_resistance', 999))
            dist_s = float(row.get('dist_to_support', 999))
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY' and not pd.isna(dist_r) and 0 < dist_r < self._sr_filter_atr:
                    self.skipped_sr += 1
                elif d == 'SELL' and not pd.isna(dist_s) and 0 < dist_s < self._sr_filter_atr:
                    self.skipped_sr += 1
                else:
                    filtered_sigs.append(sig)
            signals = filtered_sigs
            if not signals:
                return

        # Fractal confirmation filter
        if self._fractal_confirmation and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY' and float(prev.get('bot_fractal', 0)) > 0:
                    filtered_sigs.append(sig)
                elif d == 'SELL' and float(prev.get('top_fractal', 0)) > 0:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_fractal += 1
            signals = filtered_sigs
            if not signals:
                return

        # Inside Bar confirmation filter
        if self._inside_bar_confirmation and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY' and float(prev.get('inside_bar_bull', 0)) > 0:
                    filtered_sigs.append(sig)
                elif d == 'SELL' and float(prev.get('inside_bar_bear', 0)) > 0:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_inside_bar += 1
            signals = filtered_sigs
            if not signals:
                return

        # Engulfing/2B confirmation filter
        if self._engulf_confirmation and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY' and float(prev.get('engulf_bull', 0)) > 0:
                    filtered_sigs.append(sig)
                elif d == 'SELL' and float(prev.get('engulf_bear', 0)) > 0:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_engulf += 1
            signals = filtered_sigs
            if not signals:
                return

        # Any-PA confirmation: at least one pattern matches direction
        if self._any_pa_confirmation and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                has_pa = False
                if d == 'BUY':
                    has_pa = any(float(prev.get(k, 0)) > 0
                                for k in ('pinbar_bull', 'bot_fractal', 'inside_bar_bull', 'engulf_bull'))
                elif d == 'SELL':
                    has_pa = any(float(prev.get(k, 0)) > 0
                                for k in ('pinbar_bear', 'top_fractal', 'inside_bar_bear', 'engulf_bear'))
                if has_pa:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_pa_confluence += 1
            signals = filtered_sigs
            if not signals:
                return

        # PA confluence filter: require N+ patterns aligned
        if self._pa_confluence_min > 0 and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            filtered_sigs = []
            for sig in signals:
                d = sig.get('signal', '')
                if d == 'BUY':
                    cnt = float(prev.get('pa_bull_count', 0))
                elif d == 'SELL':
                    cnt = float(prev.get('pa_bear_count', 0))
                else:
                    cnt = 0
                if cnt >= self._pa_confluence_min:
                    filtered_sigs.append(sig)
                else:
                    self.skipped_pa_confluence += 1
            signals = filtered_sigs
            if not signals:
                return

        # R12: Squeeze filter — only allow KC breakout after period of compression
        if self._squeeze_filter and h1_window is not None and len(h1_window) >= self._squeeze_lookback + 1:
            recent = h1_window.iloc[-(self._squeeze_lookback + 1):-1]
            if 'KC_upper' in recent.columns and 'KC_lower' in recent.columns:
                kc_bw = (recent['KC_upper'] - recent['KC_lower']).values
                if len(kc_bw) >= 2:
                    bw_now = kc_bw[-1]
                    bw_min = kc_bw.min()
                    if bw_now > bw_min * 1.3:
                        self.skipped_squeeze += len(signals)
                        return

        # R12: Consecutive bars outside KC
        if self._consecutive_outside_bars > 0 and h1_window is not None:
            n_req = self._consecutive_outside_bars
            if len(h1_window) >= n_req + 1:
                filtered = []
                for sig in signals:
                    d = sig.get('signal', '')
                    bars = h1_window.iloc[-(n_req + 1):-1]
                    if d == 'BUY':
                        count = int((bars['Close'] > bars['KC_upper']).sum()) if 'KC_upper' in bars.columns else 0
                    elif d == 'SELL':
                        count = int((bars['Close'] < bars['KC_lower']).sum()) if 'KC_lower' in bars.columns else 0
                    else:
                        count = 0
                    if count >= n_req:
                        filtered.append(sig)
                    else:
                        self.skipped_consecutive += 1
                signals = filtered
                if not signals:
                    return

        # R12: Session tagging
        if self._entry_session_tag and h1_window is not None and len(h1_window) > 0:
            hour = h1_window.index[-1].hour
            if hour < 8:
                session = 'Asia'
            elif hour < 13:
                session = 'London'
            elif hour < 21:
                session = 'NY'
            else:
                session = 'OffHours'
            self.session_entry_counts[session] = self.session_entry_counts.get(session, 0) + len(signals)

        # Daily range filter ($15 rule)
        if self._daily_range_filter > 0 and h1_window is not None and len(h1_window) >= 1:
            row = h1_window.iloc[-1]
            dr = float(row.get('daily_max_range', 0))
            if dr > self._daily_range_filter:
                self.skipped_daily_range += len(signals)
                return

        # Pinbar + S/R standalone strategy: generate entries on Pinbar at S/R zones
        if self._pinbar_sr_strategy and h1_window is not None and len(h1_window) >= 2:
            prev = h1_window.iloc[-1]
            atr = float(prev.get('ATR', 0))
            if atr > 0:
                dist_s = float(prev.get('dist_to_support', 999))
                dist_r = float(prev.get('dist_to_resistance', 999))
                pb_bull = float(prev.get('pinbar_bull', 0))
                pb_bear = float(prev.get('pinbar_bear', 0))
                close = float(prev.get('Close', 0))
                ema100 = float(prev.get('EMA100', 0))
                if (pb_bull > 0 and not pd.isna(dist_s)
                        and 0 < dist_s < self._pinbar_sr_atr_zone
                        and close > ema100):
                    signals.append({
                        'strategy': 'pinbar_sr', 'signal': 'BUY',
                        'sl': round(atr * self._sl_atr_mult, 2) if self._sl_atr_mult else round(atr * 3.5, 2),
                        'tp': round(atr * self._tp_atr_mult, 2) if self._tp_atr_mult else round(atr * 8.0, 2),
                        'close': close,
                        'reason': f"PinbarSR BUY near support dist={dist_s:.1f}ATR",
                    })
                    self.pinbar_sr_entries += 1
                if (pb_bear > 0 and not pd.isna(dist_r)
                        and 0 < dist_r < self._pinbar_sr_atr_zone
                        and close < ema100):
                    signals.append({
                        'strategy': 'pinbar_sr', 'signal': 'SELL',
                        'sl': round(atr * self._sl_atr_mult, 2) if self._sl_atr_mult else round(atr * 3.5, 2),
                        'tp': round(atr * self._tp_atr_mult, 2) if self._tp_atr_mult else round(atr * 8.0, 2),
                        'close': close,
                        'reason': f"PinbarSR SELL near resistance dist={dist_r:.1f}ATR",
                    })
                    self.pinbar_sr_entries += 1

        # Helper for PA+SR standalone strategies
        def _pa_sr_entries(pattern_bull_key, pattern_bear_key, strategy_name, counter_attr):
            if h1_window is None or len(h1_window) < 2:
                return
            prev = h1_window.iloc[-1]
            atr = float(prev.get('ATR', 0))
            if atr <= 0:
                return
            dist_s = float(prev.get('dist_to_support', 999))
            dist_r = float(prev.get('dist_to_resistance', 999))
            bull = float(prev.get(pattern_bull_key, 0))
            bear = float(prev.get(pattern_bear_key, 0))
            close = float(prev.get('Close', 0))
            ema100 = float(prev.get('EMA100', 0))
            sl_m = self._sl_atr_mult if self._sl_atr_mult else 3.5
            tp_m = self._tp_atr_mult if self._tp_atr_mult else 8.0
            if (bull > 0 and not pd.isna(dist_s)
                    and 0 < dist_s < self._pinbar_sr_atr_zone
                    and close > ema100):
                signals.append({
                    'strategy': strategy_name, 'signal': 'BUY',
                    'sl': round(atr * sl_m, 2), 'tp': round(atr * tp_m, 2),
                    'close': close,
                    'reason': f"{strategy_name} BUY near support dist={dist_s:.1f}ATR",
                })
                setattr(self, counter_attr, getattr(self, counter_attr) + 1)
            if (bear > 0 and not pd.isna(dist_r)
                    and 0 < dist_r < self._pinbar_sr_atr_zone
                    and close < ema100):
                signals.append({
                    'strategy': strategy_name, 'signal': 'SELL',
                    'sl': round(atr * sl_m, 2), 'tp': round(atr * tp_m, 2),
                    'close': close,
                    'reason': f"{strategy_name} SELL near resistance dist={dist_r:.1f}ATR",
                })
                setattr(self, counter_attr, getattr(self, counter_attr) + 1)

        if self._fractal_sr_strategy:
            _pa_sr_entries('bot_fractal', 'top_fractal', 'fractal_sr', 'fractal_sr_entries')
        if self._inside_bar_sr_strategy:
            _pa_sr_entries('inside_bar_bull', 'inside_bar_bear', 'inside_bar_sr', 'inside_bar_sr_entries')
        if self._engulf_sr_strategy:
            _pa_sr_entries('engulf_bull', 'engulf_bear', 'engulf_sr', 'engulf_sr_entries')

        self._pending_signals.append((signals, 'H1'))

    # ── M15 Entries ───────────────────────────────────────────

    def _check_m15_entries(self, m15_window, h1_window, bar_time, *,
                           h1_idx: Optional[int] = None):
        if len(self.positions) >= self._max_pos:
            return

        # Intraday trend gating — skip M15 in neutral regime
        if self._intraday_adaptive:
            if self._current_regime == 'choppy':
                self.skipped_choppy += 1
                return
            if self._current_regime == 'neutral':
                self.skipped_neutral_m15 += 1
                return

        # Regime-based disable
        if self._regime_config and h1_idx is not None and h1_idx >= 0:
            atr_pct = self._get_atr_percentile_at(h1_idx)
            regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
            rc = self._regime_config.get(regime, {})
            if rc.get('disable_rsi', False):
                return

        # Custom RSI thresholds — bypass scan_all_signals
        if self._rsi_buy_threshold > 0 or self._rsi_sell_threshold > 0:
            self._check_m15_custom_rsi(m15_window, h1_window, bar_time, h1_idx=h1_idx)
            return

        signals = signals_mod.scan_all_signals(m15_window, 'M15')
        if not signals:
            return

        filtered = []
        for sig in signals:
            self.rsi_total_signals += 1
            blocked = False

            if not self._rsi_sell_enabled and sig.get('signal') == 'SELL':
                self.rsi_filtered_count += 1
                blocked = True

            if not blocked and self._rsi_adx_filter > 0 and h1_idx is not None and h1_idx >= 0:
                adx_val = self._h1_adx_arr[h1_idx]
                if not np.isnan(adx_val) and adx_val > self._rsi_adx_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked and self._rsi_atr_pct_filter > 0 and h1_idx is not None and h1_idx >= 0:
                atr_pct = self._get_atr_percentile_at(h1_idx)
                if atr_pct > self._rsi_atr_pct_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked and self._rsi_atr_pct_min_filter > 0 and h1_idx is not None and h1_idx >= 0:
                atr_pct = self._get_atr_percentile_at(h1_idx)
                if atr_pct < self._rsi_atr_pct_min_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked:
                filtered.append(sig)

        if filtered:
            self._pending_signals.append((filtered, 'M15'))

    # ── M30 Entries (R236 integration) ─────────────────────────

    def _check_m30_entries(self, bar_time, m15_bar_idx: int, *,
                           h1_idx: Optional[int] = None):
        """Check M30 strategy signals. Called on M30 boundaries (minute==0 or 30)."""
        if not self._m30_enabled or self._m30_df is None or self._m30_signal_func is None:
            return
        if len(self.positions) >= self._max_pos:
            self.m30_skipped_slot += 1
            return

        # Cooldown check
        if (m15_bar_idx - self._m30_last_entry_bar) < self._m30_cooldown_m15:
            self.m30_skipped_cooldown += 1
            return

        # Choppy Gate: block M30 entries in choppy regime
        if self._intraday_adaptive and self._current_regime == 'choppy':
            self.m30_skipped_choppy += 1
            return

        # ATR Percentile Floor: skip in low-vol
        if h1_idx is not None and h1_idx >= 0:
            atr_pct = self._get_atr_percentile_at(h1_idx)
            if atr_pct < 0.30:
                self.m30_skipped_atr += 1
                return

        # Find M30 bar for this time
        m30_time = bar_time.floor('30min')
        if m30_time not in self._m30_lookup:
            return
        m30_idx = self._m30_lookup[m30_time]

        # Need lookback window for signal function
        m30_window_start = max(0, m30_idx - 60)
        m30_window = self._m30_df.iloc[m30_window_start:m30_idx + 1]
        if len(m30_window) < 55:
            return

        sig = self._m30_signal_func(m30_window)
        if sig is None:
            return

        self.m30_total_signals += 1
        direction = sig.get('signal', '')
        if direction not in ('BUY', 'SELL'):
            return

        # Rule B check: ATR > 2.5σ → skip
        if h1_idx is not None and h1_idx >= 0:
            cur_atr = self._h1_atr_arr[h1_idx]
            if h1_idx >= 60:
                atr_slice = self._h1_atr_arr[max(0, h1_idx - 60):h1_idx]
                atr_mean = float(atr_slice.mean())
                atr_std = float(atr_slice.std())
                if atr_std > 0 and cur_atr > atr_mean + 2.5 * atr_std:
                    self.m30_skipped_atr += 1
                    return

        # Get M30 ATR for SL/TP calculation
        m30_atr = float(m30_window.iloc[-1].get('ATR', 0))
        if m30_atr <= 0:
            return

        sl_dist = m30_atr * self._m30_sl_atr_mult
        tp_dist = m30_atr * self._m30_tp_atr_mult

        signal_dict = {
            'signal': direction,
            'strategy': self._m30_strategy_name,
            'sl': sl_dist,
            'tp': tp_dist,
            '_m30_entry': True,
        }

        self._pending_signals.append(([signal_dict], 'M30'))
        self._m30_last_entry_bar = m15_bar_idx

    def _check_m15_custom_rsi(self, m15_window, h1_window, bar_time, *,
                              h1_idx: Optional[int] = None):
        """Custom RSI threshold logic (replaces ParamExploreEngine)."""
        latest = m15_window.iloc[-1]
        close = float(latest['Close'])
        rsi2 = float(latest['RSI2'])
        sma50 = float(latest['SMA50'])
        ema100 = float(latest['EMA100'])
        if pd.isna(rsi2) or pd.isna(sma50) or pd.isna(ema100):
            return

        h1_adx_val = 0.0
        if h1_idx is not None and h1_idx >= 0:
            v = self._h1_adx_arr[h1_idx]
            if not np.isnan(v):
                h1_adx_val = float(v)

        self.rsi_total_signals += 1

        if self._rsi_adx_filter > 0 and h1_adx_val > self._rsi_adx_filter:
            self.rsi_filtered_count += 1
            return

        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
        sl = round(atr_val * signals_mod.ATR_SL_MULTIPLIER, 2) if atr_val > 0 else 15
        sl = max(signals_mod.ATR_SL_MIN, min(signals_mod.ATR_SL_MAX, sl))

        buy_th = self._rsi_buy_threshold or 15
        sell_th = self._rsi_sell_threshold or 85

        sig = None
        if rsi2 < buy_th and close > sma50 and close > ema100:
            sig = {'strategy': 'm15_rsi', 'signal': 'BUY', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI BUY: RSI2={rsi2:.1f}<{buy_th}"}
        elif rsi2 > sell_th and close < sma50 and close < ema100:
            sig = {'strategy': 'm15_rsi', 'signal': 'SELL', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI SELL: RSI2={rsi2:.1f}>{sell_th}"}

        if sig:
            self._pending_signals.append(([sig], 'M15'))

    # ── Signal processing ─────────────────────────────────────

    def _process_signals(self, signals: List[Dict], bar_time, source: str,
                         entry_price_override: float = 0.0,
                         *, h1_idx: Optional[int] = None):
        # Global entry gap check
        if self._min_entry_gap_hours > 0 and self._last_entry_time is not None:
            gap = (pd.Timestamp(bar_time) - self._last_entry_time).total_seconds() / 3600
            if gap < self._min_entry_gap_hours:
                return

        if h1_idx is None:
            h1_idx = self._resolve_h1_idx(pd.Timestamp(bar_time))

        active_strategies = {p.strategy for p in self.positions}
        current_dir = self.positions[0].direction if self.positions else None
        slots = self._max_pos - len(self.positions)
        entered = False

        # Cache the H1 ATR once for this bar (used 0-2x in SL/TP override + entry_atr)
        bar_h1_atr = self._get_h1_atr_at(h1_idx)

        for sig in signals[:slots]:
            strategy = sig['strategy']
            direction = sig['signal']
            entry_price = entry_price_override if entry_price_override > 0 else sig['close']

            # Apply realistic entry slippage (worsens entry for trader)
            slip = self._calc_entry_slippage(direction)
            if slip != 0.0:
                if direction == 'BUY':
                    entry_price += slip
                else:
                    entry_price -= slip
                self.total_entry_slippage += abs(slip)
                self.slippage_count += 1

            sl = sig.get('sl', config.STOP_LOSS_PIPS)
            tp = sig.get('tp', 0)

            # SL/TP ATR overrides (skip for M30 signals which bring their own)
            is_m30_signal = sig.get('_m30_entry', False)
            if not is_m30_signal:
                if self._sl_atr_mult > 0 and bar_h1_atr > 0:
                    sl = round(bar_h1_atr * self._sl_atr_mult, 2)
                    sl = max(signals_mod.ATR_SL_MIN, min(signals_mod.ATR_SL_MAX, sl))
                if self._tp_atr_mult > 0 and bar_h1_atr > 0:
                    tp = round(bar_h1_atr * self._tp_atr_mult, 2)

            if tp <= 0:
                tp = sl * 2

            cooldown = self.cooldown_until.get(strategy)
            if cooldown and bar_time <= cooldown:
                continue
            if strategy in active_strategies:
                continue
            if current_dir and direction != current_dir:
                continue

            # R17: Equity drawdown protection — pause trading during deep drawdowns
            if self._drawdown_protection and self._current_capital < self._equity_peak * (1 - self._drawdown_max_pct):
                self._trading_paused_dd = True
                self.dd_pause_count += 1
                continue
            self._trading_paused_dd = False

            # R17: Equity curve filter — only trade when equity > its own MA
            if self._equity_curve_filter and len(self._trade_equity_history) >= self._equity_ma_period:
                eq_ma = sum(self._trade_equity_history[-self._equity_ma_period:]) / self._equity_ma_period
                if self._current_capital < eq_ma:
                    self.equity_filter_skip_count += 1
                    continue

            rpt = self._risk_per_trade
            if self._compounding and self._current_capital > 0:
                rpt = self._current_capital * (self._risk_per_trade / self._initial_capital)

            # R17: Kelly fraction sizing
            if self._kelly_fraction > 0 and len(self.trades) >= 20:
                wins_k = [t.pnl for t in self.trades[-100:] if t.pnl > 0]
                losses_k = [abs(t.pnl) for t in self.trades[-100:] if t.pnl <= 0]
                if wins_k and losses_k:
                    win_prob = len(wins_k) / (len(wins_k) + len(losses_k))
                    avg_win_k = sum(wins_k) / len(wins_k)
                    avg_loss_k = sum(losses_k) / len(losses_k)
                    if avg_loss_k > 0:
                        b = avg_win_k / avg_loss_k
                        kelly = win_prob - (1 - win_prob) / b if b > 0 else 0
                        kelly = max(0, kelly * self._kelly_fraction)
                        rpt = self._current_capital * kelly

            if sl > 0:
                lots = round(rpt / (sl * config.POINT_VALUE_PER_LOT), 2)
            else:
                lots = self._min_lot
            lots = max(self._min_lot, min(self._max_lot, lots))

            # R17: Drawdown reduction — reduce lots during moderate drawdowns
            if (self._drawdown_protection
                    and self._current_capital < self._equity_peak * (1 - self._drawdown_reduce_pct)):
                lots = round(lots * self._drawdown_reduce_factor, 2)
                lots = max(self._min_lot, lots)
                self.dd_reduce_count += 1

            # R17: Anti-martingale — adjust lots based on consecutive win/loss streaks
            if self._anti_martingale:
                streak = min(self._consecutive_wins, self._anti_martingale_max_streak)
                if streak > 0:
                    lots = round(lots * (self._anti_martingale_win_mult ** streak), 2)
                streak_l = min(self._consecutive_losses, self._anti_martingale_max_streak)
                if streak_l > 0:
                    lots = round(lots * (self._anti_martingale_loss_mult ** streak_l), 2)
                lots = max(self._min_lot, min(self._max_lot, lots))

            if self._atr_regime_lots:
                if h1_idx is not None and h1_idx >= 0:
                    atr_pct = self._get_atr_percentile_at(h1_idx)
                    if atr_pct > 0.70:
                        lots = round(lots * 1.2, 2)
                    elif atr_pct < 0.30:
                        lots = round(lots * 0.7, 2)
                lots = max(self._min_lot, min(self._max_lot, lots))

            # R157: Session-based lot override
            if self._session_lot_map is not None:
                utc_hour = pd.Timestamp(bar_time).hour
                if utc_hour in self._session_lot_map:
                    lots = self._session_lot_map[utc_hour]

            entry_atr = bar_h1_atr

            pos = Position(
                strategy=strategy, direction=direction,
                entry_price=entry_price, entry_time=bar_time,
                lots=lots, sl_distance=sl, tp_distance=tp,
                entry_atr=entry_atr,
            )
            self.positions.append(pos)
            active_strategies.add(strategy)
            if current_dir is None:
                current_dir = direction
            entered = True

            if source == 'H1':
                self.h1_entry_count += 1
            elif source == 'M30':
                self.m30_entries += 1
            else:
                self.m15_entry_count += 1

        if entered and self._min_entry_gap_hours > 0:
            self._last_entry_time = pd.Timestamp(bar_time)

    # ── Entry slippage ─────────────────────────────────────────

    def _calc_entry_slippage(self, direction: str) -> float:
        """Return entry slippage in price units (positive = worse for trader).

        Models calibrated from 91 real EA trades (2026-04 to 2026-05):
          BUY  mean +$0.67, SELL mean +$0.17 (direction-asymmetric).
        """
        if self._slippage_model == "none":
            return 0.0

        if self._slippage_model == "fixed":
            return self._slippage_buy if direction == 'BUY' else self._slippage_sell

        if self._slippage_model == "empirical":
            pool = self._empirical_buy_slips if direction == 'BUY' else self._empirical_sell_slips
            return float(self._slippage_rng.choice(pool))

        if self._slippage_model == "realistic":
            # Lognormal-ish: sample from empirical with 50% chance, else use mean
            if self._slippage_rng.rand() < 0.5:
                pool = self._empirical_buy_slips if direction == 'BUY' else self._empirical_sell_slips
                return float(self._slippage_rng.choice(pool))
            return self._slippage_buy if direction == 'BUY' else self._slippage_sell

        return 0.0

    # ── Close position ────────────────────────────────────────

    def _calc_dynamic_spread(self, bar_time, h1_atr: float = 0,
                             *, h1_idx: Optional[int] = None) -> float:
        """Calculate spread cost based on the active model."""
        if self._spread_model == "fixed":
            return self._spread_cost

        if self._spread_model == "atr_scaled":
            if h1_atr <= 0:
                return self._spread_base
            if h1_idx is None:
                h1_idx = self._resolve_h1_idx(pd.Timestamp(bar_time))
            atr_pct = self._get_atr_percentile_at(h1_idx) if h1_idx is not None else 0.5
            scaled = self._spread_base * (1 + atr_pct)
            return min(scaled, self._spread_max)

        if self._spread_model == "session_aware":
            hour = pd.Timestamp(bar_time).hour
            if 0 <= hour < 8:       # Asia session
                mult = 1.5
            elif 8 <= hour < 14:    # London session
                mult = 1.0
            elif 14 <= hour < 21:   # NY session (tightest)
                mult = 0.8
            else:                   # Late/close
                mult = 2.0
            return min(self._spread_base * mult, self._spread_max)

        if self._spread_model == "historical" and self._spread_series is not None:
            ts_ms = int(pd.Timestamp(bar_time).timestamp() * 1000)
            idx = self._spread_series.index.searchsorted(ts_ms, side='right') - 1
            if 0 <= idx < len(self._spread_series):
                return min(float(self._spread_series.iloc[idx]), self._spread_max)
            return self._spread_base

        if self._spread_model == "realistic":
            # Session-aware spread calibrated from real broker data + exit slippage
            hour = pd.Timestamp(bar_time).hour
            if 0 <= hour < 8:
                base = 0.40    # Asia: wider spread
            elif 8 <= hour < 14:
                base = 0.25    # London: tightest
            elif 14 <= hour < 21:
                base = 0.25    # NY: tight
            else:
                base = 0.45    # Late: widest
            exit_slip = 0.10   # conservative exit slippage estimate
            return min(base + exit_slip, self._spread_max)

        return self._spread_cost

    def _close_position(self, pos: Position, exit_price: float, exit_time, reason: str,
                        *, h1_idx: Optional[int] = None):
        if pos.direction == 'BUY':
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price
        pnl = round(pnl_points * pos.lots * config.POINT_VALUE_PER_LOT, 2)

        if h1_idx is None:
            h1_idx = self._resolve_h1_idx(pd.Timestamp(exit_time))
        h1_atr = self._get_h1_atr_at(h1_idx)

        spread = self._calc_dynamic_spread(exit_time, h1_atr, h1_idx=h1_idx)
        if spread > 0:
            pnl -= round(spread * pos.lots * config.POINT_VALUE_PER_LOT, 2)

        trade = TradeRecord(
            strategy=pos.strategy, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=exit_time,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=pos.bars_held,
        )
        self.trades.append(trade)
        self.positions.remove(pos)

        if self._compounding:
            self._realized_pnl += pnl
            self._current_capital = self._initial_capital + self._realized_pnl
        elif self._profit_reinvest_pct > 0 and pnl > 0:
            self._realized_pnl += pnl
            reinvest = pnl * self._profit_reinvest_pct
            self._current_capital = self._initial_capital + reinvest
        else:
            self._realized_pnl += pnl
            self._current_capital = self._initial_capital + self._realized_pnl

        # R17: Track equity peak and streak
        if self._current_capital > self._equity_peak:
            self._equity_peak = self._current_capital
        self._trade_equity_history.append(self._current_capital)

        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        elif pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        if pnl < 0:
            self.daily_loss_count += 1
            hours = self._cooldown_hours_override or (config.COOLDOWN_MINUTES / 60)
            if self._escalating_cooldown and self.daily_loss_count >= 2:
                hours *= self._escalating_cooldown_mult
                self.escalated_cooldowns += 1
            self.cooldown_until[pos.strategy] = (
                pd.Timestamp(exit_time) + pd.Timedelta(hours=hours)
            )

    # ── Intraday adaptive ─────────────────────────────────────

    def _precompute_h1_dates(self):
        self._h1_date_map = {}
        if self.h1_df is not None:
            dates = self.h1_df.index.date
            for i, d in enumerate(dates):
                if d not in self._h1_date_map:
                    self._h1_date_map[d] = []
                self._h1_date_map[d].append(i)

    def _update_intraday_score(self, h1_window, bar_time):
        # Fast path: P1 pre-computed score/regime arrays.  Equivalent output
        # to the legacy iloc-heavy logic but ~12s faster on 6-month window.
        if self._h1_score_arr is not None:
            h1_time = pd.Timestamp(bar_time).floor('h')
            max_idx = self.h1_lookup.get(h1_time, -1)
            if max_idx < 0:
                # fall back to <=  scan (rare, e.g. timestamp gaps)
                h1_idx = self._resolve_h1_idx(pd.Timestamp(bar_time))
                if h1_idx is None:
                    return
                max_idx = h1_idx
            if max_idx >= len(self._h1_score_arr):
                max_idx = len(self._h1_score_arr) - 1
            self._current_score = float(self._h1_score_arr[max_idx])
            self._current_regime = self._h1_regime_arr[max_idx]
            self._cached_date = pd.Timestamp(bar_time).date()
            self._cached_h1_count = max_idx
            return

        # Legacy path retained for engines without intraday_adaptive
        if h1_window is None or len(h1_window) < 2:
            return
        bar_date = pd.Timestamp(bar_time).date()

        h1_time = pd.Timestamp(bar_time).floor('h')
        if h1_time in self.h1_lookup:
            max_idx = self.h1_lookup[h1_time]
        else:
            h1_times = self.h1_df.index
            mask = h1_times <= pd.Timestamp(bar_time)
            if not mask.any():
                return
            max_idx = int(mask.sum()) - 1

        if bar_date == self._cached_date and max_idx == self._cached_h1_count:
            return

        indices = self._h1_date_map.get(bar_date)
        if indices:
            valid = [i for i in indices if i <= max_idx]
            if len(valid) >= 2:
                today_bars = self.h1_df.iloc[valid]
                self._current_score = self._calc_realtime_score(today_bars)
                if self._current_score >= self._kc_only_threshold:
                    self._current_regime = 'trending'
                elif self._current_score >= self._choppy_threshold:
                    self._current_regime = 'neutral'
                else:
                    self._current_regime = 'choppy'
            else:
                self._current_score = 0.5
                self._current_regime = 'neutral'
        elif bar_date != self._cached_date:
            self._current_score = 0.5
            self._current_regime = 'neutral'

        self._cached_date = bar_date
        self._cached_h1_count = max_idx

    @staticmethod
    def _calc_realtime_score(today_bars: pd.DataFrame) -> float:
        if len(today_bars) < 2:
            return 0.5
        latest = today_bars.iloc[-1]

        adx = float(latest.get('ADX', 20))
        if np.isnan(adx):
            adx = 20
        adx_score = min(adx / 40.0, 1.0)

        kc_upper = today_bars.get('KC_upper')
        kc_lower = today_bars.get('KC_lower')
        if kc_upper is not None and kc_lower is not None:
            breaks = ((today_bars['Close'] > kc_upper) | (today_bars['Close'] < kc_lower)).sum()
            kc_score = min(float(breaks) / len(today_bars), 1.0)
        else:
            kc_score = 0.0

        ema9 = today_bars.get('EMA9')
        ema21 = today_bars.get('EMA21')
        ema100 = today_bars.get('EMA100')
        if ema9 is not None and ema21 is not None and ema100 is not None:
            bullish = (ema9 > ema21) & (ema21 > ema100)
            bearish = (ema9 < ema21) & (ema21 < ema100)
            aligned = (bullish | bearish).sum()
            ema_score = float(aligned) / len(today_bars)
        else:
            ema_score = 0.0

        day_open = float(today_bars.iloc[0]['Open'])
        day_close = float(latest['Close'])
        day_high = float(today_bars['High'].max())
        day_low = float(today_bars['Low'].min())
        day_range = day_high - day_low
        ti = abs(day_close - day_open) / day_range if day_range > 0.01 else 0.0

        return round(0.30 * adx_score + 0.25 * kc_score + 0.25 * ema_score + 0.20 * ti, 3)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_h1_lookup(h1_df: pd.DataFrame) -> Dict[pd.Timestamp, int]:
        return {ts: i for i, ts in enumerate(h1_df.index)}

    @staticmethod
    def _build_m30_lookup(m30_df: pd.DataFrame) -> Dict[pd.Timestamp, int]:
        return {ts: i for i, ts in enumerate(m30_df.index)}

    def _resolve_h1_idx(self, m15_time: pd.Timestamp) -> Optional[int]:
        """Resolve the H1 bar index aligned to m15_time."""
        h1_time = m15_time.floor('h')
        if h1_time in self.h1_lookup:
            h1_idx = self.h1_lookup[h1_time]
        else:
            h1_times = self.h1_df.index
            mask = h1_times <= m15_time
            if not mask.any():
                return None
            h1_idx = mask.sum() - 1
        h1_len = len(self.h1_df)
        if h1_idx >= h1_len:
            h1_idx = h1_len - 1
        return h1_idx

    def _h1_window_from_idx(self, h1_idx: Optional[int]) -> Optional[pd.DataFrame]:
        """Slice H1 window from a pre-resolved index."""
        if h1_idx is None or h1_idx < 0:
            return None
        start = max(0, h1_idx - self.H1_WINDOW + 1)
        return self.h1_df.iloc[start:h1_idx + 1]

    def _get_h1_window_with_idx(self, m15_time: pd.Timestamp):
        """Return (h1_idx, h1_window) — avoids duplicate lookups in the hot loop."""
        h1_idx = self._resolve_h1_idx(m15_time)
        return h1_idx, self._h1_window_from_idx(h1_idx)

    def _get_h1_window(self, m15_time: pd.Timestamp, closed_only: bool = False) -> Optional[pd.DataFrame]:
        """Get H1 data window aligned to m15_time.

        Args:
            closed_only: If True, exclude the current (potentially unclosed) H1 bar.
                         H1 timestamps represent bar OPEN time (Dukascopy convention),
                         so H1[14:00] covers 14:00-15:00 and is not closed until 15:00.
                         At M15[14:00], H1[14:00] is still open — using its Close is
                         look-ahead bias. With closed_only=True, the window ends at
                         H1[13:00] instead (the last fully closed bar).
        """
        h1_idx = self._resolve_h1_idx(m15_time)
        if h1_idx is None:
            return None
        if closed_only:
            h1_idx -= 1
        return self._h1_window_from_idx(h1_idx)

    def _get_atr_percentile(self, h1_window: Optional[pd.DataFrame]) -> float:
        """Get ATR percentile using either precomputed column or live-style rolling-50."""
        if h1_window is None or len(h1_window) == 0:
            return 0.5
        if self._live_atr_pct:
            atr_series = h1_window['ATR'].dropna()
            if len(atr_series) >= 50:
                current_atr = float(atr_series.iloc[-1])
                return float((atr_series.iloc[-50:] < current_atr).mean())
            return 0.5
        val = h1_window.iloc[-1].get('atr_percentile', 0.5)
        return 0.5 if pd.isna(val) else float(val)

    @staticmethod
    def _get_h1_atr(h1_window: Optional[pd.DataFrame]) -> float:
        if h1_window is None or len(h1_window) == 0:
            return 0
        atr = float(h1_window.iloc[-1].get('ATR', 0))
        return atr if not np.isnan(atr) else 0

    def _calc_unrealized(self, current_price: float) -> float:
        total = 0.0
        for pos in self.positions:
            if pos.direction == 'BUY':
                pnl = (current_price - pos.entry_price) * pos.lots * config.POINT_VALUE_PER_LOT
            else:
                pnl = (pos.entry_price - current_price) * pos.lots * config.POINT_VALUE_PER_LOT
            total += pnl
        return total

    @staticmethod
    def _reset_global_state():
        orb = get_orb_strategy()
        orb.reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
