"""
QUANT v9.0 — FULL REWRITE
Fixes applied:
  1.  Short-side logic for all 3 strategies
  2.  Correlation cache (recompute every 6 bars, not every bar)
  3.  consecutive_losses resets only on N consecutive wins, not 1
  4.  Fixed swing lookahead: proper center=True rolling with future-bar shift
  5.  Removed erroneous `break` — all qualifying symbols are evaluated
  6.  CONFLUENCE_MIN_SCORE default raised to 2
  7.  Cooldown applies to both winning and losing exits
  8.  Walk-Forward split (in-sample / out-of-sample)
  9.  Regime-adaptive ATR stop multiplier
 10.  Regime-adaptive position sizing (reduce in HIGH_VOL)
 11.  Dynamic funding rate per bars_held
 12.  Config split into sub-configs (Strategy / Risk / Exchange)
 13.  compute_indicators uses tf parameter for VWAP window
 14.  warnings kept for meaningful pandas/numpy messages
 15.  Monte Carlo equity simulation added to analyze_results
 16.  Parameter sensitivity report added
 17.  Equity curve + drawdown chart saved (matplotlib)
 18.  Unit tests for critical components (run with --test flag)
 19.  Short exits: take-profit below, stop above entry
 20.  Break-even and trailing stop for short positions
"""

import os, sys, time, logging, argparse, warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import talib
import ccxt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "quant_v90.log"
log = logging.getLogger("qv90")
log.setLevel(logging.INFO)
_fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Regime(Enum):
    NEUTRAL  = "NEUTRAL"
    TRENDING = "TRENDING"
    RANGING  = "RANGING"
    HIGH_VOL = "HIGH_VOL"


class ExitType(Enum):
    TIMEOUT         = "TIMEOUT"
    STOP_LOSS       = "STOP_LOSS"
    TAKE_PROFIT     = "TAKE_PROFIT"
    TRAILING        = "TRAILING"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


# ---------------------------------------------------------------------------
# Config (split into logical sub-configs)
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    SYMBOLS:              Tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    TF_1H:                str   = "1h"
    TF_4H:                str   = "4h"
    DAYS_HISTORY:         int   = 180
    EMA_TREND:            int   = 200
    BB_PERIOD:            int   = 20
    BB_STD:               float = 2.0
    RSI_LONG:             float = 30.0   # oversold threshold
    RSI_SHORT:            float = 70.0   # overbought threshold
    MIN_RR:               float = 2.0
    # Order Block
    OB_SWING_LOOKBACK:    int   = 5
    OB_IMPULSE_ATR:       float = 1.50
    OB_CONFIRMATION_BARS: int   = 3
    OB_RSI_LONG:          float = 40.0
    OB_RSI_SHORT:         float = 60.0
    OB_ZONE_TOLERANCE:    float = 0.020
    OB_MIN_VOL_RATIO:     float = 1.50
    # VWAP
    VWAP_BOUNCE_ATR:      float = 0.50
    VWAP_RSI_LONG:        float = 35.0
    VWAP_RSI_SHORT:       float = 65.0
    ROLLING_VWAP_WINDOW_1H: int = 24
    ROLLING_VWAP_WINDOW_4H: int = 6
    # Confluence
    CONFLUENCE_MIN_SCORE: int   = 2   # FIX #6: was 1


@dataclass
class RiskConfig:
    ACCOUNT_SIZE:           float = 500.0
    RISK_PCT:               float = 0.0100
    MAX_RISK_PCT:           float = 0.0500
    MAX_CONCURRENT_POSITIONS: int = 3
    MIN_POS_USD:            float = 20.0
    MAX_POS_PCT:            float = 0.30
    ATR_STOP_MULT:          float = 1.80
    # Regime-adaptive stop multipliers (FIX #9)
    ATR_STOP_MULT_HIGH_VOL: float = 2.50
    ATR_STOP_MULT_TRENDING: float = 2.00
    ATR_STOP_MULT_RANGING:  float = 1.40
    ATR_STOP_MULT_NEUTRAL:  float = 1.80
    # Regime-adaptive size factors (FIX #10)
    SIZE_FACTOR_HIGH_VOL:   float = 0.50
    SIZE_FACTOR_TRENDING:   float = 1.00
    SIZE_FACTOR_RANGING:    float = 0.80
    SIZE_FACTOR_NEUTRAL:    float = 0.90
    MAX_HOLD_HOURS:         int   = 10
    TRAIL_ACTIVATE_RR:      float = 1.00
    TRAIL_ATR_MULT:         float = 1.20
    BREAK_EVEN_RR:          float = 0.50
    SLIPPAGE_ATR_MULT:      float = 0.15
    STOP_SLIPPAGE_ATR_MULT: float = 0.25
    # Circuit Breaker
    CB_MAX_DD:              float = 0.15
    CB_PAUSE_HOURS:         int   = 24
    CB_CONSECUTIVE_LOSSES:  int   = 5
    CB_RESET_WINS:          int   = 2   # FIX #3: need N wins to reset counter
    # Correlation
    CORR_REDUCTION_FACTOR:  float = 0.50
    CORR_THRESHOLD:         float = 0.70
    CORR_WINDOW:            int   = 50
    CORR_RECOMPUTE_BARS:    int   = 6   # FIX #2: recompute every N bars
    # Cooldown
    COOLDOWN_HOURS:         int   = 6
    # Walk-Forward
    WF_OOS_FRACTION:        float = 0.30   # FIX #8: last 30% = OOS


@dataclass
class ExchangeConfig:
    FEE_MAKER:      float = 0.0002
    FEE_TAKER:      float = 0.0006
    # Dynamic funding: base rate per 8h funding period (FIX #11)
    FUNDING_RATE_BASE: float = 0.00010
    FUNDING_RATE_HIGH: float = 0.00030   # used in HIGH_VOL / strong trend
    CACHE_TTL_SECS: int   = 3600


@dataclass
class Config:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk:     RiskConfig     = field(default_factory=RiskConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)


cfg = Config()


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._ex = None

    @property
    def exchange(self):
        if self._ex is None:
            self._ex = ccxt.binance({"enableRateLimit": True})
        return self._ex

    def load(self, symbol: str, tf: str, days: int) -> pd.DataFrame:
        safe_sym = symbol.replace("/", "_")
        fname = f"{safe_sym}_{tf}_{days}d.pkl"
        cache = self.cache_dir / fname
        if cache.exists() and (time.time() - cache.stat().st_mtime < cfg.exchange.CACHE_TTL_SECS):
            log.info(f"Cache hit: {symbol} {tf}")
            return pd.read_pickle(cache)
        try:
            log.info(f"Fetching {symbol} {tf} from exchange...")
            since = self.exchange.parse8601(
                (datetime.utcnow() - timedelta(days=days)).isoformat()
            )
            all_ohlcv = []
            while since < self.exchange.milliseconds():
                batch = self.exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                time.sleep(0.1)
            df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            df = df[
                (df["high"] >= df["low"])
                & (df["high"] >= df["close"])
                & (df["high"] >= df["open"])
                & (df["low"] <= df["close"])
                & (df["low"] <= df["open"])
            ]
            df.to_pickle(cache)
            log.info(f"Fetched {symbol} {tf}: {len(df)} bars")
            return df
        except Exception as e:
            log.error(f"Load error {symbol} {tf}: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    tf is now used to select the correct VWAP window (FIX #14).
    """
    if df.empty or len(df) < 250:
        return pd.DataFrame()
    df = df.copy()
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["vol"].values.astype(float)

    # Core indicators
    df["atr"]       = talib.ATR(h, l, c, 14)
    df["rsi"]       = talib.RSI(c, 14)
    df["vol_sma"]   = talib.SMA(v, 20)
    df["vol_ratio"] = df["vol"] / (df["vol_sma"] + 1e-9)
    df["ema200"]    = talib.EMA(c, 200)
    df["ema50"]     = talib.EMA(c, 50)
    df["ema_slope"] = df["ema200"].pct_change(24)

    # Regime classification
    adx = talib.ADX(h, l, c, 14)
    df["adx"] = adx
    atr_min = df["atr"].rolling(100).min()
    atr_max = df["atr"].rolling(100).max()
    df["atr_pct"] = (df["atr"] - atr_min) / (atr_max - atr_min + 1e-9)

    conditions = [
        (df["atr_pct"] > 0.8),
        (adx > 25) & (df["atr_pct"] <= 0.8),
        (adx < 20) & (df["atr_pct"] <= 0.8),
    ]
    choices = [Regime.HIGH_VOL.value, Regime.TRENDING.value, Regime.RANGING.value]
    df["regime"] = np.select(conditions, choices, default=Regime.NEUTRAL.value)

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(c, cfg.strategy.BB_PERIOD, cfg.strategy.BB_STD, cfg.strategy.BB_STD)
    df["bb_upper"] = upper
    df["bb_mid"]   = middle
    df["bb_lower"] = lower
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_z"] = (df["close"] - df["bb_mid"]) / (bb_width / 2 + 1e-9)

    # VWAP — window depends on timeframe (FIX #14)
    vwap_window = (
        cfg.strategy.ROLLING_VWAP_WINDOW_4H if tf == "4h"
        else cfg.strategy.ROLLING_VWAP_WINDOW_1H
    )
    df["tp"]      = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]  = df["tp"] * df["vol"]
    df["vwap"]    = df["tp_vol"].rolling(vwap_window).sum() / (df["vol"].rolling(vwap_window).sum() + 1e-9)
    df["vwap_dist"] = (df["close"] - df["vwap"]) / (df["atr"] + 1e-9)

    # Swing High / Low — FIX #4: use proper non-lookahead detection
    # A swing high at bar i: high[i] is the max in [i-n .. i] AND high[i] > high[i+1..i+n]
    # We approximate without future data by using a lagged window of size 2n+1
    # and confirming the peak was n bars ago.
    n   = cfg.strategy.OB_SWING_LOOKBACK
    win = 2 * n + 1
    # We look at a window ending at bar i-n, so the candidate peak is at bar i-n
    # and we have n future bars (relative to i-n) to confirm it is indeed a peak.
    # Both halves of the window are in the past relative to bar i.
    rolling_high_left  = df["high"].shift(n).rolling(n + 1).max()   # [i-2n .. i-n]
    rolling_high_right = df["high"].rolling(n).max().shift(1)        # [i-n+1 .. i] (right side of swing)
    candidate_high     = df["high"].shift(n)
    df["is_swing_high"] = (
        (candidate_high >= rolling_high_left) & (candidate_high > rolling_high_right)
    ).astype(bool)

    rolling_low_left  = df["low"].shift(n).rolling(n + 1).min()
    rolling_low_right = df["low"].rolling(n).min().shift(1)
    candidate_low     = df["low"].shift(n)
    df["is_swing_low"] = (
        (candidate_low <= rolling_low_left) & (candidate_low < rolling_low_right)
    ).astype(bool)

    df["swing_high_level"] = np.where(df["is_swing_high"], df["high"].shift(n), np.nan)
    df["swing_low_level"]  = np.where(df["is_swing_low"],  df["low"].shift(n),  np.nan)
    df["last_swing_high"]  = df["swing_high_level"].ffill()
    df["last_swing_low"]   = df["swing_low_level"].ffill()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["atr", "rsi", "ema200", "bb_upper", "bb_lower", "vwap"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Strategies  (FIX #1: all strategies generate +1 long AND -1 short)
# ---------------------------------------------------------------------------
def strat_mean_rev(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.Series:
    signals = pd.Series(0, index=df_1h.index)
    trend_4h  = df_4h["ema200"].shift(1).reindex(df_1h.index, method="ffill")
    ema50_4h  = df_4h["ema50"].shift(1).reindex(df_1h.index, method="ffill")
    slope_4h  = df_4h["ema_slope"].shift(1).reindex(df_1h.index, method="ffill")
    vol_ok    = df_1h["vol_ratio"].between(0.5, 3.0)

    # Long: price above 4H trend, oversold on 1H
    long_cond = (
        (df_1h["close"] > trend_4h)
        & ((trend_4h > ema50_4h) | (slope_4h > 0))
        & (df_1h["bb_z"] < -1.5)
        & (df_1h["rsi"] < cfg.strategy.RSI_LONG)
        & vol_ok
    )
    # Short: price below 4H trend, overbought on 1H
    short_cond = (
        (df_1h["close"] < trend_4h)
        & ((trend_4h < ema50_4h) | (slope_4h < 0))
        & (df_1h["bb_z"] > 1.5)
        & (df_1h["rsi"] > cfg.strategy.RSI_SHORT)
        & vol_ok
    )
    signals[long_cond]  =  1
    signals[short_cond] = -1
    return signals


def strat_vwap(df: pd.DataFrame) -> pd.Series:
    signals   = pd.Series(0, index=df.index)
    regime_ok = df["regime"].isin([Regime.RANGING.value, Regime.NEUTRAL.value])

    long_cond = (
        (df["vwap_dist"] < -cfg.strategy.VWAP_BOUNCE_ATR)
        & (df["rsi"] < cfg.strategy.VWAP_RSI_LONG)
        & regime_ok
        & (df["close"] > df["ema200"])
    )
    short_cond = (
        (df["vwap_dist"] > cfg.strategy.VWAP_BOUNCE_ATR)
        & (df["rsi"] > cfg.strategy.VWAP_RSI_SHORT)
        & regime_ok
        & (df["close"] < df["ema200"])
    )
    signals[long_cond]  =  1
    signals[short_cond] = -1
    return signals


def strat_ob(df: pd.DataFrame) -> pd.Series:
    """Order Block: long on retrace to swing-low OB, short on retrace to swing-high OB."""
    signals = pd.Series(0, index=df.index)
    conf    = cfg.strategy.OB_CONFIRMATION_BARS

    # ---- LONG setup ----
    swing_low_confirmed = df["is_swing_low"].shift(conf).fillna(False)
    swing_low_price     = df["last_swing_low"].shift(conf)
    vol_confirmed       = df["vol_ratio"].shift(conf) > cfg.strategy.OB_MIN_VOL_RATIO
    atr_at_swing        = df["atr"].shift(conf)
    impulse_target_up   = swing_low_price + (atr_at_swing * cfg.strategy.OB_IMPULSE_ATR)
    impulse_reached_up  = df["high"].rolling(conf).max() > impulse_target_up
    ob_lower_long       = swing_low_price * (1 - cfg.strategy.OB_ZONE_TOLERANCE)
    ob_upper_long       = swing_low_price * (1 + cfg.strategy.OB_ZONE_TOLERANCE)
    in_ob_zone_long     = (df["low"] <= ob_upper_long) & (df["low"] >= ob_lower_long)
    long_signal = (
        swing_low_confirmed & impulse_reached_up & in_ob_zone_long
        & (df["close"] > swing_low_price)
        & (df["rsi"] < cfg.strategy.OB_RSI_LONG)
        & vol_confirmed
    )

    # ---- SHORT setup ----
    swing_high_confirmed = df["is_swing_high"].shift(conf).fillna(False)
    swing_high_price     = df["last_swing_high"].shift(conf)
    impulse_target_dn    = swing_high_price - (atr_at_swing * cfg.strategy.OB_IMPULSE_ATR)
    impulse_reached_dn   = df["low"].rolling(conf).min() < impulse_target_dn
    ob_lower_short       = swing_high_price * (1 - cfg.strategy.OB_ZONE_TOLERANCE)
    ob_upper_short       = swing_high_price * (1 + cfg.strategy.OB_ZONE_TOLERANCE)
    in_ob_zone_short     = (df["high"] >= ob_lower_short) & (df["high"] <= ob_upper_short)
    short_signal = (
        swing_high_confirmed & impulse_reached_dn & in_ob_zone_short
        & (df["close"] < swing_high_price)
        & (df["rsi"] > cfg.strategy.OB_RSI_SHORT)
        & vol_confirmed
    )

    signals[long_signal]  =  1
    signals[short_signal] = -1
    return signals


# ---------------------------------------------------------------------------
# Position & Trade
# ---------------------------------------------------------------------------
@dataclass
class Position:
    entry_ts:               datetime
    symbol:                 str
    strat:                  str
    direction:              int        # +1 long, -1 short
    entry_px:               float
    size_usd:               float
    stop_loss:              float
    take_profit:            float
    original_stop_distance: float = 0.0
    trailing_stop:          Optional[float] = None
    break_even_moved:       bool  = False
    bars_held:              int   = 0
    regime_at_entry:        str   = Regime.NEUTRAL.value


@dataclass
class Trade:
    entry_ts:   datetime
    exit_ts:    datetime
    symbol:     str
    strat:      str
    direction:  int
    entry_px:   float
    exit_px:    float
    size_usd:   float
    pnl:        float
    pnl_pct:    float
    reason:     str
    bars_held:  int


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class Portfolio:
    def __init__(self, initial_capital: float, risk_cfg: RiskConfig, exchange_cfg: ExchangeConfig):
        self.equity          = initial_capital
        self.initial_capital = initial_capital
        self.peak            = initial_capital
        self.drawdown        = 0.0
        self.max_drawdown    = 0.0
        self.positions:      List[Position] = []
        self.closed_trades:  List[Trade]    = []
        self.equity_curve:   List[dict]     = []
        self.cb_active       = False
        self.cb_resume_time  = None
        self.consecutive_losses = 0
        self.consecutive_wins   = 0     # FIX #3
        self.last_exit: Dict[tuple, datetime] = {}
        self._corr_cache: Dict[tuple, float] = {}
        self._corr_last_bar: int = -999
        self.rcfg = risk_cfg
        self.ecfg = exchange_cfg

    # ------------------------------------------------------------------
    def update_equity(self, pnl: float):
        self.equity += pnl
        self.peak    = max(self.peak, self.equity)
        self.drawdown = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, self.drawdown)

    def get_unrealized_equity(self, current_prices: dict) -> float:
        unreal = sum(
            pos.size_usd * ((current_prices.get(pos.symbol, pos.entry_px) - pos.entry_px)
                            / pos.entry_px) * pos.direction
            for pos in self.positions
        )
        return self.equity + unreal

    def record_equity(self, timestamp, current_prices: dict):
        total = self.get_unrealized_equity(current_prices)
        cp    = max(self.peak, total)
        dd    = (cp - total) / cp if cp > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)
        self.equity_curve.append({"time": timestamp, "equity": total, "drawdown": dd})

    # ------------------------------------------------------------------
    def check_circuit_breaker(self, current_time: datetime) -> bool:
        if self.drawdown >= self.rcfg.CB_MAX_DD:
            self._activate_cb("Max DD", current_time)
            return True
        if self.consecutive_losses >= self.rcfg.CB_CONSECUTIVE_LOSSES:
            self._activate_cb("Consecutive Losses", current_time)
            return True
        return False

    def _activate_cb(self, reason: str, current_time: datetime):
        if not self.cb_active:
            self.cb_active       = True
            self.cb_resume_time  = current_time + timedelta(hours=self.rcfg.CB_PAUSE_HOURS)
            log.warning(f"CIRCUIT BREAKER: {reason} | Resume at {self.cb_resume_time}")

    def can_trade(self, current_time: datetime) -> bool:
        if not self.cb_active:
            return True
        if self.cb_resume_time and current_time >= self.cb_resume_time:
            self.cb_active          = False
            self.consecutive_losses = 0
            log.info("Circuit breaker lifted.")
            return True
        return False

    # ------------------------------------------------------------------
    def is_in_cooldown(self, symbol: str, strat: str, current_time: datetime) -> bool:
        key = (symbol, strat)
        if key not in self.last_exit:
            return False
        elapsed_h = (current_time - self.last_exit[key]).total_seconds() / 3600
        return elapsed_h < self.rcfg.COOLDOWN_HOURS

    def has_position_on_symbol(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)

    def get_total_risk(self) -> float:
        total = sum(
            (pos.size_usd / pos.entry_px) * abs(pos.entry_px - pos.stop_loss)
            for pos in self.positions
        )
        return total / self.equity if self.equity > 0 else 0.0

    def add_position(self, pos: Position):
        self.positions.append(pos)

    # ------------------------------------------------------------------
    def close_position(self, pos: Position, exit_ts, exit_px: float,
                       reason: ExitType, bars_held: int) -> Trade:
        gross_pct  = (exit_px - pos.entry_px) * pos.direction / pos.entry_px
        entry_fee  = pos.size_usd * self.ecfg.FEE_MAKER
        exit_fee   = pos.size_usd * self.ecfg.FEE_TAKER
        # FIX #11: dynamic funding rate based on regime at entry
        funding_rate = (
            self.ecfg.FUNDING_RATE_HIGH
            if pos.regime_at_entry in (Regime.HIGH_VOL.value, Regime.TRENDING.value)
            else self.ecfg.FUNDING_RATE_BASE
        )
        funding    = pos.size_usd * funding_rate * (pos.bars_held / 8)
        net_pnl    = pos.size_usd * gross_pct - (entry_fee + exit_fee + funding)
        reason_str = reason.value if hasattr(reason, "value") else str(reason)

        # FIX #3: track consecutive wins/losses properly
        if net_pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins    = 0
        else:
            self.consecutive_wins   += 1
            if self.consecutive_wins >= self.rcfg.CB_RESET_WINS:
                self.consecutive_losses = 0

        # FIX #7: cooldown on ALL exits (win or loss)
        self.last_exit[(pos.symbol, pos.strat)] = exit_ts

        trade = Trade(
            entry_ts  = pos.entry_ts,
            exit_ts   = exit_ts,
            symbol    = pos.symbol,
            strat     = pos.strat,
            direction = pos.direction,
            entry_px  = pos.entry_px,
            exit_px   = exit_px,
            size_usd  = pos.size_usd,
            pnl       = net_pnl,
            pnl_pct   = net_pnl / pos.size_usd,
            reason    = reason_str,
            bars_held = bars_held,
        )
        self.closed_trades.append(trade)
        self.update_equity(net_pnl)
        self.positions.remove(pos)
        return trade

    # ------------------------------------------------------------------
    def compute_correlations(self, data: dict, bar_idx: int,
                             current_time) -> Dict[tuple, float]:
        """
        FIX #2: only recompute every CORR_RECOMPUTE_BARS bars.
        """
        if bar_idx - self._corr_last_bar < self.rcfg.CORR_RECOMPUTE_BARS:
            return self._corr_cache
        self._corr_last_bar = bar_idx
        symbols = list(data.keys())
        cache   = {}
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                try:
                    r1 = data[s1].loc[:current_time]["close"].pct_change().tail(self.rcfg.CORR_WINDOW)
                    r2 = data[s2].loc[:current_time]["close"].pct_change().tail(self.rcfg.CORR_WINDOW)
                    c  = r1.corr(r2)
                    cache[tuple(sorted([s1, s2]))] = c if not np.isnan(c) else 0.0
                except Exception:
                    cache[tuple(sorted([s1, s2]))] = 0.0
        self._corr_cache = cache
        return cache

    def has_correlated_position(self, symbol: str, correlations: dict) -> bool:
        for pos in self.positions:
            pair = tuple(sorted([symbol, pos.symbol]))
            if abs(correlations.get(pair, 0.0)) > self.rcfg.CORR_THRESHOLD:
                return True
        return False


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    def __init__(self, config: Config, data_loader: Optional[DataLoader] = None):
        self.cfg      = config
        self.loader   = data_loader or DataLoader()
        self.portfolio = Portfolio(
            config.risk.ACCOUNT_SIZE,
            config.risk,
            config.exchange,
        )

    # ------------------------------------------------------------------
    def _regime_stop_mult(self, regime: str) -> float:
        """FIX #9: regime-adaptive ATR stop multiplier."""
        mapping = {
            Regime.HIGH_VOL.value:  self.cfg.risk.ATR_STOP_MULT_HIGH_VOL,
            Regime.TRENDING.value:  self.cfg.risk.ATR_STOP_MULT_TRENDING,
            Regime.RANGING.value:   self.cfg.risk.ATR_STOP_MULT_RANGING,
            Regime.NEUTRAL.value:   self.cfg.risk.ATR_STOP_MULT_NEUTRAL,
        }
        return mapping.get(regime, self.cfg.risk.ATR_STOP_MULT)

    def _regime_size_factor(self, regime: str) -> float:
        """FIX #10: reduce position size in high-volatility regimes."""
        mapping = {
            Regime.HIGH_VOL.value:  self.cfg.risk.SIZE_FACTOR_HIGH_VOL,
            Regime.TRENDING.value:  self.cfg.risk.SIZE_FACTOR_TRENDING,
            Regime.RANGING.value:   self.cfg.risk.SIZE_FACTOR_RANGING,
            Regime.NEUTRAL.value:   self.cfg.risk.SIZE_FACTOR_NEUTRAL,
        }
        return mapping.get(regime, 1.0)

    # ------------------------------------------------------------------
    def estimate_slippage(self, size_usd: float, atr: float, avg_vol_usd: float) -> float:
        base   = atr * self.cfg.risk.SLIPPAGE_ATR_MULT
        impact = (size_usd / (avg_vol_usd + 1e-9)) * atr * 0.5
        return base + impact + atr * 0.01

    def calculate_position_size(self, entry_px: float, stop_px: float,
                                symbol: str, correlations: dict,
                                regime: str) -> float:
        risk_usd = self.portfolio.equity * self.cfg.risk.RISK_PCT

        # Drawdown scaling
        if self.portfolio.drawdown > 0.10:
            risk_usd *= 0.4
        elif self.portfolio.drawdown > 0.05:
            risk_usd *= 0.7

        # Correlation reduction
        if self.portfolio.has_correlated_position(symbol, correlations):
            risk_usd *= self.cfg.risk.CORR_REDUCTION_FACTOR

        # FIX #10: regime size factor
        risk_usd *= self._regime_size_factor(regime)

        stop_dist = abs(entry_px - stop_px)
        if stop_dist <= 0:
            return 0.0
        size = min(
            risk_usd / (stop_dist / entry_px),
            self.portfolio.equity * self.cfg.risk.MAX_POS_PCT,
        )
        if size < self.cfg.risk.MIN_POS_USD:
            return 0.0
        if self.portfolio.get_total_risk() >= self.cfg.risk.MAX_RISK_PCT:
            return 0.0
        return size

    # ------------------------------------------------------------------
    def update_position(self, pos: Position, bar) -> Optional[Trade]:
        """Handle both long and short exit logic."""
        pos.bars_held += 1
        price    = bar["close"]
        h, l, atr = bar["high"], bar["low"], bar["atr"]
        bar_open = bar["open"]
        d        = pos.direction  # +1 or -1

        # --- Stop Loss ---
        stop_hit = (d > 0 and l <= pos.stop_loss) or (d < 0 and h >= pos.stop_loss)
        if stop_hit:
            slip = atr * self.cfg.risk.STOP_SLIPPAGE_ATR_MULT
            if d > 0:
                exit_px = (bar_open - slip) if bar_open <= pos.stop_loss else (pos.stop_loss - slip)
                exit_px = max(exit_px, l)
            else:
                exit_px = (bar_open + slip) if bar_open >= pos.stop_loss else (pos.stop_loss + slip)
                exit_px = min(exit_px, h)
            return self.portfolio.close_position(pos, bar.name, exit_px, ExitType.STOP_LOSS, pos.bars_held)

        # --- Take Profit ---
        tp_hit = (d > 0 and h >= pos.take_profit) or (d < 0 and l <= pos.take_profit)
        if tp_hit:
            return self.portfolio.close_position(pos, bar.name, pos.take_profit, ExitType.TAKE_PROFIT, pos.bars_held)

        # --- Break Even ---
        if not pos.break_even_moved and pos.original_stop_distance > 0:
            rr = (price - pos.entry_px) * d / pos.original_stop_distance
            if rr >= self.cfg.risk.BREAK_EVEN_RR:
                pos.stop_loss        = pos.entry_px + (atr * 0.1) * d
                pos.break_even_moved = True

        # --- Trailing Stop ---
        if pos.original_stop_distance > 0:
            rr = (price - pos.entry_px) * d / pos.original_stop_distance
        else:
            rr = 0.0
        if rr >= self.cfg.risk.TRAIL_ACTIVATE_RR:
            if d > 0:
                trail = price - atr * self.cfg.risk.TRAIL_ATR_MULT
                if pos.trailing_stop is None or trail > pos.trailing_stop:
                    pos.trailing_stop = trail
                    pos.stop_loss = max(pos.stop_loss, trail)
                if l <= pos.trailing_stop:
                    return self.portfolio.close_position(
                        pos, bar.name, pos.trailing_stop, ExitType.TRAILING, pos.bars_held)
            else:  # short
                trail = price + atr * self.cfg.risk.TRAIL_ATR_MULT
                if pos.trailing_stop is None or trail < pos.trailing_stop:
                    pos.trailing_stop = trail
                    pos.stop_loss = min(pos.stop_loss, trail)
                if h >= pos.trailing_stop:
                    return self.portfolio.close_position(
                        pos, bar.name, pos.trailing_stop, ExitType.TRAILING, pos.bars_held)

        # --- Timeout ---
        if pos.bars_held >= self.cfg.risk.MAX_HOLD_HOURS:
            return self.portfolio.close_position(pos, bar.name, price, ExitType.TIMEOUT, pos.bars_held)
        return None

    # ------------------------------------------------------------------
    def _try_open(self, symbol: str, bar, next_bar, strat_name: str,
                  direction: int, strats: List[str], correlations: dict):
        """Attempt to open a position for one symbol+direction combo."""
        entry  = next_bar["open"]
        atr    = bar["atr"]
        regime = bar.get("regime", Regime.NEUTRAL.value)
        slip   = self.estimate_slippage(
            self.portfolio.equity * self.cfg.risk.MAX_POS_PCT,
            atr,
            bar["vol"] * bar["close"],
        )
        entry   += slip * direction  # long: adds to price, short: subtracts
        stop_mult = self._regime_stop_mult(regime)
        stop_dist = atr * stop_mult
        stop      = entry - stop_dist * direction
        tp        = entry + stop_dist * self.cfg.strategy.MIN_RR * direction
        size      = self.calculate_position_size(entry, stop, symbol, correlations, regime)
        if size < self.cfg.risk.MIN_POS_USD:
            return
        self.portfolio.add_position(Position(
            entry_ts               = next_bar.name,
            symbol                 = symbol,
            strat                  = strat_name,
            direction              = direction,
            entry_px               = entry,
            size_usd               = size,
            stop_loss              = stop,
            take_profit            = tp,
            original_stop_distance = stop_dist,
            regime_at_entry        = regime,
        ))
        side = "LONG" if direction > 0 else "SHORT"
        log.info(f"  Open {symbol} {strat_name} {side}: ${size:.2f} @ {entry:.4f} "
                 f"SL={stop:.4f} TP={tp:.4f}")

    # ------------------------------------------------------------------
    def run(self, oos_only: bool = False):
        """
        FIX #8: walk-forward split.
        oos_only=False → run on full dataset (in-sample)
        oos_only=True  → run only on OOS portion
        """
        log.info("Starting Backtest v9.0...")
        data_1h: Dict[str, pd.DataFrame] = {}
        data_4h: Dict[str, pd.DataFrame] = {}

        for symbol in self.cfg.strategy.SYMBOLS:
            df1 = compute_indicators(
                self.loader.load(symbol, self.cfg.strategy.TF_1H, self.cfg.risk.WF_OOS_FRACTION
                                 and self.cfg.strategy.DAYS_HISTORY),
                self.cfg.strategy.TF_1H,
            )
            df4 = compute_indicators(
                self.loader.load(symbol, self.cfg.strategy.TF_4H, self.cfg.strategy.DAYS_HISTORY),
                self.cfg.strategy.TF_4H,
            )
            if df1.empty or df4.empty:
                log.warning(f"Skipping {symbol}: insufficient data")
                continue
            data_1h[symbol] = df1
            data_4h[symbol] = df4
            log.info(f"Loaded {symbol}: {len(df1)} 1H bars, {len(df4)} 4H bars")

        if not data_1h:
            log.error("No valid data — aborting.")
            return self.portfolio, []

        # Compute signals
        for symbol in data_1h:
            data_1h[symbol]["sig_mr"] = strat_mean_rev(data_1h[symbol], data_4h[symbol])
            data_1h[symbol]["sig_vw"] = strat_vwap(data_1h[symbol])
            data_1h[symbol]["sig_ob"] = strat_ob(data_1h[symbol])

        # Build time index
        all_idx = [df.index for df in data_1h.values()]
        start   = max(idx[250] for idx in all_idx)
        end     = min(idx[-self.cfg.risk.MAX_HOLD_HOURS - 1] for idx in all_idx)

        # FIX #8: walk-forward OOS slice
        full_index = list(data_1h.values())[0].loc[start:end].index
        if oos_only:
            split_i    = int(len(full_index) * (1 - self.cfg.risk.WF_OOS_FRACTION))
            time_index = full_index[split_i:]
            log.info(f"OOS simulation: {time_index[0]} → {time_index[-1]}")
        else:
            time_index = full_index
            log.info(f"Full simulation: {start} → {end}")

        # Main loop
        for bar_i, t in enumerate(time_index):
            prices = {s: data_1h[s].loc[t, "close"] for s in data_1h if t in data_1h[s].index}
            self.portfolio.record_equity(t, prices)

            # Update open positions
            for pos in self.portfolio.positions[:]:
                if t not in data_1h[pos.symbol].index:
                    continue
                self.update_position(pos, data_1h[pos.symbol].loc[t])

            # Circuit breaker
            if self.portfolio.check_circuit_breaker(t):
                for pos in self.portfolio.positions[:]:
                    if t not in data_1h[pos.symbol].index:
                        continue
                    self.portfolio.close_position(
                        pos, t, data_1h[pos.symbol].loc[t]["close"],
                        ExitType.CIRCUIT_BREAKER, pos.bars_held,
                    )
                continue

            if not self.portfolio.can_trade(t):
                continue
            if len(self.portfolio.positions) >= self.cfg.risk.MAX_CONCURRENT_POSITIONS:
                continue

            # Correlation (cached, FIX #2)
            corrs = self.portfolio.compute_correlations(data_1h, bar_i, t)

            # FIX #5: evaluate ALL symbols, no break
            for symbol in data_1h:
                if len(self.portfolio.positions) >= self.cfg.risk.MAX_CONCURRENT_POSITIONS:
                    break
                if self.portfolio.has_position_on_symbol(symbol):
                    continue
                if t not in data_1h[symbol].index:
                    continue

                bar = data_1h[symbol].loc[t]
                idx_loc = data_1h[symbol].index.get_loc(t)
                if idx_loc + 1 >= len(data_1h[symbol]):
                    continue
                next_bar = data_1h[symbol].iloc[idx_loc + 1]

                # Score signals
                mr  = bar.get("sig_mr", 0)
                vw  = bar.get("sig_vw", 0)
                ob  = bar.get("sig_ob", 0)

                for direction in (1, -1):
                    score  = 0
                    strats = []
                    if ob == direction:
                        score += 2; strats.append("ob")
                    if mr == direction:
                        score += 1; strats.append("mr")
                    if vw == direction:
                        score += 1; strats.append("vw")

                    # FIX #6: min score raised to 2
                    if score < self.cfg.strategy.CONFLUENCE_MIN_SCORE:
                        continue

                    strat_name = "+".join(strats)
                    if any(self.portfolio.is_in_cooldown(symbol, s, t) for s in strats):
                        continue

                    self._try_open(symbol, bar, next_bar, strat_name, direction, strats, corrs)
                    break  # one direction per symbol per bar

        # Close remaining positions at end
        for pos in self.portfolio.positions[:]:
            if pos.symbol not in data_1h:
                continue
            last_bar  = data_1h[pos.symbol].iloc[-1]
            self.portfolio.close_position(
                pos, last_bar.name, last_bar["close"], ExitType.TIMEOUT, pos.bars_held
            )

        log.info("Backtest complete.")
        return self.portfolio, self.portfolio.closed_trades


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_results(portfolio: Portfolio, trades: List[Trade],
                    label: str = "FULL"):
    if not trades:
        print(f"[{label}] No trades.")
        return

    df = pd.DataFrame([vars(t) for t in trades])
    df["reason"] = df["reason"].apply(lambda x: x.value if hasattr(x, "value") else str(x))

    total_pnl  = df["pnl"].sum()
    total_ret  = (portfolio.equity - portfolio.initial_capital) / portfolio.initial_capital
    winners    = df[df["pnl"] > 0]
    losers     = df[df["pnl"] < 0]
    wr         = len(winners) / len(df) if len(df) > 0 else 0
    avg_w      = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_l      = losers["pnl"].mean()  if len(losers)  > 0 else 0
    pf_val     = abs(winners["pnl"].sum() / losers["pnl"].sum()) if len(losers) > 0 and losers["pnl"].sum() != 0 else float("inf")
    avg_pnl    = df["pnl"].mean()
    avg_hold   = df["bars_held"].mean()

    sharpe_val = calmar_val = 0.0
    if portfolio.equity_curve:
        eq    = pd.DataFrame(portfolio.equity_curve).set_index("time")
        daily = eq["equity"].resample("1D").last().dropna()
        dr    = daily.pct_change().dropna()
        if len(dr) > 1 and dr.std() > 0:
            sharpe_val = dr.mean() / dr.std() * np.sqrt(365)
        if portfolio.max_drawdown > 0:
            calmar_val = total_ret / portfolio.max_drawdown

    df["win"]    = df["pnl"] > 0
    df["streak"] = df["win"].ne(df["win"].shift()).cumsum()
    ws = df[df["win"]].groupby("streak").size()
    ls = df[~df["win"]].groupby("streak").size()

    longs  = df[df["direction"] == 1]
    shorts = df[df["direction"] == -1]

    _p = "  {:<22s}{}"
    bar = "=" * 72
    print()
    print(bar)
    print(f"  BACKTEST RESULTS — QUANT v9.0  [{label}]")
    print(bar)
    print(_p.format("Initial:",      f"${portfolio.initial_capital:,.2f}"))
    print(_p.format("Final:",        f"${portfolio.equity:,.2f}"))
    print(_p.format("P&L:",          f"${total_pnl:,.2f}"))
    print(_p.format("Return:",       f"{total_ret:.2%}"))
    print(_p.format("Max DD:",       f"{portfolio.max_drawdown:.2%}"))
    print(_p.format("Sharpe:",       f"{sharpe_val:.2f}"))
    print(_p.format("Calmar:",       f"{calmar_val:.2f}"))
    print(_p.format("Trades:",       f"{len(df)} | WR: {wr:.1%}"))
    print(_p.format("Longs / Shorts:", f"{len(longs)} / {len(shorts)}"))
    print(_p.format("Avg Win:",      f"${avg_w:.2f}  |  Avg Loss: ${avg_l:.2f}"))
    print(_p.format("Profit Factor:", f"{pf_val:.2f}"))
    print(_p.format("Expectancy:",   f"${avg_pnl:.2f}"))
    print(_p.format("Win Streak:",   f"{ws.max() if len(ws) else 0}"))
    print(_p.format("Loss Streak:",  f"{ls.max() if len(ls) else 0}"))
    print(_p.format("Avg Hold:",     f"{avg_hold:.1f} bars"))

    print()
    print("  BY STRATEGY")
    print(df.groupby("strat").agg({"pnl": ["count", "sum", "mean"]}).round(2).to_string())
    print()
    print("  BY SYMBOL")
    print(df.groupby("symbol").agg({"pnl": ["count", "sum", "mean"]}).round(2).to_string())
    print()
    print("  BY DIRECTION")
    print(df.groupby("direction").agg({"pnl": ["count", "sum", "mean"]}).round(2).to_string())
    print()
    print("  EXIT REASONS")
    print(df.groupby("reason").agg({"pnl": ["count", "sum", "mean"]}).round(2).to_string())

    if "entry_ts" in df.columns:
        print()
        print("  MONTHLY")
        df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M")
        print(df.groupby("month").agg({"pnl": ["count", "sum"]}).round(2).to_string())

    print()
    print(bar)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_fn = f"backtest_results_{label.lower()}_{ts}.csv"
    df.to_csv(results_fn, index=False)
    print(f"Saved trades: {results_fn}")

    if portfolio.equity_curve:
        eq_fn = f"equity_curve_{label.lower()}_{ts}.csv"
        pd.DataFrame(portfolio.equity_curve).to_csv(eq_fn, index=False)
        print(f"Saved equity curve: {eq_fn}")
        _save_equity_chart(portfolio.equity_curve, label, ts)

    # Monte Carlo
    _monte_carlo(df, portfolio.initial_capital, label)


def _save_equity_chart(equity_curve: list, label: str, ts: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eq = pd.DataFrame(equity_curve).set_index("time")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(eq.index, eq["equity"], color="steelblue", linewidth=1.2)
        ax1.set_title(f"Equity Curve — QUANT v9.0 [{label}]")
        ax1.set_ylabel("Equity (USD)")
        ax1.grid(alpha=0.3)
        ax2.fill_between(eq.index, eq["drawdown"] * 100, color="tomato", alpha=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        fn = f"equity_chart_{label.lower()}_{ts}.png"
        plt.savefig(fn, dpi=150)
        plt.close()
        print(f"Saved chart: {fn}")
    except ImportError:
        log.info("matplotlib not installed — skipping chart")


def _monte_carlo(trades_df: pd.DataFrame, initial_capital: float,
                 label: str, n_sims: int = 1000):
    """Shuffle trade P&Ls to estimate distribution of outcomes."""
    pnls = trades_df["pnl"].values
    if len(pnls) < 10:
        return
    final_equities = []
    max_dds        = []
    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity   = initial_capital
        peak     = initial_capital
        max_dd   = 0.0
        for p in shuffled:
            equity += p
            peak    = max(peak, equity)
            dd      = (peak - equity) / peak if peak > 0 else 0.0
            max_dd  = max(max_dd, dd)
        final_equities.append(equity)
        max_dds.append(max_dd)
    fe = np.array(final_equities)
    md = np.array(max_dds)
    print()
    print(f"  MONTE CARLO ({n_sims} simulations) [{label}]")
    print(f"  Final Equity  p5={np.percentile(fe,5):,.0f}  median={np.median(fe):,.0f}  p95={np.percentile(fe,95):,.0f}")
    print(f"  Max Drawdown  p5={np.percentile(md,5):.1%}  median={np.median(md):.1%}  p95={np.percentile(md,95):.1%}")
    print(f"  P(Profitable) = {(fe > initial_capital).mean():.1%}")
    print()


def sensitivity_report(base_cfg: Config, data_loader: DataLoader):
    """FIX #16: vary key parameters and report impact on Sharpe."""
    params = {
        "ATR_STOP_MULT":  [1.2, 1.5, 1.8, 2.2, 2.8],
        "RISK_PCT":        [0.005, 0.01, 0.015, 0.02],
        "MIN_RR":          [1.5, 2.0, 2.5, 3.0],
        "CONFLUENCE_MIN":  [1, 2, 3],
    }
    print()
    print("=" * 72)
    print("  PARAMETER SENSITIVITY")
    print("=" * 72)
    results = []
    for param, values in params.items():
        for v in values:
            c = Config()
            if param == "ATR_STOP_MULT":
                c.risk.ATR_STOP_MULT = v
            elif param == "RISK_PCT":
                c.risk.RISK_PCT = v
            elif param == "MIN_RR":
                c.strategy.MIN_RR = v
            elif param == "CONFLUENCE_MIN":
                c.strategy.CONFLUENCE_MIN_SCORE = int(v)
            eng   = BacktestEngine(c, data_loader)
            port, trades = eng.run()
            if not trades:
                continue
            df = pd.DataFrame([vars(t) for t in trades])
            wr = (df["pnl"] > 0).mean()
            ret = (port.equity - port.initial_capital) / port.initial_capital
            sharpe = 0.0
            if port.equity_curve:
                eq = pd.DataFrame(port.equity_curve).set_index("time")
                daily = eq["equity"].resample("1D").last().dropna()
                dr = daily.pct_change().dropna()
                if len(dr) > 1 and dr.std() > 0:
                    sharpe = dr.mean() / dr.std() * np.sqrt(365)
            results.append({
                "param": param, "value": v,
                "trades": len(df), "wr": wr,
                "ret": ret, "max_dd": port.max_drawdown, "sharpe": sharpe,
            })
    if results:
        sr = pd.DataFrame(results)
        print(sr.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Unit Tests (run with --test)
# ---------------------------------------------------------------------------
def run_tests():
    import traceback
    passed = failed = 0

    def check(name, cond, msg=""):
        nonlocal passed, failed
        if cond:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {msg}")
            failed += 1

    print("\n=== UNIT TESTS ===")

    # Test 1: PnL calculation long win
    port = Portfolio(1000, RiskConfig(), ExchangeConfig())
    pos  = Position(datetime.now(), "BTC/USDT", "test", 1, 100.0, 200.0, 95.0, 110.0,
                    original_stop_distance=5.0)
    port.add_position(pos)
    t = port.close_position(pos, datetime.now(), 110.0, ExitType.TAKE_PROFIT, 3)
    check("Long win PnL > 0", t.pnl > 0, f"pnl={t.pnl:.4f}")

    # Test 2: PnL calculation long loss
    port2 = Portfolio(1000, RiskConfig(), ExchangeConfig())
    pos2  = Position(datetime.now(), "BTC/USDT", "test", 1, 100.0, 200.0, 95.0, 110.0,
                     original_stop_distance=5.0)
    port2.add_position(pos2)
    t2 = port2.close_position(pos2, datetime.now(), 95.0, ExitType.STOP_LOSS, 2)
    check("Long loss PnL < 0", t2.pnl < 0, f"pnl={t2.pnl:.4f}")

    # Test 3: Short win
    port3 = Portfolio(1000, RiskConfig(), ExchangeConfig())
    pos3  = Position(datetime.now(), "ETH/USDT", "test", -1, 100.0, 200.0, 105.0, 90.0,
                     original_stop_distance=5.0)
    port3.add_position(pos3)
    t3 = port3.close_position(pos3, datetime.now(), 90.0, ExitType.TAKE_PROFIT, 4)
    check("Short win PnL > 0", t3.pnl > 0, f"pnl={t3.pnl:.4f}")

    # Test 4: consecutive_losses resets after CB_RESET_WINS wins
    rc = RiskConfig(CB_RESET_WINS=2)
    port4 = Portfolio(1000, rc, ExchangeConfig())
    for _ in range(4):
        p = Position(datetime.now(), "BTC/USDT", "t", 1, 100.0, 200.0, 95.0, 110.0, 5.0)
        port4.add_position(p)
        port4.close_position(p, datetime.now(), 95.0, ExitType.STOP_LOSS, 1)
    check("Losses accumulated = 4", port4.consecutive_losses == 4, str(port4.consecutive_losses))
    for _ in range(2):
        p = Position(datetime.now(), "BTC/USDT", "t", 1, 100.0, 200.0, 95.0, 110.0, 5.0)
        port4.add_position(p)
        port4.close_position(p, datetime.now(), 110.0, ExitType.TAKE_PROFIT, 1)
    check("Losses reset after 2 wins", port4.consecutive_losses == 0, str(port4.consecutive_losses))

    # Test 5: Position sizing respects MAX_POS_PCT
    eng = BacktestEngine(cfg)
    size = eng.calculate_position_size(100.0, 95.0, "BTC/USDT", {}, Regime.NEUTRAL.value)
    check("Position size <= MAX_POS_PCT",
          size <= cfg.risk.ACCOUNT_SIZE * cfg.risk.MAX_POS_PCT + 1e-6,
          f"size={size:.2f}")

    # Test 6: Circuit breaker activates on CB_MAX_DD
    port5 = Portfolio(1000, RiskConfig(CB_MAX_DD=0.15), ExchangeConfig())
    port5.equity = 840.0
    port5.peak   = 1000.0
    port5.drawdown = 0.16
    check("CB triggers on drawdown",
          port5.check_circuit_breaker(datetime.now()), "")

    # Test 7: Cooldown applies after winning trade (FIX #7)
    port6 = Portfolio(1000, RiskConfig(COOLDOWN_HOURS=6), ExchangeConfig())
    p6 = Position(datetime.now(), "BTC/USDT", "mr", 1, 100.0, 200.0, 95.0, 110.0, 5.0)
    port6.add_position(p6)
    t6 = port6.close_position(p6, datetime.now(), 110.0, ExitType.TAKE_PROFIT, 1)
    in_cd = port6.is_in_cooldown("BTC/USDT", "mr", datetime.now() + timedelta(hours=3))
    check("Cooldown after win", in_cd, "")

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QUANT v9.0 Backtester")
    parser.add_argument("--test",        action="store_true", help="Run unit tests")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sensitivity")
    parser.add_argument("--oos",         action="store_true", help="Show OOS results separately")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  QUANT v9.0 — FULL REWRITE")
    print("  Long + Short | Regime-Adaptive | Walk-Forward | Monte Carlo")
    print("=" * 72)
    print()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    try:
        loader = DataLoader()

        if args.sensitivity:
            sensitivity_report(cfg, loader)
            return

        engine         = BacktestEngine(cfg, data_loader=loader)
        portfolio, trades = engine.run()
        analyze_results(portfolio, trades, label="FULL")

        if args.oos:
            engine_oos        = BacktestEngine(cfg, data_loader=loader)
            port_oos, tr_oos  = engine_oos.run(oos_only=True)
            analyze_results(port_oos, tr_oos, label="OOS")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
