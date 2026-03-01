"""
BYBIT FUTURES v1.0 — Extreme Funding Rate Strategy
=====================================================
Стратегия: Контр-трендовая торговля на экстремальных значениях funding rate
           с подтверждением через Open Interest и RSI.

Логика:
  ШОРТ: funding > 0.1% + OI не растёт + цена не делает новый хай + RSI > 65
  ЛОНГ: funding < -0.05% + OI стабилен/падает + цена не делает новый лой + RSI < 35

Пары: Топ-10 по объёму (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT)
Плечо: 2-3x
Таймфрейм: 1H
Биржа: Bybit (через ccxt)
"""

import os
import sys
import time
import logging
import argparse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import ccxt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "bybit_futures_v1.log"
log = logging.getLogger("bfv1")
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
class ExitType(Enum):
    STOP_LOSS       = "STOP_LOSS"
    TAKE_PROFIT     = "TAKE_PROFIT"
    TIMEOUT         = "TIMEOUT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    TRAILING        = "TRAILING"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    # Пары для торговли (топ-10 по объёму на Bybit)
    SYMBOLS: Tuple[str, ...] = (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "BNB/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
        "DOT/USDT:USDT",
    )
    TIMEFRAME:              str   = "1h"
    DAYS_HISTORY:           int   = 180

    # Funding Rate пороги
    FUNDING_LONG_THRESHOLD:  float = -0.0005   # < -0.05% → сигнал лонг
    FUNDING_SHORT_THRESHOLD: float =  0.0010   # > 0.10% → сигнал шорт

    # OI фильтр — максимальный рост OI за последние N баров
    OI_LOOKBACK:            int   = 3
    OI_MAX_GROWTH_SHORT:    float = 0.02   # OI не должен расти более 2% для шорта
    OI_MAX_GROWTH_LONG:     float = 0.02   # OI не должен расти более 2% для лонга

    # Ценовые фильтры
    PRICE_LOOKBACK:         int   = 4     # баров для проверки нового хая/лоя

    # RSI
    RSI_PERIOD:             int   = 14
    RSI_SHORT:              float = 65.0
    RSI_LONG:               float = 35.0

    # ATR
    ATR_PERIOD:             int   = 14


@dataclass
class RiskConfig:
    ACCOUNT_SIZE:           float = 500.0
    LEVERAGE:               float = 2.0
    RISK_PCT:               float = 0.01    # 1% риска на сделку
    MAX_CONCURRENT:         int   = 3       # максимум открытых позиций
    MAX_POS_PCT:            float = 0.20    # максимум 20% депозита на позицию

    # Stop/TP
    ATR_STOP_MULT:          float = 1.5
    ATR_TP_MULT:            float = 2.5    # RR = 1.67

    # Timeout
    MAX_HOLD_HOURS:         int   = 8      # до следующего funding

    # Trailing stop
    TRAIL_ACTIVATE_RR:      float = 1.0    # активировать после 1R прибыли
    TRAIL_ATR_MULT:         float = 1.0

    # Circuit Breaker
    CB_MAX_DD:              float = 0.15   # 15% просадка → стоп
    CB_CONSECUTIVE_LOSSES:  int   = 6
    CB_RESET_WINS:          int   = 2
    CB_PAUSE_HOURS:         int   = 12

    # Cooldown после сделки
    COOLDOWN_HOURS:         int   = 2

    # Walk-Forward
    WF_OOS_FRACTION:        float = 0.30


@dataclass
class ExchangeConfig:
    FEE_TAKER:   float = 0.00055   # Bybit taker fee
    FEE_MAKER:   float = 0.00020   # Bybit maker fee
    SLIPPAGE:    float = 0.0003    # 0.03% slippage


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
    CACHE_DIR = Path("data_cache_bybit")

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })

    def _cache_path(self, symbol: str, tf: str, days: int) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.CACHE_DIR / f"{safe}_{tf}_{days}d.pkl"

    def fetch_ohlcv(self, symbol: str, days: int = None) -> pd.DataFrame:
        days = days or cfg.strategy.DAYS_HISTORY
        tf   = cfg.strategy.TIMEFRAME
        path = self._cache_path(symbol, tf, days)

        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < 3600:  # кэш 1 час
                return pd.read_pickle(path)

        log.info(f"Fetching OHLCV: {symbol} {tf} {days}d")
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        all_bars = []
        while True:
            bars = self.exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < 1000:
                break
            since = bars[-1][0] + 1
            time.sleep(0.2)

        df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        df.to_pickle(path)
        log.info(f"  {symbol}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
        return df

    def fetch_funding_history(self, symbol: str, days: int = None) -> pd.Series:
        """Получить историю funding rate."""
        days = days or cfg.strategy.DAYS_HISTORY
        safe = symbol.replace("/", "_").replace(":", "_")
        path = self.CACHE_DIR / f"{safe}_funding_{days}d.pkl"

        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < 3600:
                return pd.read_pickle(path)

        log.info(f"Fetching funding history: {symbol}")
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        try:
            raw = self.exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not raw:
                log.warning(f"No funding data for {symbol}")
                return pd.Series(dtype=float)

            df = pd.DataFrame(raw)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)
            series = df["fundingRate"].sort_index()
            series.to_pickle(path)
            log.info(f"  {symbol}: {len(series)} funding records")
            return series
        except Exception as e:
            log.error(f"Funding fetch error {symbol}: {e}")
            return pd.Series(dtype=float)

    def fetch_oi_history(self, symbol: str, days: int = None) -> pd.Series:
        """Получить историю Open Interest."""
        days = days or cfg.strategy.DAYS_HISTORY
        safe = symbol.replace("/", "_").replace(":", "_")
        path = self.CACHE_DIR / f"{safe}_oi_{days}d.pkl"

        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < 3600:
                return pd.read_pickle(path)

        log.info(f"Fetching OI history: {symbol}")
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        try:
            raw = self.exchange.fetch_open_interest_history(
                symbol, "1h", since=since, limit=1000
            )
            if not raw:
                log.warning(f"No OI data for {symbol}")
                return pd.Series(dtype=float)

            df = pd.DataFrame(raw)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)
            series = df["openInterestValue"].sort_index().astype(float)
            series.to_pickle(path)
            log.info(f"  {symbol}: {len(series)} OI records")
            return series
        except Exception as e:
            log.error(f"OI fetch error {symbol}: {e}")
            return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    # ATR
    try:
        import talib
        df["atr"] = talib.ATR(h, l, c, cfg.strategy.ATR_PERIOD)
        df["rsi"] = talib.RSI(c, cfg.strategy.RSI_PERIOD)
    except ImportError:
        # Fallback без talib
        df["atr"] = pd.Series(h - l, index=df.index).rolling(cfg.strategy.ATR_PERIOD).mean()
        delta = pd.Series(c, index=df.index).diff()
        gain  = delta.clip(lower=0).rolling(cfg.strategy.RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(cfg.strategy.RSI_PERIOD).mean()
        rs    = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

    # Новые хаи/лои за N баров
    n = cfg.strategy.PRICE_LOOKBACK
    df["new_high"] = df["high"] == df["high"].rolling(n).max()
    df["new_low"]  = df["low"]  == df["low"].rolling(n).min()

    return df


def merge_funding(df: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Добавляем funding rate к OHLCV — каждые 8 часов."""
    if funding.empty:
        df["funding"] = 0.0
        return df

    # Funding выплачивается каждые 8 часов — forward fill на 1H бары
    funding_reindexed = funding.reindex(df.index, method="ffill")
    df["funding"] = funding_reindexed.fillna(0.0)
    return df


def merge_oi(df: pd.DataFrame, oi: pd.Series) -> pd.DataFrame:
    """Добавляем OI к OHLCV."""
    if oi.empty:
        df["oi"] = np.nan
        df["oi_growth"] = 0.0
        return df

    oi_reindexed = oi.reindex(df.index, method="ffill")
    df["oi"] = oi_reindexed
    n = cfg.strategy.OI_LOOKBACK
    df["oi_growth"] = (df["oi"] - df["oi"].shift(n)) / (df["oi"].shift(n) + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Strategy Signals
# ---------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Генерация сигналов на основе Extreme Funding Rate.

    +1 = ЛОНГ (funding экстремально отрицательный — шорты перегружены)
    -1 = ШОРТ (funding экстремально положительный — лонги перегружены)
     0 = нет сигнала
    """
    signals = pd.Series(0, index=df.index)

    # --- ШОРТ сигнал ---
    # 1. Funding перегрет в сторону лонгов
    funding_extreme_short = df["funding"] >= cfg.strategy.FUNDING_SHORT_THRESHOLD
    # 2. OI не растёт (нет новых покупателей)
    oi_not_growing_short  = df["oi_growth"] <= cfg.strategy.OI_MAX_GROWTH_SHORT
    # 3. Цена не делает новый хай (слабость)
    no_new_high           = ~df["new_high"]
    # 4. RSI перекуплен
    rsi_overbought        = df["rsi"] >= cfg.strategy.RSI_SHORT

    short_signal = (
        funding_extreme_short
        & oi_not_growing_short
        & no_new_high
        & rsi_overbought
    )

    # --- ЛОНГ сигнал ---
    # 1. Funding перегрет в сторону шортов
    funding_extreme_long  = df["funding"] <= cfg.strategy.FUNDING_LONG_THRESHOLD
    # 2. OI не растёт (нет новых продавцов)
    oi_not_growing_long   = df["oi_growth"] <= cfg.strategy.OI_MAX_GROWTH_LONG
    # 3. Цена не делает новый лой
    no_new_low            = ~df["new_low"]
    # 4. RSI перепродан
    rsi_oversold          = df["rsi"] <= cfg.strategy.RSI_LONG

    long_signal = (
        funding_extreme_long
        & oi_not_growing_long
        & no_new_low
        & rsi_oversold
    )

    signals[short_signal] = -1
    signals[long_signal]  =  1

    return signals


# ---------------------------------------------------------------------------
# Position & Trade
# ---------------------------------------------------------------------------
@dataclass
class Position:
    entry_ts:              datetime
    symbol:                str
    direction:             int      # +1 long, -1 short
    entry_price:           float
    size_usd:              float
    stop_loss:             float
    take_profit:           float
    original_stop_dist:    float
    funding_at_entry:      float = 0.0
    trailing_active:       bool  = False
    trailing_stop:         float = 0.0


@dataclass
class Trade:
    entry_ts:    datetime
    exit_ts:     datetime
    symbol:      str
    direction:   int
    entry_price: float
    exit_price:  float
    size_usd:    float
    pnl:         float
    pnl_pct:     float
    reason:      str
    bars_held:   int
    funding_pnl: float = 0.0
    win:         bool  = False


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class Portfolio:
    def __init__(self, account_size: float, rcfg: RiskConfig, ecfg: ExchangeConfig):
        self.equity            = account_size
        self.peak              = account_size
        self.drawdown          = 0.0
        self.rcfg              = rcfg
        self.ecfg              = ecfg
        self.positions: Dict[str, Position] = {}
        self.trades:    List[Trade]         = []
        self.consecutive_losses = 0
        self.consecutive_wins   = 0
        self.cb_active          = False
        self.cb_resume_time:    Optional[datetime] = None
        self.cooldowns:         Dict[str, datetime] = {}

    def check_circuit_breaker(self, ts: datetime) -> bool:
        if self.cb_active and self.cb_resume_time and ts >= self.cb_resume_time:
            self.cb_active = False
            log.info(f"Circuit breaker lifted at {ts}")

        if self.cb_active:
            return True

        if self.drawdown >= self.rcfg.CB_MAX_DD:
            self.cb_active       = True
            self.cb_resume_time  = ts + timedelta(hours=self.rcfg.CB_PAUSE_HOURS)
            log.warning(f"CIRCUIT BREAKER: Max DD {self.drawdown:.1%} | Resume at {self.cb_resume_time}")
            return True

        if self.consecutive_losses >= self.rcfg.CB_CONSECUTIVE_LOSSES:
            self.cb_active       = True
            self.cb_resume_time  = ts + timedelta(hours=self.rcfg.CB_PAUSE_HOURS)
            log.warning(f"CIRCUIT BREAKER: {self.consecutive_losses} losses | Resume at {self.cb_resume_time}")
            return True

        return False

    def is_in_cooldown(self, symbol: str, ts: datetime) -> bool:
        key = symbol
        if key in self.cooldowns and ts < self.cooldowns[key]:
            return True
        return False

    def calculate_position_size(self, entry: float, stop: float) -> float:
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return 0.0

        risk_amount = self.equity * self.rcfg.RISK_PCT
        size_by_risk = (risk_amount / stop_dist) * entry

        # Применяем плечо
        size_with_leverage = size_by_risk * self.rcfg.LEVERAGE

        # Ограничиваем максимальным размером позиции
        max_size = self.equity * self.rcfg.MAX_POS_PCT * self.rcfg.LEVERAGE
        size = min(size_with_leverage, max_size)

        return max(size, 0.0)

    def open_position(self, ts: datetime, symbol: str, direction: int,
                      price: float, atr: float, funding: float) -> Optional[Position]:

        if self.check_circuit_breaker(ts):
            return None
        if self.is_in_cooldown(symbol, ts):
            return None
        if symbol in self.positions:
            return None
        if len(self.positions) >= self.rcfg.MAX_CONCURRENT:
            return None

        stop_dist  = atr * self.rcfg.ATR_STOP_MULT
        stop_loss  = price - stop_dist * direction
        take_profit = price + atr * self.rcfg.ATR_TP_MULT * direction

        size = self.calculate_position_size(price, stop_loss)
        if size <= 0:
            return None

        # Комиссия на вход
        fee    = size * self.ecfg.FEE_TAKER
        slip   = size * self.ecfg.SLIPPAGE
        self.equity -= (fee + slip)

        pos = Position(
            entry_ts           = ts,
            symbol             = symbol,
            direction          = direction,
            entry_price        = price,
            size_usd           = size,
            stop_loss          = stop_loss,
            take_profit        = take_profit,
            original_stop_dist = stop_dist,
            funding_at_entry   = funding,
        )
        self.positions[symbol] = pos
        log.info(
            f"OPEN {'LONG' if direction == 1 else 'SHORT'} {symbol} "
            f"@ {price:.4f} | SL={stop_loss:.4f} TP={take_profit:.4f} "
            f"size=${size:.0f} funding={funding:.4%}"
        )
        return pos

    def close_position(self, symbol: str, ts: datetime, price: float,
                       reason: ExitType, bars_held: int,
                       funding_series: Optional[pd.Series] = None) -> Optional[Trade]:

        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        direction  = pos.direction
        raw_pnl    = (price - pos.entry_price) * direction * (pos.size_usd / pos.entry_price)

        # Funding PnL — собираем funding за время удержания
        funding_pnl = 0.0
        if funding_series is not None and not funding_series.empty:
            mask = (funding_series.index > pos.entry_ts) & (funding_series.index <= ts)
            period_funding = funding_series[mask].sum()
            # Шорт получает funding когда он положительный
            funding_pnl = period_funding * pos.size_usd * (-direction)

        # Комиссия на выход
        fee  = pos.size_usd * self.ecfg.FEE_TAKER
        slip = pos.size_usd * self.ecfg.SLIPPAGE

        net_pnl = raw_pnl + funding_pnl - fee - slip

        self.equity += net_pnl
        if self.equity > self.peak:
            self.peak = self.equity
        self.drawdown = (self.peak - self.equity) / self.peak

        win = net_pnl > 0
        if win:
            self.consecutive_wins  += 1
            self.consecutive_losses = 0
            if self.consecutive_wins >= self.rcfg.CB_RESET_WINS:
                self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins   = 0

        # Cooldown
        self.cooldowns[symbol] = ts + timedelta(hours=self.rcfg.COOLDOWN_HOURS)

        trade = Trade(
            entry_ts    = pos.entry_ts,
            exit_ts     = ts,
            symbol      = symbol,
            direction   = direction,
            entry_price = pos.entry_price,
            exit_price  = price,
            size_usd    = pos.size_usd,
            pnl         = round(net_pnl, 4),
            pnl_pct     = round(net_pnl / pos.size_usd, 6) if pos.size_usd > 0 else 0,
            reason      = reason.value,
            bars_held   = bars_held,
            funding_pnl = round(funding_pnl, 4),
            win         = win,
        )
        self.trades.append(trade)

        log.info(
            f"CLOSE {symbol} @ {price:.4f} | "
            f"PnL=${net_pnl:+.2f} ({trade.pnl_pct:+.2%}) | "
            f"{reason.value} | {bars_held}bars | "
            f"funding_pnl=${funding_pnl:+.2f}"
        )
        return trade

    def update_trailing(self, symbol: str, price: float, atr: float) -> Optional[float]:
        """Обновление trailing stop. Возвращает новый стоп или None."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        profit_r = (price - pos.entry_price) * pos.direction / pos.original_stop_dist

        if profit_r >= self.rcfg.TRAIL_ACTIVATE_RR:
            trail = price - atr * self.rcfg.TRAIL_ATR_MULT * pos.direction
            if not pos.trailing_active:
                pos.trailing_active = True
                pos.trailing_stop   = trail
                log.info(f"Trailing activated {symbol} trail={trail:.4f}")
            else:
                if pos.direction == 1:
                    pos.trailing_stop = max(pos.trailing_stop, trail)
                else:
                    pos.trailing_stop = min(pos.trailing_stop, trail)

        return pos.trailing_stop if pos.trailing_active else None


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    def __init__(self, config: Config, data_loader: Optional[DataLoader] = None):
        self.cfg    = config
        self.loader = data_loader or DataLoader()

    def _load_symbol_data(self, symbol: str) -> Optional[Dict]:
        try:
            df       = self.loader.fetch_ohlcv(symbol)
            funding  = self.loader.fetch_funding_history(symbol)
            oi       = self.loader.fetch_oi_history(symbol)

            df = compute_indicators(df)
            df = merge_funding(df, funding)
            df = merge_oi(df, oi)
            df.dropna(subset=["atr", "rsi"], inplace=True)

            signals = generate_signals(df)
            df["signal"] = signals

            # Диагностика сигналов
            n_long  = (signals == 1).sum()
            n_short = (signals == -1).sum()
            log.info(f"{symbol}: {n_long} long signals, {n_short} short signals")

            return {"df": df, "funding": funding}
        except Exception as e:
            log.error(f"Failed to load {symbol}: {e}")
            return None

    def run(self, oos_only: bool = False) -> Tuple[Portfolio, List[Trade]]:
        portfolio = Portfolio(
            self.cfg.risk.ACCOUNT_SIZE,
            self.cfg.risk,
            self.cfg.exchange,
        )

        # Загружаем данные для всех символов
        symbol_data = {}
        for symbol in self.cfg.strategy.SYMBOLS:
            data = self._load_symbol_data(symbol)
            if data:
                symbol_data[symbol] = data

        if not symbol_data:
            log.error("No data loaded")
            return portfolio, []

        # Определяем общий временной диапазон
        all_indices = [d["df"].index for d in symbol_data.values()]
        common_start = max(idx[0] for idx in all_indices)
        common_end   = min(idx[-1] for idx in all_indices)

        if oos_only:
            oos_start = common_start + (common_end - common_start) * (
                1 - self.cfg.risk.WF_OOS_FRACTION
            )
            common_start = oos_start

        log.info(f"Backtest period: {common_start} → {common_end}")

        # Получаем все временные метки
        ref_symbol = list(symbol_data.keys())[0]
        timestamps = symbol_data[ref_symbol]["df"].loc[common_start:common_end].index

        # Основной цикл
        for ts in timestamps:
            # Обновляем открытые позиции
            for symbol in list(portfolio.positions.keys()):
                if symbol not in symbol_data:
                    continue

                df  = symbol_data[symbol]["df"]
                if ts not in df.index:
                    continue

                row      = df.loc[ts]
                pos      = portfolio.positions[symbol]
                price    = row["close"]
                atr      = row["atr"]
                bars_held = int((ts - pos.entry_ts).total_seconds() / 3600)

                # Проверяем trailing stop
                trail = portfolio.update_trailing(symbol, price, atr)

                # Условия выхода
                exit_price  = None
                exit_reason = None

                if pos.direction == 1:  # ЛОНГ
                    if price <= pos.stop_loss:
                        exit_price  = min(price, pos.stop_loss)
                        exit_reason = ExitType.STOP_LOSS
                    elif price >= pos.take_profit:
                        exit_price  = pos.take_profit
                        exit_reason = ExitType.TAKE_PROFIT
                    elif trail and price <= trail:
                        exit_price  = price
                        exit_reason = ExitType.TRAILING
                else:  # ШОРТ
                    if price >= pos.stop_loss:
                        exit_price  = max(price, pos.stop_loss)
                        exit_reason = ExitType.STOP_LOSS
                    elif price <= pos.take_profit:
                        exit_price  = pos.take_profit
                        exit_reason = ExitType.TAKE_PROFIT
                    elif trail and price >= trail:
                        exit_price  = price
                        exit_reason = ExitType.TRAILING

                if bars_held >= self.cfg.risk.MAX_HOLD_HOURS:
                    exit_price  = price
                    exit_reason = ExitType.TIMEOUT

                if exit_reason:
                    portfolio.close_position(
                        symbol, ts, exit_price, exit_reason, bars_held,
                        symbol_data[symbol]["funding"]
                    )

            # Проверяем новые сигналы
            if portfolio.check_circuit_breaker(ts):
                continue

            for symbol, data in symbol_data.items():
                if symbol in portfolio.positions:
                    continue

                df  = data["df"]
                if ts not in df.index:
                    continue

                row     = df.loc[ts]
                signal  = row["signal"]
                if signal == 0:
                    continue

                price   = row["close"]
                atr     = row["atr"]
                funding = row["funding"]

                portfolio.open_position(ts, symbol, int(signal), price, atr, funding)

        # Закрываем оставшиеся позиции
        last_ts = timestamps[-1]
        for symbol in list(portfolio.positions.keys()):
            df = symbol_data[symbol]["df"]
            if last_ts in df.index:
                price = df.loc[last_ts, "close"]
                bars_held = int(
                    (last_ts - portfolio.positions[symbol].entry_ts).total_seconds() / 3600
                )
                portfolio.close_position(
                    symbol, last_ts, price, ExitType.TIMEOUT, bars_held,
                    symbol_data[symbol]["funding"]
                )

        return portfolio, portfolio.trades


# ---------------------------------------------------------------------------
# Results Analysis
# ---------------------------------------------------------------------------
def analyze_results(portfolio: Portfolio, trades: List[Trade], label: str = "FULL"):
    n = len(trades)
    print()
    print("=" * 72)
    print(f"  BACKTEST RESULTS — BYBIT FUTURES v1.0  [{label}]")
    print("=" * 72)

    initial = cfg.risk.ACCOUNT_SIZE
    final   = portfolio.equity
    pnl     = final - initial
    ret     = pnl / initial

    print(f"  Initial:              ${initial:,.2f}")
    print(f"  Final:                ${final:,.2f}")
    print(f"  P&L:                  ${pnl:+,.2f}")
    print(f"  Return:               {ret:+.2%}")
    print(f"  Max DD:               {portfolio.drawdown:.2%}")

    if n == 0:
        print(f"  Trades:               0")
        print("=" * 72)
        return

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["month"] = df["entry_ts"].dt.to_period("M").astype(str)

    wins     = df[df["win"]]
    losses   = df[~df["win"]]
    wr       = len(wins) / n
    avg_win  = wins["pnl"].mean()  if len(wins)  > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    # Sharpe
    if df["pnl_pct"].std() > 0:
        sharpe = (df["pnl_pct"].mean() / df["pnl_pct"].std()) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0

    # Calmar
    calmar = ret / portfolio.drawdown if portfolio.drawdown > 0 else 0.0

    # Profit Factor
    gross_profit = wins["pnl"].sum()  if len(wins)  > 0 else 0
    gross_loss   = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    longs  = (df["direction"] ==  1).sum()
    shorts = (df["direction"] == -1).sum()

    print(f"  Sharpe:               {sharpe:.2f}")
    print(f"  Calmar:               {calmar:.2f}")
    print(f"  Trades:               {n} | WR: {wr:.1%}")
    print(f"  Longs / Shorts:       {longs} / {shorts}")
    print(f"  Avg Win:              ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print(f"  Profit Factor:        {pf:.2f}")
    print(f"  Expectancy:           ${df['pnl'].mean():.2f}")
    print(f"  Avg Hold:             {df['bars_held'].mean():.1f} bars")
    print(f"  Total Funding PnL:    ${df['funding_pnl'].sum():+.2f}")

    print()
    print("  BY SYMBOL")
    print(df.groupby("symbol")["pnl"].agg(["count", "sum", "mean"]).to_string())

    print()
    print("  BY DIRECTION")
    print(df.groupby("direction")["pnl"].agg(["count", "sum", "mean"]).to_string())

    print()
    print("  EXIT REASONS")
    print(df.groupby("reason")["pnl"].agg(["count", "sum", "mean"]).to_string())

    print()
    print("  MONTHLY")
    print(df.groupby("month")["pnl"].agg(["count", "sum"]).to_string())

    # Monte Carlo
    print()
    print(f"  MONTE CARLO (1000 simulations) [{label}]")
    pnls   = df["pnl"].values
    finals = []
    max_dds = []
    for _ in range(1000):
        shuffled = np.random.choice(pnls, size=len(pnls), replace=True)
        equity   = np.cumsum(shuffled) + initial
        peak     = np.maximum.accumulate(equity)
        dd       = ((peak - equity) / peak).max()
        finals.append(equity[-1])
        max_dds.append(dd)

    finals  = np.array(finals)
    max_dds = np.array(max_dds)
    print(f"  Final Equity  p5=${np.percentile(finals, 5):.0f}  "
          f"median=${np.median(finals):.0f}  p95=${np.percentile(finals, 95):.0f}")
    print(f"  Max Drawdown  p5={np.percentile(max_dds, 5):.1%}  "
          f"median={np.median(max_dds):.1%}  p95={np.percentile(max_dds, 95):.1%}")
    print(f"  P(Profitable) = {(finals > initial).mean():.1%}")

    print("=" * 72)

    # Сохраняем результаты
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_lower = label.lower()
    df.to_csv(f"bybit_results_{label_lower}_{ts_str}.csv", index=False)
    print(f"Saved: bybit_results_{label_lower}_{ts_str}.csv")

    # Equity curve
    eq_df = pd.DataFrame({
        "ts":     df["exit_ts"],
        "equity": initial + df["pnl"].cumsum(),
    })
    eq_df.to_csv(f"bybit_equity_{label_lower}_{ts_str}.csv", index=False)
    print(f"Saved: bybit_equity_{label_lower}_{ts_str}.csv")

    # Диагностика сигналов
    print()
    print("  SIGNAL DIAGNOSTICS")
    print(f"  Funding threshold SHORT: >= {cfg.strategy.FUNDING_SHORT_THRESHOLD:.4%}")
    print(f"  Funding threshold LONG:  <= {cfg.strategy.FUNDING_LONG_THRESHOLD:.4%}")
    print(f"  RSI SHORT: >= {cfg.strategy.RSI_SHORT}")
    print(f"  RSI LONG:  <= {cfg.strategy.RSI_LONG}")


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
def run_tests() -> bool:
    print("\n=== UNIT TESTS ===")
    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} {detail}")

    rc = RiskConfig()
    ec = ExchangeConfig()

    # Test 1: Long win
    port1 = Portfolio(1000, rc, ec)
    pos1  = port1.open_position(datetime.now(), "BTC/USDT:USDT", 1, 100.0, 2.0, 0.0001)
    if pos1:
        t1 = port1.close_position("BTC/USDT:USDT", datetime.now(), 110.0, ExitType.TAKE_PROFIT, 4)
        check("Long win PnL > 0", t1 and t1.pnl > 0, f"pnl={t1.pnl if t1 else 'N/A'}")
    else:
        check("Long win PnL > 0", False, "position not opened")

    # Test 2: Short win
    port2 = Portfolio(1000, rc, ec)
    pos2  = port2.open_position(datetime.now(), "ETH/USDT:USDT", -1, 100.0, 2.0, 0.001)
    if pos2:
        t2 = port2.close_position("ETH/USDT:USDT", datetime.now(), 90.0, ExitType.TAKE_PROFIT, 4)
        check("Short win PnL > 0", t2 and t2.pnl > 0, f"pnl={t2.pnl if t2 else 'N/A'}")
    else:
        check("Short win PnL > 0", False, "position not opened")

    # Test 3: Circuit breaker на DD
    port3 = Portfolio(1000, RiskConfig(CB_MAX_DD=0.10), ec)
    port3.equity   = 880.0
    port3.drawdown = 0.12
    check("CB triggers on DD", port3.check_circuit_breaker(datetime.now()), "")

    # Test 4: Max concurrent positions
    port4 = Portfolio(1000, RiskConfig(MAX_CONCURRENT=2), ec)
    port4.open_position(datetime.now(), "BTC/USDT:USDT", 1, 100.0, 2.0, 0.0)
    port4.open_position(datetime.now(), "ETH/USDT:USDT", 1, 100.0, 2.0, 0.0)
    pos_overflow = port4.open_position(datetime.now(), "SOL/USDT:USDT", 1, 100.0, 2.0, 0.0)
    check("Max concurrent respected", pos_overflow is None, "")

    # Test 5: Funding PnL для шорта
    port5 = Portfolio(1000, rc, ec)
    pos5  = port5.open_position(datetime.now(), "BTC/USDT:USDT", -1, 100.0, 2.0, 0.001)
    if pos5:
        ts_now = datetime.now()
        funding_series = pd.Series(
            [0.001, 0.001],
            index=pd.DatetimeIndex([ts_now - timedelta(hours=2), ts_now - timedelta(hours=1)])
        )
        t5 = port5.close_position("BTC/USDT:USDT", ts_now, 100.0, ExitType.TIMEOUT, 8, funding_series)
        check("Short collects positive funding", t5 and t5.funding_pnl > 0,
              f"funding_pnl={t5.funding_pnl if t5 else 'N/A'}")
    else:
        check("Short collects positive funding", False, "position not opened")

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BYBIT FUTURES v1.0")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--oos",  action="store_true", help="Show OOS results")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  BYBIT FUTURES v1.0")
    print("  Extreme Funding Rate Strategy | Top-10 Pairs | 2-3x Leverage")
    print("=" * 72)
    print()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    try:
        loader = DataLoader()
        engine = BacktestEngine(cfg, data_loader=loader)

        portfolio, trades = engine.run()
        analyze_results(portfolio, trades, label="FULL")

        if args.oos:
            engine_oos       = BacktestEngine(cfg, data_loader=loader)
            port_oos, tr_oos = engine_oos.run(oos_only=True)
            analyze_results(port_oos, tr_oos, label="OOS")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
