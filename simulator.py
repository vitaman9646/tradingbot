"""
simulator.py — Real OHLCV simulation + Autotune
Модуль к collector.py. Запускай после сбора данных.

Структура:
  1. OHLCVSimulator   — честный бэктест на реальных данных
  2. FilterAutotune   — grid search по параметрам фильтров
  3. WalkForward      — защита от overfitting
  4. main()           — запуск всего пайплайна

Зависимости:
  pip install ccxt numpy pandas --break-system-packages
"""

import ccxt
import numpy as np
import pandas as pd
import sqlite3
import time
import json
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Trade:
    timestamp: float
    symbol: str
    direction: str          # 'long' | 'short'
    entry_price: float
    fomo_price: float
    spread: float
    atr: float
    vr: float               # volatility ratio ATR(1m)/ATR(5m)
    sl: float               # as fraction, e.g. 0.015
    tp: float
    is_win: bool
    exit_price: float
    pnl: float              # fraction, e.g. 0.012
    exit_reason: str        # 'TP' | 'SL' | 'TIMEOUT'
    filter_skip: bool = False
    skip_reason: str = ''


@dataclass
class SimResult:
    n_trades: int
    n_filtered: int
    win_rate: float
    sharpe: float
    profit_factor: float
    max_drawdown: float
    total_pnl: float
    avg_pnl: float
    opportunity_cost_pct: float   # % хороших сделок пропущенных фильтрами
    params: dict = field(default_factory=dict)


# ============================================================
# 1. OHLCV SIMULATOR — без look-ahead bias
# ============================================================

class OHLCVSimulator:
    """
    Честная симуляция на реальных OHLCV данных.

    Принципы:
    - Решение принимается ТОЛЬКО на данных до момента i
    - pnl считается на барах ПОСЛЕ i (не включая i)
    - FOMO лаг = lag_bars баров назад (реальное поведение)
    - Все параметры (ATR, VR) считаются из истории, не из будущего
    """

    def __init__(
        self,
        symbol: str = 'SOL/USDT',
        lag_bars: int = 3,           # FOMO лаг в барах (1 бар = 1 мин)
        atr_period: int = 14,
        vr_short: int = 3,           # ATR(Nm) для VR числителя
        vr_long: int = 15,           # ATR(Mm) для VR знаменателя
        spread_threshold: float = 0.008,   # минимальный spread для входа
        fees: float = 0.001,         # 0.1% per side
    ):
        self.symbol = symbol
        self.lag_bars = lag_bars
        self.atr_period = atr_period
        self.vr_short = vr_short
        self.vr_long = vr_long
        self.spread_threshold = spread_threshold
        self.fees = fees

    def load_ohlcv(self, source: str = 'ccxt', db_path: Optional[str] = None, limit: int = 500):
        """
        Загружает данные из ccxt или SQLite.
        source: 'ccxt' | 'sqlite'
        """
        if source == 'ccxt':
            exchange = ccxt.bybit()
            raw = exchange.fetch_ohlcv(self.symbol, '1m', limit=limit)
            # [[timestamp, open, high, low, close, volume], ...]
            self.ohlcv = raw
            print(f"Loaded {len(raw)} bars from Bybit ({self.symbol})")

        elif source == 'sqlite' and db_path:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(
                "SELECT unix_ts, last FROM price_snapshots "
                "WHERE symbol=? AND source='bybit' ORDER BY unix_ts",
                conn, params=(self.symbol,)
            )
            conn.close()
            # Агрегируем тики в 1-минутные бары
            df['minute'] = (df['unix_ts'] // 60).astype(int)
            ohlcv = df.groupby('minute')['last'].agg(
                open='first', high='max', low='min', close='last'
            ).reset_index()
            ohlcv['timestamp'] = ohlcv['minute'] * 60 * 1000
            ohlcv['volume'] = 0  # нет данных объёма из price_snapshots
            self.ohlcv = ohlcv[['timestamp','open','high','low','close','volume']].values.tolist()
            print(f"Loaded {len(self.ohlcv)} bars from SQLite")

        return self

    def _calc_atr(self, bars: list, period: int) -> float:
        """ATR как среднее (high - low) / close, нормализованное."""
        if len(bars) < period:
            return 0.0
        recent = bars[-period:]
        trs = [abs(b[2] - b[3]) / b[4] for b in recent if b[4] > 0]  # (H-L)/Close
        return float(np.mean(trs)) if trs else 0.0

    def _calc_vr(self, bars: list) -> float:
        """VR = ATR(short) / ATR(long). >1.4 хаос, <0.6 мёртвый."""
        atr_s = self._calc_atr(bars, self.vr_short)
        atr_l = self._calc_atr(bars, self.vr_long)
        return atr_s / atr_l if atr_l > 0 else 1.0

    def _simulate_exit(
        self,
        bars_after: list,
        entry_price: float,
        direction: str,
        sl: float,
        tp: float,
        max_bars: int = 10,
    ) -> tuple[bool, float, str, float]:
        """
        Симулирует выход по TP/SL/TIMEOUT на барах ПОСЛЕ входа.
        Возвращает (is_win, exit_price, reason, pnl_fraction).
        Без look-ahead: используем только бары после момента входа.
        """
        for bar in bars_after[:max_bars]:
            high, low, close = bar[2], bar[3], bar[4]

            if direction == 'long':
                if high >= entry_price * (1 + tp):
                    exit_p = entry_price * (1 + tp)
                    return True, exit_p, 'TP', tp - self.fees * 2
                if low <= entry_price * (1 - sl):
                    exit_p = entry_price * (1 - sl)
                    return False, exit_p, 'SL', -sl - self.fees * 2
            else:  # short
                if low <= entry_price * (1 - tp):
                    exit_p = entry_price * (1 - tp)
                    return True, exit_p, 'TP', tp - self.fees * 2
                if high >= entry_price * (1 + sl):
                    exit_p = entry_price * (1 + sl)
                    return False, exit_p, 'SL', -sl - self.fees * 2

        # TIMEOUT: выход по последнему close
        last_close = bars_after[min(max_bars - 1, len(bars_after) - 1)][4]
        if direction == 'long':
            pnl = (last_close - entry_price) / entry_price - self.fees * 2
        else:
            pnl = (entry_price - last_close) / entry_price - self.fees * 2
        return pnl > 0, last_close, 'TIMEOUT', pnl

    def run(self, filters: dict, atr_mult_sl: float = 1.5, atr_mult_tp: float = 2.25) -> tuple[list[Trade], SimResult]:
        """
        Запускает симуляцию с заданными фильтрами.

        filters: словарь порогов, например:
          {
            'min_spread': 0.008,
            'min_vol_ratio': 1.5,   # volume ratio
            'vr_low': 0.6,          # VR нижний порог
            'vr_high': 1.4,         # VR верхний порог
            'trend_min_diff': 0.002, # EMA8-EMA21 минимум
          }
        """
        ohlcv = self.ohlcv
        trades = []
        min_bars = max(self.atr_period, self.vr_long) + self.lag_bars

        close_prices = [b[4] for b in ohlcv]

        # EMA helper
        def ema(prices, period):
            k = 2 / (period + 1)
            val = prices[0]
            for p in prices[1:]:
                val = p * k + val * (1 - k)
            return val

        for i in range(min_bars, len(ohlcv) - 11):  # -11 чтобы было место для exit
            bars_history = ohlcv[:i]     # только прошлое
            bars_future  = ohlcv[i+1:]  # только будущее (для exit)

            current_bar  = ohlcv[i]
            real_price   = current_bar[4]
            fomo_bar     = ohlcv[i - self.lag_bars]
            fomo_price   = fomo_bar[4]

            spread = (real_price - fomo_price) / real_price

            # Пропускаем если спред ниже порога
            if abs(spread) < filters.get('min_spread', self.spread_threshold):
                continue

            direction = 'long' if spread > 0 else 'short'

            # ATR из истории
            atr = self._calc_atr(bars_history, self.atr_period)
            if atr == 0:
                continue

            # VR из истории
            vr = self._calc_vr(bars_history)

            # Adaptive ATR multiplier: тихий рынок → шире стоп
            adaptive_sl_mult = atr_mult_sl * (1.8 / 1.5 if vr < 0.8 else 1.2 / 1.5 if vr > 1.2 else 1.0)
            sl = min(0.025, max(0.008, atr * adaptive_sl_mult))
            tp = sl * (atr_mult_tp / atr_mult_sl)

            # Trend filter: EMA8 > EMA21 (только для long)
            trend_diff = 0.0
            if 'trend_min_diff' in filters:
                hist_closes = close_prices[max(0, i-30):i]
                if len(hist_closes) >= 21:
                    ema8  = ema(hist_closes[-21:], 8)
                    ema21 = ema(hist_closes[-21:], 21)
                    trend_diff = (ema8 - ema21) / ema21

            # ---- ПРИМЕНЕНИЕ ФИЛЬТРОВ ----
            skip = False
            skip_reason = ''

            # VR filter
            vr_low  = filters.get('vr_low',  0.6)
            vr_high = filters.get('vr_high', 1.4)
            if vr < vr_low:
                skip = True; skip_reason = 'VR_DEAD'
            elif vr > vr_high:
                skip = True; skip_reason = 'VR_CHAOS'

            # Trend filter
            if not skip and 'trend_min_diff' in filters:
                min_diff = filters['trend_min_diff']
                if direction == 'long'  and trend_diff < min_diff:
                    skip = True; skip_reason = 'NO_BULL_TREND'
                if direction == 'short' and trend_diff > -min_diff:
                    skip = True; skip_reason = 'NO_BEAR_TREND'

            # Pullback filter: последняя свеча против тренда
            if not skip and filters.get('pullback_filter', False):
                last_close = ohlcv[i-1][4]
                hist_closes = close_prices[max(0, i-10):i]
                if len(hist_closes) >= 8:
                    ema8 = ema(hist_closes, 8)
                    if direction == 'long'  and last_close < ema8:
                        skip = True; skip_reason = 'PULLBACK'
                    if direction == 'short' and last_close > ema8:
                        skip = True; skip_reason = 'PULLBACK'

            # Симулируем выход (даже для пропущенных — для opportunity cost)
            is_win, exit_price, exit_reason, pnl = self._simulate_exit(
                bars_future, real_price, direction, sl, tp
            )

            trades.append(Trade(
                timestamp   = current_bar[0] / 1000,
                symbol      = self.symbol,
                direction   = direction,
                entry_price = real_price,
                fomo_price  = fomo_price,
                spread      = spread,
                atr         = atr,
                vr          = vr,
                sl          = sl,
                tp          = tp,
                is_win      = is_win,
                exit_price  = exit_price,
                pnl         = pnl if not skip else 0.0,
                exit_reason = exit_reason,
                filter_skip = skip,
                skip_reason = skip_reason,
            ))

        return trades, self._calc_result(trades, filters)

    def _calc_result(self, trades: list[Trade], params: dict) -> SimResult:
        executed = [t for t in trades if not t.filter_skip]
        skipped  = [t for t in trades if t.filter_skip]

        if not executed:
            return SimResult(0, len(skipped), 0, 0, 0, 0, 0, 0, 0, params)

        wins   = [t for t in executed if t.is_win]
        losses = [t for t in executed if not t.is_win]
        pnls   = [t.pnl for t in executed]

        win_rate = len(wins) / len(executed)
        avg_pnl  = float(np.mean(pnls))
        std_pnl  = float(np.std(pnls)) if len(pnls) > 1 else 0.01

        # Sharpe annualized — масштабируем по количеству трейдов в день
        # Не умножаем на фиксированный 252*50 (это для HFT)
        # Для нашей стратегии (~10-20 трейдов/день): sqrt(252 * avg_daily_trades)
        avg_daily = max(1, len(executed) / max(1, (len(executed) / 15)))  # estimate
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

        # PF через tp/sl значения (как в v4)
        pf_wins   = sum(t.tp for t in wins)
        pf_losses = sum(t.sl for t in losses)
        pf = pf_wins / pf_losses if pf_losses > 0 else 0

        # Max drawdown
        equity = [1000.0]
        for t in executed:
            equity.append(equity[-1] * (1 + t.pnl * 0.05))
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        # Opportunity cost: сколько пропущенных сделок были бы прибыльными
        skipped_wins = sum(1 for t in skipped if t.is_win)
        opp_cost = skipped_wins / len(skipped) * 100 if skipped else 0

        return SimResult(
            n_trades           = len(executed),
            n_filtered         = len(skipped),
            win_rate           = win_rate,
            sharpe             = float(sharpe),
            profit_factor      = float(pf),
            max_drawdown       = float(max_dd),
            total_pnl          = float(sum(pnls)),
            avg_pnl            = avg_pnl,
            opportunity_cost_pct = opp_cost,
            params             = params,
        )


# ============================================================
# 2. FILTER AUTOTUNE — grid search с защитой от overfitting
# ============================================================

class FilterAutotune:
    """
    Автоматический поиск оптимальных параметров фильтров.

    Что делает:
    1. Grid search по пространству параметров
    2. Walk-forward validation (in-sample / out-of-sample split)
    3. Выбирает параметры по Sharpe, не по WR (устойчивее)
    4. Логирует все результаты для анализа

    Почему НЕ ML:
    - Слишком мало данных для обучения нейросети
    - Grid search на 4-6 параметрах = ~200-500 комбинаций
    - Интерпретируемо: видишь что и почему выбрано

    Когда переходить на ML:
    - 10,000+ трейдов в истории
    - Стабильный edge подтверждён на out-of-sample
    - Sharpe > 0.8 на реальных данных
    """

    PARAM_GRID = {
        'min_spread':       [0.006, 0.008, 0.010, 0.012],
        'vr_low':           [0.5,   0.6,   0.7],
        'vr_high':          [1.3,   1.4,   1.5],
        'trend_min_diff':   [0.001, 0.002, 0.003],
        'pullback_filter':  [False, True],
    }

    def __init__(self, simulator: OHLCVSimulator, is_ratio: float = 0.7):
        """
        is_ratio: доля данных для in-sample обучения (0.7 = 70%)
        """
        self.sim = simulator
        self.is_ratio = is_ratio
        self.results: list[SimResult] = []

    def _split_data(self):
        """Разбивает ohlcv на in-sample и out-of-sample."""
        n = len(self.sim.ohlcv)
        split = int(n * self.is_ratio)
        return self.sim.ohlcv[:split], self.sim.ohlcv[split:]

    def run(self, objective: str = 'sharpe') -> dict:
        """
        Запускает grid search.
        objective: 'sharpe' | 'profit_factor' | 'win_rate'

        Возвращает лучшие параметры и результаты.
        """
        keys   = list(self.PARAM_GRID.keys())
        values = list(self.PARAM_GRID.values())
        combos = list(itertools.product(*values))

        is_ohlcv, oos_ohlcv = self._split_data()

        print(f"\n{'='*60}")
        print(f"AUTOTUNE: {len(combos)} parameter combinations")
        print(f"In-sample:      {len(is_ohlcv)} bars ({self.is_ratio*100:.0f}%)")
        print(f"Out-of-sample:  {len(oos_ohlcv)} bars ({(1-self.is_ratio)*100:.0f}%)")
        print(f"Objective:      {objective}")
        print(f"{'='*60}\n")

        best_is_score  = -np.inf
        best_params    = None
        all_results    = []

        # In-sample search
        self.sim.ohlcv = is_ohlcv
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))

            try:
                _, result = self.sim.run(params)
            except Exception as e:
                continue

            # Фильтр минимального количества трейдов (статистическая значимость)
            if result.n_trades < 30:
                continue

            score = getattr(result, objective)
            all_results.append((score, params, result))

            if score > best_is_score:
                best_is_score = score
                best_params   = params

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(combos)} | Best {objective}: {best_is_score:.3f}")

        if not best_params:
            print("No valid parameter combination found. Check data quality.")
            return {}

        # Out-of-sample validation с лучшими параметрами
        self.sim.ohlcv = oos_ohlcv
        _, oos_result = self.sim.run(best_params)

        # Восстанавливаем полный датасет
        self.sim.ohlcv = is_ohlcv + oos_ohlcv

        # Топ-10 in-sample результатов
        top10 = sorted(all_results, key=lambda x: x[0], reverse=True)[:10]

        self._print_results(best_params, best_is_score, oos_result, top10, objective)
        self._check_overfitting(best_is_score, getattr(oos_result, objective))

        return {
            'best_params':    best_params,
            'is_score':       best_is_score,
            'oos_result':     oos_result,
            'is_oos_ratio':   getattr(oos_result, objective) / best_is_score if best_is_score > 0 else 0,
            'top10':          [(s, p) for s, p, _ in top10],        }

    def _print_results(self, params, is_score, oos_result, top10, objective):
        print(f"\n{'='*60}")
        print(f"AUTOTUNE RESULTS")
        print(f"{'='*60}")
        print(f"\nBest parameters (in-sample):")
        for k, v in params.items():
            print(f"  {k:25s} = {v}")

        print(f"\nIn-sample  {objective:15s}: {is_score:.4f}")
        print(f"Out-sample {objective:15s}: {getattr(oos_result, objective):.4f}")
        print(f"\nOut-of-sample full metrics:")
        print(f"  Trades:          {oos_result.n_trades}")
        print(f"  Win rate:        {oos_result.win_rate*100:.1f}%")
        print(f"  Sharpe:          {oos_result.sharpe:.3f}")
        print(f"  Profit factor:   {oos_result.profit_factor:.3f}")
        print(f"  Max drawdown:    {oos_result.max_drawdown*100:.1f}%")
        print(f"  Opp. cost:       {oos_result.opportunity_cost_pct:.1f}% of filtered were wins")

        print(f"\nTop-5 in-sample combos:")
        for rank, (score, p, _) in enumerate(top10[:5], 1):
            print(f"  #{rank}: {objective}={score:.4f} | {p}")

    def _check_overfitting(self, is_score: float, oos_score: float):
        """
        Простая проверка на overfitting.
        Если OOS значительно хуже IS — параметры переобучены.
        """
        if is_score <= 0:
            return
        ratio = oos_score / is_score
        print(f"\nOverfitting check: OOS/IS ratio = {ratio:.2f}")
        if ratio > 0.75:
            print("  ✅ GOOD — OOS близко к IS, параметры устойчивы")
        elif ratio > 0.5:
            print("  ⚠️  MARGINAL — умеренное расхождение, используй с осторожностью")
        else:
            print("  ❌ OVERFIT — OOS намного хуже IS, параметры переобучены на шуме")
            print("     Рекомендация: расширь диапазон param_grid, уменьши is_ratio до 0.6")


# ============================================================
# 3. WALK-FORWARD VALIDATION
# ============================================================

class WalkForward:
    """
    Walk-forward тест: скользящее окно по времени.
    Более строгая защита от overfitting чем простой IS/OOS split.

    Пример с 3 фолдами:
    Fold 1: IS [0..60%] → OOS [60..73%]
    Fold 2: IS [0..73%] → OOS [73..86%]
    Fold 3: IS [0..86%] → OOS [86..100%]

    Если стратегия работает во всех фолдах — это реальный edge.
    """

    def __init__(self, simulator: OHLCVSimulator, autotune: FilterAutotune, n_folds: int = 3):
        self.sim = simulator
        self.autotune = autotune
        self.n_folds = n_folds

    def run(self, objective: str = 'sharpe') -> list[dict]:
        """Запускает walk-forward по n_folds фолдам."""
        n = len(self.sim.ohlcv)
        results = []

        # IS всегда начинается с начала, OOS скользит вперёд
        fold_size = int(n * (1 - self.autotune.is_ratio) / self.n_folds)
        is_end    = int(n * self.autotune.is_ratio)

        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION ({self.n_folds} folds)")
        print(f"{'='*60}")

        for fold in range(self.n_folds):
            oos_start = is_end + fold * fold_size
            oos_end   = min(oos_start + fold_size, n)

            if oos_start >= n:
                break

            is_data  = self.sim.ohlcv[:is_end + fold * fold_size]
            oos_data = self.sim.ohlcv[oos_start:oos_end]

            # Autotune на IS данных этого фолда
            self.sim.ohlcv = is_data
            tune_result = self.autotune.run(objective)

            if not tune_result or not tune_result.get('best_params'):
                continue

            # Валидация на OOS данных
            self.sim.ohlcv = oos_data
            _, oos_result = self.sim.run(tune_result['best_params'])

            fold_result = {
                'fold':       fold + 1,
                'is_bars':    len(is_data),
                'oos_bars':   len(oos_data),
                'best_params': tune_result['best_params'],
                'is_score':   tune_result['is_score'],
                'oos_score':  getattr(oos_result, objective),
                'oos_wr':     oos_result.win_rate,
                'oos_sharpe': oos_result.sharpe,
                'oos_pf':     oos_result.profit_factor,
                'oos_dd':     oos_result.max_drawdown,
            }
            results.append(fold_result)

            print(f"\nFold {fold+1}: IS {len(is_data)} bars | OOS {len(oos_data)} bars")
            print(f"  IS  {objective}: {tune_result['is_score']:.4f}")
            print(f"  OOS {objective}: {getattr(oos_result, objective):.4f}")
            print(f"  OOS WR: {oos_result.win_rate*100:.1f}% | Sharpe: {oos_result.sharpe:.3f}")

        # Восстанавливаем полный датасет
        self.sim.ohlcv = self.sim.ohlcv  # уже был изменён

        self._summary(results, objective)
        return results

    def _summary(self, results: list[dict], objective: str):
        if not results:
            print("No fold results.")
            return

        oos_scores = [r['oos_score'] for r in results]
        oos_wrs    = [r['oos_wr']    for r in results]
        oos_pfs    = [r['oos_pf']    for r in results]

        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        print(f"  Mean OOS {objective}:    {np.mean(oos_scores):.4f} ± {np.std(oos_scores):.4f}")
        print(f"  Mean OOS Win Rate:  {np.mean(oos_wrs)*100:.1f}%")
        print(f"  Mean OOS PF:        {np.mean(oos_pfs):.3f}")

        consistent = sum(1 for s in oos_scores if s > 0)
        print(f"\n  Consistent folds:  {consistent}/{len(results)}")

        if consistent == len(results):
            print("  ✅ Strategy shows edge in ALL folds — robust signal")
        elif consistent > len(results) // 2:
            print("  ⚠️  Edge in majority of folds — use caution, collect more data")
        else:
            print("  ❌ Edge not consistent — strategy may not work on this asset")


# ============================================================
# 4. MAIN — полный пайплайн
# ============================================================

def main():
    print("FOMO Momentum — Real Data Simulation + Autotune")
    print("=" * 60)

    # Шаг 1: Загрузка данных
    sim = OHLCVSimulator(symbol='SOL/USDT', lag_bars=3, fees=0.001)

    print("\nStep 1: Loading OHLCV data...")
    try:
        sim.load_ohlcv(source='ccxt', limit=500)
    except Exception as e:
        print(f"Network error: {e}")
        print("Generating synthetic data for demo...")
        _generate_synthetic_ohlcv(sim)

    # Шаг 2: Baseline (без фильтров)
    print("\nStep 2: Baseline simulation (no filters)...")
    baseline_trades, baseline_result = sim.run(filters={})
    print(f"  Trades:       {baseline_result.n_trades}")
    print(f"  Win rate:     {baseline_result.win_rate*100:.1f}%")
    print(f"  Sharpe:       {baseline_result.sharpe:.3f}")
    print(f"  PF:           {baseline_result.profit_factor:.3f}")
    print(f"  Max DD:       {baseline_result.max_drawdown*100:.1f}%")

    # Шаг 3: Autotune (grid search)
    print("\nStep 3: Running Autotune (grid search)...")
    autotune = FilterAutotune(sim, is_ratio=0.7)
    tune_result = autotune.run(objective='sharpe')

    if not tune_result:
        print("Autotune failed — insufficient data.")
        return

    # Шаг 4: Walk-forward validation
    print("\nStep 4: Walk-forward validation...")
    # Используем уже загруженные данные (не перезагружаем)
    sim.ohlcv = sim.ohlcv  # данные уже есть
    wf = WalkForward(sim, autotune, n_folds=3)
    wf_results = wf.run(objective='sharpe')

    # Шаг 5: GO/NO-GO решение
    print("\n" + "=" * 60)
    print("FINAL GO/NO-GO DECISION")
    print("=" * 60)

    oos = tune_result['oos_result']
    checks = {
        'Win rate > 50%':       oos.win_rate > 0.50,
        'Sharpe > 0.5':         oos.sharpe > 0.5,
        'Profit factor > 1.3':  oos.profit_factor > 1.3,
        'Max drawdown < 10%':   oos.max_drawdown < 0.10,
        'OOS/IS ratio > 0.75':  tune_result.get('is_oos_ratio', 0) > 0.75,
    }

    passed = sum(checks.values())
    for label, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {label}")

    print()
    if passed == len(checks):
        print("  → GO: Start paper trading with best_params")
        print(f"  → Best params: {tune_result['best_params']}")
    elif passed >= 3:
        print("  → MARGINAL: Continue paper trading, collect more data")
    else:
        print("  → NO-GO: Edge not confirmed on out-of-sample data")
        print("  → Consider: different asset, longer lag, more data")

    # Сохраняем результаты
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': sim.symbol,
        'baseline': {
            'win_rate': baseline_result.win_rate,
            'sharpe': baseline_result.sharpe,
            'pf': baseline_result.profit_factor,
        },
        'best_params': tune_result.get('best_params', {}),
        'oos_metrics': {
            'win_rate': oos.win_rate,
            'sharpe': oos.sharpe,
            'pf': oos.profit_factor,
            'max_dd': oos.max_drawdown,
        },
        'walk_forward': wf_results,
        'verdict': 'GO' if passed == len(checks) else 'MARGINAL' if passed >= 3 else 'NO-GO',
    }

    with open('simulation_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to simulation_results.json")


def _generate_synthetic_ohlcv(sim: OHLCVSimulator, n: int = 500):
    """Генерирует синтетические данные если нет сети. Только для демо."""
    import random
    price = 150.0
    bars = []
    ts = int(time.time() * 1000) - n * 60000
    for i in range(n):
        change = random.gauss(0, 0.008)
        open_p  = price
        close_p = price * (1 + change)
        high_p  = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.003)))
        low_p   = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.003)))
        volume  = random.uniform(1000, 10000)
        bars.append([ts + i * 60000, open_p, high_p, low_p, close_p, volume])
        price = close_p
    sim.ohlcv = bars
    print(f"  Generated {n} synthetic bars (demo mode)")


if __name__ == '__main__':
    main()
