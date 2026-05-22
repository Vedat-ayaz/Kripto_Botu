"""
Walk-forward parametre adaptasyonu.

Literatür kaynakları:
- Robert Pardo, "The Evaluation and Optimization of Trading Strategies" (2nd ed.)
  → Walk-forward optimization yönteminin tanımı ve uygulaması
- Marcos López de Prado, "Advances in Financial Machine Learning" (2018)
  → Overfitting önleme, combinatorial purged cross-validation
- Andreas Clenow, "Following the Trend" (2013)
  → Parametre stabilitesi, robustness testi

Fitness metriği:
  Calmar Ratio = Yıllıklandırılmış Getiri / Max Drawdown
  Sharpe ile blended: %60 Calmar + %40 Sharpe (daha sağlam kriter)
  Calmar hesaplanamıyorsa (sıfır DD) sadece Sharpe kullanılır.

Yaklaşım:
  Her N trade sonunda (varsayılan 20), son M mumda mini-backtest.
  Yeni fitness eskisinden min_improvement kadar iyiyse parametreleri güncelle.
  Overfitting riskini azaltmak için parametre değişimini PARAM_BOUNDS ile sınırla.
  İşlem arka planda (thread) yapılır, bot'u bloklamaz.
"""

import threading
import logging
import copy
from typing import Optional, Callable
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Optimize edilecek parametre grid'i
PARAM_GRID = {
    "adx_threshold": [15.0, 20.0, 25.0, 30.0],
    "rsi_lower":     [40.0, 45.0, 50.0],
    "rsi_upper":     [65.0, 70.0, 75.0],
    "atr_stop_multiplier": [1.5, 2.0, 2.5],
}

# Parametre sınırları — aşırı değerleri önler (Pardo'nun "parameter stability" testi)
PARAM_BOUNDS = {
    "adx_threshold":       (10.0, 40.0),
    "rsi_lower":           (35.0, 55.0),
    "rsi_upper":           (60.0, 80.0),
    "atr_stop_multiplier": (1.0, 3.5),
}


def _mini_fitness(
    df_ind: pd.DataFrame,
    adx_threshold: float,
    rsi_lower: float,
    rsi_upper: float,
    atr_stop_multiplier: float,
    min_atr_ratio: float = 0.002,
    trailing_mult: float = 2.5,
    max_position_pct: float = 0.20,
    commission: float = 0.001,
) -> float:
    """
    Verilen parametreler için hızlı mini-backtest yapar.

    Fitness = 0.6 × Calmar + 0.4 × Sharpe  (blended robust metric)
    - Calmar = Yıllıklandırılmış Getiri / Max Drawdown
    - Sharpe = Mean(returns) / Std(returns) × √(252×24)
    - Calmar hesaplanamıyorsa (sıfır DD → risk yok) saf Sharpe döner.

    Gerçek Backtester yerine hafif vektörel simülasyon kullanır.
    """
    import math

    ema_fast_col = "ema_50"
    ema_slow_col = "ema_200"

    required = {ema_fast_col, ema_slow_col, "rsi", "atr", "adx", "volume_sma", "atr_ratio", "close"}
    if not required.issubset(df_ind.columns):
        return 0.0

    df = df_ind.dropna().copy()
    if len(df) < 50:
        return 0.0

    capital = 1000.0  # normalize başlangıç
    equity = [capital]
    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    trailing_stop = 0.0
    position_size = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row["close"]
        atr = row["atr"]

        if in_position:
            # Trailing stop güncelle
            new_trail = price - atr * trailing_mult
            if new_trail > trailing_stop:
                trailing_stop = new_trail

            # Çıkış kontrolü
            exit_triggered = (
                price <= stop_price or
                price <= trailing_stop or
                price < row[ema_fast_col]
            )
            if exit_triggered:
                pnl = (price - entry_price) * position_size
                pnl -= price * position_size * commission
                capital += pnl
                in_position = False

        else:
            # Giriş koşulları
            conditions = (
                row["close"] > row[ema_slow_col] and
                row[ema_fast_col] > row[ema_slow_col] and
                row["adx"] > adx_threshold and
                rsi_lower <= row["rsi"] <= rsi_upper and
                row["volume"] > row["volume_sma"] and
                row["atr_ratio"] > min_atr_ratio
            )
            if conditions:
                stop = price - atr * atr_stop_multiplier
                risk_per_unit = abs(price - stop)
                if risk_per_unit > 0:
                    raw_size = (capital * 0.01) / risk_per_unit
                    max_size = (capital * max_position_pct) / price
                    position_size = min(raw_size, max_size)
                    cost = position_size * price * commission
                    capital -= cost
                    entry_price = price
                    stop_price = stop
                    trailing_stop = price - atr * trailing_mult
                    in_position = True

        equity.append(capital)

    if len(equity) < 10:
        return 0.0

    arr = np.array(equity)
    returns = np.diff(arr) / arr[:-1]

    # ── Sharpe ───────────────────────────────────────────────────
    std = returns.std()
    sharpe = float((returns.mean() / std) * math.sqrt(252 * 24)) if std > 0 else 0.0

    # ── Calmar ───────────────────────────────────────────────────
    # Max drawdown: peak-to-trough
    peak = arr[0]
    max_dd = 0.0
    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    calmar = 0.0
    if max_dd > 0.0001:  # sıfır bölme koruması
        # Yıllıklandırılmış getiri: (son_sermaye / ilk_sermaye)^(8760/bar_sayısı) - 1
        bar_hours = len(arr)
        ann_return = (arr[-1] / arr[0]) ** (8760 / max(bar_hours, 1)) - 1
        calmar = float(ann_return / max_dd)

    # ── Blended fitness ──────────────────────────────────────────
    if max_dd > 0.0001:
        fitness = 0.6 * np.clip(calmar, -5, 20) + 0.4 * np.clip(sharpe, -5, 10)
    else:
        # Drawdown yoksa (risk almadı gibi) → sadece Sharpe, penalize edilmiş
        fitness = 0.4 * sharpe

    return float(fitness)


# Geriye dönük uyumluluk için alias
_mini_sharpe = _mini_fitness


class ParameterOptimizer:
    """
    Walk-forward parametre adaptasyonu.
    Her adaptation_window trade'den sonra arka planda grid search yapar.
    """

    def __init__(
        self,
        initial_params: dict,
        adaptation_window: int = 20,
        lookback_bars: int = 500,
        min_improvement: float = 0.10,
        on_params_updated: Optional[Callable] = None,
    ):
        self.params = copy.deepcopy(initial_params)
        self.adaptation_window = adaptation_window
        self.lookback_bars = lookback_bars
        self.min_improvement = min_improvement
        self.on_params_updated = on_params_updated  # callback: (old, new, old_s, new_s) -> None

        self._trade_count = 0
        self._adapting = False
        self._lock = threading.Lock()

        # Her sembol için son df_ind kaydı
        self._last_df: dict[str, pd.DataFrame] = {}

    def update_candle_data(self, symbol: str, df_ind: pd.DataFrame) -> None:
        """Her tick'te en son göstergeli DataFrame'i kaydeder."""
        with self._lock:
            self._last_df[symbol] = df_ind.tail(self.lookback_bars).copy()

    def record_trade(self, symbol: str) -> None:
        """Bir işlem kapandığında çağrılır."""
        self._trade_count += 1
        logger.debug(f"[Optimizer] İşlem #{self._trade_count} kaydedildi ({symbol})")

        if self._trade_count % self.adaptation_window == 0 and not self._adapting:
            self._start_adaptation(symbol)

    def _start_adaptation(self, symbol: str) -> None:
        """Arka planda adaptasyon başlatır."""
        df = self._last_df.get(symbol)
        if df is None or len(df) < 100:
            logger.info("[Optimizer] Adaptasyon için yeterli veri yok, atlandı.")
            return

        self._adapting = True
        thread = threading.Thread(
            target=self._run_grid_search,
            args=(symbol, df.copy()),
            daemon=True,
        )
        thread.start()
        logger.info(f"[Optimizer] {symbol} için walk-forward adaptasyon başladı "
                    f"({len(df)} bar, {self._trade_count} işlem).")

    def _run_grid_search(self, symbol: str, df_ind: pd.DataFrame) -> None:
        """
        Parametre grid'ini tarar, en iyi Sharpe'ı bulur.
        Pardo'nun "out-of-sample robustness" önerisiyle:
        verinin %70'i in-sample, %30'u validasyon.
        """
        try:
            split = int(len(df_ind) * 0.7)
            df_train = df_ind.iloc[:split]
            df_valid = df_ind.iloc[split:]

            # Mevcut parametrelerle baseline fitness (validasyon üzerinde)
            baseline_fitness = _mini_fitness(
                df_valid,
                adx_threshold=self.params.get("adx_threshold", 20.0),
                rsi_lower=self.params.get("rsi_lower", 45.0),
                rsi_upper=self.params.get("rsi_upper", 70.0),
                atr_stop_multiplier=self.params.get("atr_stop_multiplier", 2.0),
                min_atr_ratio=self.params.get("min_atr_ratio", 0.002),
                trailing_mult=self.params.get("trailing_stop_atr_multiplier", 2.5),
            )

            best_train_fitness = -999.0
            best_valid_fitness = baseline_fitness
            best_params = copy.deepcopy(self.params)

            # Grid search — train üzerinde en iyiyi bul
            for adx in PARAM_GRID["adx_threshold"]:
                for rsi_l in PARAM_GRID["rsi_lower"]:
                    for rsi_u in PARAM_GRID["rsi_upper"]:
                        if rsi_u <= rsi_l + 10:
                            continue  # RSI bandı çok dar
                        for atr_mult in PARAM_GRID["atr_stop_multiplier"]:
                            # Parametre sınır kontrolü (Pardo stability test)
                            if not self._within_bounds(adx, rsi_l, rsi_u, atr_mult):
                                continue

                            train_f = _mini_fitness(
                                df_train, adx, rsi_l, rsi_u, atr_mult,
                                min_atr_ratio=self.params.get("min_atr_ratio", 0.002),
                                trailing_mult=self.params.get("trailing_stop_atr_multiplier", 2.5),
                            )
                            if train_f > best_train_fitness:
                                best_train_fitness = train_f
                                best_params_candidate = {
                                    "adx_threshold": adx,
                                    "rsi_lower": rsi_l,
                                    "rsi_upper": rsi_u,
                                    "atr_stop_multiplier": atr_mult,
                                }
                                # Validasyon üzerinde kontrol et (out-of-sample test)
                                valid_f = _mini_fitness(
                                    df_valid, adx, rsi_l, rsi_u, atr_mult,
                                    min_atr_ratio=self.params.get("min_atr_ratio", 0.002),
                                    trailing_mult=self.params.get("trailing_stop_atr_multiplier", 2.5),
                                )
                                if valid_f > best_valid_fitness + self.min_improvement:
                                    best_valid_fitness = valid_f
                                    best_params = {**self.params, **best_params_candidate}

            # Güncelleme gerekiyor mu?
            improvement = best_valid_fitness - baseline_fitness
            if improvement >= self.min_improvement and best_params != self.params:
                old_params = copy.deepcopy(self.params)
                with self._lock:
                    self.params.update(best_params)

                logger.info(
                    f"[Optimizer] {symbol} parametreleri güncellendi! "
                    f"Fitness (0.6×Calmar+0.4×Sharpe): {baseline_fitness:.3f} → {best_valid_fitness:.3f} "
                    f"(+{improvement:.3f})"
                )
                logger.info(f"  Eski: {old_params}")
                logger.info(f"  Yeni: {self.params}")

                if self.on_params_updated:
                    self.on_params_updated(
                        old_params, copy.deepcopy(self.params),
                        baseline_fitness, best_valid_fitness
                    )
            else:
                logger.info(
                    f"[Optimizer] {symbol}: Mevcut parametreler optimal "
                    f"(baseline fitness: {baseline_fitness:.3f}, "
                    f"en iyi aday: {best_valid_fitness:.3f}). Değişiklik yok."
                )
        except Exception as e:
            logger.error(f"[Optimizer] Grid search hatası: {e}")
        finally:
            self._adapting = False

    def _within_bounds(self, adx: float, rsi_l: float, rsi_u: float, atr_mult: float) -> bool:
        """Parametrelerin izin verilen sınırlar içinde olup olmadığını kontrol eder."""
        return (
            PARAM_BOUNDS["adx_threshold"][0] <= adx <= PARAM_BOUNDS["adx_threshold"][1] and
            PARAM_BOUNDS["rsi_lower"][0] <= rsi_l <= PARAM_BOUNDS["rsi_lower"][1] and
            PARAM_BOUNDS["rsi_upper"][0] <= rsi_u <= PARAM_BOUNDS["rsi_upper"][1] and
            PARAM_BOUNDS["atr_stop_multiplier"][0] <= atr_mult <= PARAM_BOUNDS["atr_stop_multiplier"][1]
        )

    @property
    def current_params(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.params)

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def is_adapting(self) -> bool:
        return self._adapting
