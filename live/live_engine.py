"""
Kripto Bot — Gerçek Zamanlı Live Paper Trading Engine
======================================================
Her çağrıda SADECE son kapanmış barı işler:
  1. State yükle (bakiye, açık pozisyonlar, son işlenen bar)
  2. Son barı çek — daha önce işlendiyse çık (duplicate guard)
  3. Açık pozisyonlar: stop/trailing güncelle, çıkışları uygula
  4. Giriş sinyalleri: her coin için generate_signal() çağır
  5. State kaydet → JSON + SQLite

Kullanım (live_runner.py içinden çağrılır):
    engine = LiveEngine(mode="M4", ...)
    engine.tick()

Döngü:
    python live/live_runner.py --loop 900   # her 15 dakikada bir tick
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

# Proje kökünü import path'e ekle
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from crypto_portfolio_test import (
    fetch_ohlcv,
    prepare_indicators,
    make_strategy,
    SYMBOLS,
    UNIVERSE,
    PROFILES,
    BASELINE,
    SYMBOL_EXCHANGE,
    COMMISSION,
    SLIPPAGE,
    RISK_PER_TRADE,
    ATR_STOP_MULT,
    TRAILING_MULT,
    MAX_POSITION_PCT,
    MIN_ORDER_SIZE,
    _bars_per_day,
    _tf_to_minutes,
)
from strategy.signal import Side
from indicators.technical_indicators import TechnicalIndicators
from strategy.adaptive_regime import AdaptiveRegimeController, Regime

logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────

# Warmup gün sayısı: indikatörlerin oturması için çekilecek tarihsel veri
WARMUP_DAYS = {
    "1m":  2,    # 2 gün × 1440 bar = 2880 bar
    "15m": 7,    # 7 gün × 96 bar = 672 bar (EMA200 için yeterli)
    "1h":  14,   # 14 gün × 24 bar = 336 bar
}

# ── Pozisyon Veri Sınıfı ─────────────────────────────────────────────────────

@dataclass
class LivePosition:
    symbol: str
    entry_price: float
    stop_price: float
    trail_price: float
    size: float            # coin adedi
    cost: float            # nakit bloke ($)
    entry_date: str        # ISO timestamp
    entry_atr: float
    trailing_mult: float
    is_short: bool = False
    coin_own_bull: bool = False
    bars_held: int = 0
    min_hold_bars: int = 6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LivePosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def unrealized_pnl(self) -> float:
        return 0.0  # fiyat bilgisi dışarıdan gelir


# ── Ana Motor ─────────────────────────────────────────────────────────────────

class LiveEngine:
    """
    Tek bir modeli (M4, M5 veya M6) yöneten gerçek zamanlı paper trading motoru.

    Her tick() çağrısında:
      - Tüm coinlerin son verisi çekilir (son WARMUP_DAYS gün)
      - İndikatörler hesaplanır
      - Açık pozisyonlarda stop/trailing kontrolü yapılır
      - Yeni bar sinyalleri değerlendirilir
      - State JSON'a kaydedilir
    """

    def __init__(
        self,
        mode: str,                    # "M4", "M5", "M6"
        symbols: list[str],           # işlem yapılacak coinler
        timeframe: str,               # "15m", "1m"
        capital: float,               # başlangıç sermayesi
        state_file: str,              # JSON state dosyası
        use_universe: bool = False,   # M5 için True
        m4_mode: bool = False,
        m5_mode: bool = False,
        m6_mode: bool = False,
    ):
        self.mode = mode
        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_capital = capital
        self.state_file = Path(state_file)
        self.use_universe = use_universe
        self.m4_mode = m4_mode
        self.m5_mode = m5_mode
        self.m6_mode = m6_mode

        self._warmup_days = WARMUP_DAYS.get(timeframe, 7)
        self._bpd = _bars_per_day(timeframe)

        # Adaptive regime controller (her tick'te güncellenir)
        self._regime_ctrl = AdaptiveRegimeController(smooth_window=12)

    # ── State I/O ─────────────────────────────────────────────────────────────

    def load_state(self) -> dict:
        """JSON state dosyasını yükler. Dosya yoksa varsayılan state döner."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"State dosyası bozuk, sıfırlanıyor: {e}")

        return {
            "mode": self.mode,
            "timeframe": self.timeframe,
            "initial_capital": self.initial_capital,
            "balance": self.initial_capital,
            "open_positions": [],
            "closed_trades": [],
            "last_bar_ts": None,       # son işlenen bar timestamp'i (ISO)
            "final_balance": self.initial_capital,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "win_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "peak_balance": self.initial_capital,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_state(self, state: dict) -> None:
        """State'i JSON dosyasına kaydeder."""
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["final_balance"] = state["balance"]
        # İstatistikleri güncelle
        closed = state.get("closed_trades", [])
        if closed:
            wins = [t for t in closed if t.get("pnl", 0) > 0]
            state["win_rate"] = len(wins) / len(closed)
            state["total_pnl"] = sum(t.get("pnl", 0) for t in closed)
            state["total_pnl_pct"] = state["total_pnl"] / self.initial_capital * 100
        # Peak / drawdown
        peak = state.get("peak_balance", self.initial_capital)
        bal  = state["balance"]
        if bal > peak:
            state["peak_balance"] = bal
        if peak > 0:
            dd = (peak - bal) / peak
            state["max_drawdown_pct"] = max(state.get("max_drawdown_pct", 0.0), dd * 100)

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # ── Veri Çekimi ───────────────────────────────────────────────────────────

    def _fetch_all(self) -> dict[str, pd.DataFrame]:
        """
        Tüm coinlerin indikatörlü DataFrame'ini çeker.
        Returns: {symbol: df_with_indicators}
        """
        sym_ind: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            try:
                raw = fetch_ohlcv(sym, days=self._warmup_days, timeframe=self.timeframe)
                if raw.empty or len(raw) < 50:
                    logger.warning(f"  ⚠ {sym}: yeterli veri yok ({len(raw)} bar)")
                    continue
                inds = TechnicalIndicators()
                df   = prepare_indicators(raw, inds, timeframe=self.timeframe)
                sym_ind[sym] = df
                time.sleep(0.1)   # rate limit
            except Exception as e:
                logger.error(f"  ❌ {sym} veri çekme hatası: {e}")
        return sym_ind

    def _latest_closed_bar_ts(self, sym_ind: dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
        """Tüm coinlerdeki en son ortak kapanmış bar timestamp'ini döndürür."""
        tss = []
        for df in sym_ind.values():
            if not df.empty:
                tss.append(df.index[-1])
        if not tss:
            return None
        # En kısıtlayıcı (en eski son bar) — tüm coinlerde var olan son bar
        return min(tss)

    # ── Stop / Trailing Kontrolü ──────────────────────────────────────────────

    def _check_stops(
        self,
        state: dict,
        bar_ts: pd.Timestamp,
        sym_ind: dict[str, pd.DataFrame],
    ) -> None:
        """
        Açık pozisyonları stop/trailing açısından kontrol eder.
        Kapanan pozisyonları closed_trades'e taşır, balance'ı günceller.
        """
        open_positions = state.get("open_positions", [])
        still_open = []
        balance = state["balance"]

        for pos_d in open_positions:
            sym = pos_d["symbol"]
            df  = sym_ind.get(sym)
            if df is None or bar_ts not in df.index:
                # Veri yok — pozisyonu koru
                pos_d["bars_held"] = pos_d.get("bars_held", 0) + 1
                still_open.append(pos_d)
                continue

            row      = df.loc[bar_ts]
            price    = float(row["close"])
            high_px  = float(row["high"])
            low_px   = float(row["low"])
            atr      = float(row.get("atr", pos_d.get("entry_atr", price * 0.01)))

            pos_d["bars_held"] = pos_d.get("bars_held", 0) + 1
            is_short  = pos_d.get("is_short", False)
            min_hold  = pos_d.get("min_hold_bars", 6)
            trail_px  = float(pos_d["trail_price"])
            stop_px   = float(pos_d["stop_price"])
            trail_mlt = float(pos_d.get("trailing_mult", TRAILING_MULT))

            # ── Trailing stop güncelle ────────────────────────────────────
            if not is_short:
                # LONG: trail_price YUKARI gider, asla aşağı gelmez
                new_trail = high_px - trail_mlt * atr
                if new_trail > trail_px:
                    trail_px = new_trail
                    pos_d["trail_price"] = trail_px
            else:
                # SHORT: trail_price AŞAĞI gider
                new_trail = low_px + trail_mlt * atr
                if new_trail < trail_px:
                    trail_px = new_trail
                    pos_d["trail_price"] = trail_px

            # ── Min hold süresi dolmadıysa stop kontrol etme ─────────────
            if pos_d["bars_held"] < min_hold:
                still_open.append(pos_d)
                continue

            # ── Stop tetiklendi mi? ───────────────────────────────────────
            hit_stop = False
            exit_reason = ""
            exit_price  = price

            if not is_short:
                # LONG: low_px < stop veya low_px < trail
                effective_stop = max(stop_px, trail_px)
                if low_px <= effective_stop:
                    hit_stop    = True
                    exit_price  = effective_stop
                    exit_reason = "trail_stop" if effective_stop == trail_px else "stop_loss"
            else:
                # SHORT: high_px > stop veya high_px > trail
                effective_stop = min(stop_px, trail_px)
                if high_px >= effective_stop:
                    hit_stop    = True
                    exit_price  = effective_stop
                    exit_reason = "trail_stop" if effective_stop == trail_px else "stop_loss"

            if hit_stop:
                # PnL hesapla
                size = float(pos_d["size"])
                if not is_short:
                    fill   = exit_price * (1 - SLIPPAGE)
                    comm   = fill * size * COMMISSION
                    gross  = (fill - float(pos_d["entry_price"])) * size
                    pnl    = gross - comm
                    balance += fill * size - comm
                else:
                    fill   = exit_price * (1 + SLIPPAGE)
                    comm   = fill * size * COMMISSION
                    gross  = (float(pos_d["entry_price"]) - fill) * size
                    pnl    = gross - comm
                    balance += float(pos_d["entry_price"]) * size + pnl  # margin geri + kâr/zarar

                state["closed_trades"].append({
                    "symbol":       sym,
                    "entry_price":  pos_d["entry_price"],
                    "exit_price":   float(fill),
                    "size":         size,
                    "pnl":          float(pnl),
                    "exit_reason":  exit_reason,
                    "entry_date":   pos_d["entry_date"],
                    "exit_date":    bar_ts.isoformat(),
                    "is_short":     is_short,
                    "bars_held":    pos_d["bars_held"],
                })
                pnl_pct = pnl / self.initial_capital * 100
                icon = "✅" if pnl > 0 else "❌"
                logger.info(
                    f"  {icon} KAPAT {sym} | {exit_reason} | "
                    f"giriş={pos_d['entry_price']:.4f} çıkış={fill:.4f} | "
                    f"PnL=${pnl:.2f} ({pnl_pct:+.2f}%)"
                )
            else:
                # Unrealized PnL ekle (sadece loglama için)
                if not is_short:
                    pos_d["unrealized_pnl"] = (price - float(pos_d["entry_price"])) * float(pos_d["size"])
                else:
                    pos_d["unrealized_pnl"] = (float(pos_d["entry_price"]) - price) * float(pos_d["size"])
                still_open.append(pos_d)

        state["open_positions"] = still_open
        state["balance"] = balance

    # ── Giriş Sinyalleri ──────────────────────────────────────────────────────

    def _check_entries(
        self,
        state: dict,
        bar_ts: pd.Timestamp,
        sym_ind: dict[str, pd.DataFrame],
        max_positions: int = 9,
    ) -> None:
        """
        Her coin için son barda BUY/SHORT sinyali var mı kontrol eder.
        Sinyal varsa pozisyon açar, balance'ı günceller.
        """
        balance = state["balance"]
        open_syms = {p["symbol"] for p in state.get("open_positions", [])}
        closed_trades = state.get("closed_trades", [])

        # Rejim parametreleri: BTC verisiyle AdaptiveRegimeCtrl güncelle
        regime_params = None
        btc_df = sym_ind.get("BTC/USDT")
        if btc_df is not None and bar_ts in btc_df.index:
            try:
                _, regime_params = self._regime_ctrl.update(bar_ts, btc_df, sym_ind)
            except Exception as e:
                logger.debug(f"Regime update hata: {e}")

        # Portföy koruyucu: max pozisyon sınırı
        if len(state.get("open_positions", [])) >= max_positions:
            return

        # Global bear/bull tespiti
        in_global_bear = False
        in_strong_bear = False
        in_global_bull  = False
        if regime_params:
            in_global_bear = regime_params.entry_score_boost >= 0.07
            # M6 (1m scalping): 1m data gürültülü → STRONG_BEAR eşiği daha yüksek tutulur
            # 15m/1h için 0.20 yeterli; 1m için 0.20 çok sık tetiklenir → 0.20 (STRONG_BEAR) korunur
            # ama M6'da bear rejimde SHORT sinyalleri de değerlendirilir (normal: sadece bear+coin_bull)
            in_strong_bear = regime_params.entry_score_boost >= 0.20 and not self.m6_mode
            in_global_bull  = regime_params.entry_score_boost <= -0.03

        # Coin max kayıp eşiği
        coin_max_loss = self.initial_capital * 0.012

        for sym in self.symbols:
            if sym in open_syms:
                continue  # Zaten açık pozisyon var

            df = sym_ind.get(sym)
            if df is None or bar_ts not in df.index:
                continue

            # Son kapanmış bara kadar olan slice
            slice_df = df.loc[:bar_ts]
            if len(slice_df) < 50:
                continue

            row   = slice_df.iloc[-1]
            price = float(row.get("close", 0.0))
            atr   = float(row.get("atr", 0.0))
            if atr <= 0 or price <= 0:
                continue

            # ── Per-coin trend durumu ─────────────────────────────────────
            ema200_col = next(
                (c for c in ("ema_200", "ema_slow", "ema200") if c in slice_df.columns), None
            )
            coin_own_bull = False
            if ema200_col:
                ema200 = float(slice_df[ema200_col].iloc[-1])
                if ema200 > 0:
                    coin_own_bull = price >= ema200

            # ── Bear rejim filtresi ───────────────────────────────────────
            if in_strong_bear:
                continue  # STRONG BEAR: hiç giriş yok
            if in_global_bear and not coin_own_bull:
                # M5: bear + coin kendi bull değil → SHORT kabul (M5 daha agresif)
                if not self.m5_mode or not is_short_signal:
                    continue
            # M4: bear'de ekstra ADX filtresi — sadece güçlü trend sinyali
            # M4 konservatif: bear'de coin bull olsa bile ADX < 25 → giriş yok
            if self.m4_mode and in_global_bear and coin_own_bull:
                adx_now = float(row.get("adx", 0))
                if adx_now < 25:
                    continue

            # ── 30 günlük kayan kayıp limiti ─────────────────────────────
            sym_trades_30d = [
                t for t in closed_trades
                if t["symbol"] == sym
                and (pd.Timestamp(bar_ts) - pd.Timestamp(t.get("exit_date", "2000-01-01")))
                    <= pd.Timedelta(days=30)
            ]
            rolling_pnl_30d = sum(t.get("pnl", 0) for t in sym_trades_30d)
            if rolling_pnl_30d < -coin_max_loss:
                continue

            # ── Strateji sinyali ──────────────────────────────────────────
            try:
                strategy, risk_params = make_strategy(
                    sym, coin_df=slice_df, timeframe=self.timeframe
                )
                signal = strategy.generate_signal(sym, slice_df, allow_short=True)
            except Exception as e:
                logger.debug(f"  Signal error {sym}: {e}")
                continue

            is_short_signal = signal.side == Side.SHORT
            if signal.side not in (Side.BUY, Side.SHORT):
                continue

            # ── EMA50 onayı ───────────────────────────────────────────────
            # M6 (1m): son 2 bar yeterli (3 bar çok strict, 1m'de oscillasyon fazla)
            # M4/M5 (15m): son 3 bar (daha az gürültü, güçlü onay gerekli)
            ema50_confirm_bars = 2 if self.m6_mode else 3
            ema50_col = next(
                (c for c in ("ema_50", "ema50", "ema_fast") if c in slice_df.columns), None
            )
            if not is_short_signal:
                if ema50_col and len(slice_df) >= ema50_confirm_bars:
                    last_n = slice_df.tail(ema50_confirm_bars)
                    if not (last_n["close"] > last_n[ema50_col]).all():
                        continue
            else:
                if ema50_col and len(slice_df) >= 2:
                    last2 = slice_df.tail(2)
                    if not (last2["close"] < last2[ema50_col]).all():
                        continue

            # ── Global bull'da SHORT yasak ────────────────────────────────
            if is_short_signal and in_global_bull:
                continue
            if is_short_signal and coin_own_bull:
                continue

            # ── Pozisyon boyutu ───────────────────────────────────────────
            risk_pct    = risk_params.get("risk_per_trade", RISK_PER_TRADE)
            atr_stop    = risk_params.get("atr_stop_multiplier", ATR_STOP_MULT)
            max_pos_pct = risk_params.get("max_position_pct", MAX_POSITION_PCT)

            # M5: M4'e göre daha agresif pozisyon boyutu (backtest farkını live'a taşı)
            # M4: konservatif (defansif), M5: agresif (momentum odaklı, büyük upside)
            if self.m5_mode:
                risk_pct    *= 1.25   # %25 daha büyük risk
                max_pos_pct *= 1.30   # %30 daha geniş pozisyon limiti

            # M6 (scalping): daha küçük pozisyonlar, hızlı dönüş
            if self.m6_mode:
                risk_pct    *= 0.80   # %20 daha küçük risk (scalping)
                max_pos_pct = min(max_pos_pct, 0.15)  # max %15 per pozisyon

            # Rejim boyut çarpanı
            pos_mult = 1.0
            if regime_params:
                pos_mult = regime_params.position_size_mult
            if in_global_bear and coin_own_bull:
                pos_mult = max(pos_mult, 0.75)
            if in_global_bear:
                max_pos_pct *= 0.5
                risk_pct    *= 0.5

            risk_amt  = balance * risk_pct * pos_mult
            if is_short_signal:
                stop_px   = price + atr_stop * atr
                stop_dist = max(stop_px - price, price * 0.001)
            else:
                stop_px   = price - atr_stop * atr
                stop_dist = max(price - stop_px, price * 0.001)

            size = risk_amt / stop_dist

            # Max position cap
            max_cost = balance * max_pos_pct
            if size * price > max_cost:
                size = max_cost / price

            cost = size * price
            if cost < MIN_ORDER_SIZE:
                continue
            if cost > balance * 0.95:
                continue

            # Trailing stop
            base_trail = risk_params.get("trailing_stop_atr_multiplier", TRAILING_MULT)
            trail_boost = regime_params.trailing_mult_boost if regime_params else 0.0
            if coin_own_bull and not is_short_signal:
                trail_mult = max(2.0, base_trail * 1.4 + max(0.0, trail_boost))
            else:
                trail_mult = max(1.5, base_trail + trail_boost)

            # Giriş fiyatı (slippage dahil)
            if is_short_signal:
                fill_price = price * (1 - SLIPPAGE)
                comm       = fill_price * size * COMMISSION
                total_cost = atr_stop * atr * size + comm   # marjin rezervi
                trail_px   = fill_price + trail_mult * atr
            else:
                fill_price = price * (1 + SLIPPAGE)
                comm       = fill_price * size * COMMISSION
                total_cost = fill_price * size + comm
                trail_px   = fill_price - trail_mult * atr

            if total_cost > balance:
                continue

            # Min hold bars (TF-aware)
            # M6 (1m scalping): 12 bar = 12 dk / 6 bar = 6 dk — scalping'e uygun
            # M4/M5 (15m swing): hold_scale ile saatlerce tutma
            if self.m6_mode:
                min_hold = 12 if coin_own_bull else 6   # 1m bar = dakika cinsinden
            else:
                hold_scale = self._bpd / 24
                min_hold   = int((12 if coin_own_bull else 6) * hold_scale)

            balance -= total_cost

            pos = LivePosition(
                symbol        = sym,
                entry_price   = float(fill_price),
                stop_price    = float(price + atr_stop * atr if is_short_signal else price - atr_stop * atr),
                trail_price   = float(trail_px),
                size          = float(size),
                cost          = float(total_cost),
                entry_date    = bar_ts.isoformat(),
                entry_atr     = float(atr),
                trailing_mult = float(trail_mult),
                is_short      = is_short_signal,
                coin_own_bull = coin_own_bull,
                min_hold_bars = min_hold,
            )
            state["open_positions"].append(pos.to_dict())
            open_syms.add(sym)

            side_str = "SHORT" if is_short_signal else "LONG"
            adx_val  = float(row.get("adx", 0))
            logger.info(
                f"  📈 AÇILIYOR {side_str} {sym} @ {fill_price:.4f} | "
                f"stop={pos.stop_price:.4f} trail={trail_px:.4f} | "
                f"size={size:.4f} cost=${total_cost:.2f} | "
                f"ADX={adx_val:.1f} ATR={atr:.4f}"
            )

            # Max pozisyon kontrolü
            if len(state["open_positions"]) >= max_positions:
                break

        state["balance"] = balance

    # ── Ana Tick ──────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        Bir tick işlemi:
        1. State yükle
        2. Veri çek (tüm coinler)
        3. Son bar belirle — zaten işlendiyse çık
        4. Stop/trailing kontrolü
        5. Giriş sinyalleri
        6. State kaydet
        Returns: güncellenmiş state dict
        """
        print(f"\n  [{self.mode}] {self.timeframe} tick başladı — {datetime.now().strftime('%H:%M:%S')}")

        # 1) State yükle
        state = self.load_state()

        # 2) Veri çek
        print(f"  [{self.mode}] {len(self.symbols)} coin verisi çekiliyor...")
        sym_ind = self._fetch_all()
        if not sym_ind:
            print(f"  [{self.mode}] ❌ Veri çekme başarısız — tick atlandı")
            return state

        # 3) Son bar timestamp'ini belirle
        bar_ts = self._latest_closed_bar_ts(sym_ind)
        if bar_ts is None:
            print(f"  [{self.mode}] ❌ Bar timestamp bulunamadı — tick atlandı")
            return state

        # Duplicate guard: bu bar daha önce işlendi mi?
        last_bar_ts_str = state.get("last_bar_ts")
        if last_bar_ts_str:
            last_bar_ts = pd.Timestamp(last_bar_ts_str)
            if bar_ts <= last_bar_ts:
                print(
                    f"  [{self.mode}] ⏸  Son bar {bar_ts} zaten işlendi "
                    f"(son: {last_bar_ts}) — tick atlandı"
                )
                return state

        print(
            f"  [{self.mode}] 🕐 Yeni bar: {bar_ts} | "
            f"Bakiye: ${state['balance']:.2f} | "
            f"Açık: {len(state.get('open_positions', []))} | "
            f"Kapalı: {len(state.get('closed_trades', []))}"
        )

        # 4) Stop/trailing kontrolü (önce çıkışlar)
        self._check_stops(state, bar_ts, sym_ind)

        # 5) Yeni giriş sinyalleri
        max_pos = 9 if not self.m6_mode else 5
        if regime_params := self._get_regime_params(bar_ts, sym_ind):
            max_pos = regime_params.max_positions
        self._check_entries(state, bar_ts, sym_ind, max_positions=max_pos)

        # 6) Coin benchmark güncelle (bot başlangıcından bu yana fiyat değişimi)
        self._update_benchmarks(state, bar_ts, sym_ind)

        # 7) Last bar timestamp'i güncelle
        state["last_bar_ts"] = bar_ts.isoformat()

        # 8) State kaydet
        self.save_state(state)

        n_open   = len(state.get("open_positions", []))
        n_closed = len(state.get("closed_trades", []))
        balance  = state["balance"]
        print(
            f"  [{self.mode}] ✅ Tick tamamlandı | "
            f"Bakiye: ${balance:.2f} | Açık: {n_open} | Kapalı: {n_closed}"
        )
        return state

    def _update_benchmarks(
        self,
        state: dict,
        bar_ts: pd.Timestamp,
        sym_ind: dict[str, pd.DataFrame],
    ) -> None:
        """
        Bot başlatıldığından bu yana her coinin fiyat değişimini takip eder.
        İlk çalışmada start_price kaydedilir, sonraki ticklerde current_price güncellenir.
        """
        existing: dict[str, dict] = {
            b["symbol"]: b
            for b in state.get("coin_benchmarks", [])
        }
        updated = []
        now_iso = bar_ts.isoformat()

        for sym, df in sym_ind.items():
            if df.empty or bar_ts not in df.index:
                continue
            current_price = float(df.loc[bar_ts, "close"])
            if current_price <= 0:
                continue

            if sym in existing:
                bm = dict(existing[sym])
                bm["current_price"] = current_price
                start_px = bm.get("start_price", current_price)
                bm["pct_change"] = ((current_price - start_px) / start_px * 100) if start_px > 0 else 0.0
                bm["updated_at"] = now_iso
            else:
                bm = {
                    "symbol":        sym,
                    "start_price":   current_price,
                    "current_price": current_price,
                    "pct_change":    0.0,
                    "start_date":    now_iso,
                    "updated_at":    now_iso,
                }
            updated.append(bm)

        state["coin_benchmarks"] = sorted(updated, key=lambda b: b["symbol"])

    def _get_regime_params(self, bar_ts, sym_ind):
        """Mevcut rejim parametrelerini döndürür (hata durumunda None)."""
        btc_df = sym_ind.get("BTC/USDT")
        if btc_df is not None and bar_ts in btc_df.index:
            try:
                _, params = self._regime_ctrl.update(bar_ts, btc_df, sym_ind)
                return params
            except Exception:
                pass
        return None
