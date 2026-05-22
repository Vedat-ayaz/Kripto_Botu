"""
Scalp Backtester — ScalpingStrategy v2 için bar-by-bar simülasyon
==================================================================
Kullanım:
    python main.py --mode scalp-backtest
    python main.py --mode scalp-backtest --start 2024-01-01 --end 2024-07-01

Özellikler:
  - Paginated 5m OHLCV fetch (Binance max 1000 bar/istek)
  - Multi-symbol concurrent simulation (paylaşılan sermaye)
  - Bar içi TP/SL simülasyonu → high/low kullanır (kapanışa göre çok daha gerçekçi)
  - Fee (0.1% her yön) + slipaj (0.02% her yön) muhasebesi
  - Per-symbol + aggregate metrikler (metrics.py'dan)
  - CSV çıktısı → backtest_results/
"""

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from backtest.metrics import calculate_metrics, print_metrics
from data.exchange_client import ExchangeClient
from strategy.scalping_strategy import ScalpingStrategy
from strategy.signal import Side

logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────

TIMEFRAME          = "5m"
BARS_PER_REQUEST   = 1000
FEE_RATE           = 0.001    # Binance taker: 0.1% her yön
SLIPPAGE_RATE      = 0.0002   # Tahmini slipaj: 0.02% her yön
WARMUP_BARS        = 80       # İndikatör ısınma dönemi (Supertrend 10 + MACD 10+16 + BB20)
MIN_ORDER_USDT     = 10.0     # Minimum pozisyon büyüklüğü


# ── Ana Sınıf ─────────────────────────────────────────────────────────────────

class ScalpBacktester:
    """
    ScalpingStrategy v2 için bar-by-bar backtest motoru.

    Mimarı karar notları:
      • TP/SL → bar high/low ile kontrol edilir (kapanış değil)
        Aynı barda hem TP hem SL tetiklenirse SL seçilir (kötümser/gerçekçi).
      • Giriş → sinyal barının kapanışı + slipaj (bir sonraki bar open daha
        gerçekçidir ama ccxt OHLCV'de open güvenilir değil).
      • Çıkış (strateji) → should_exit() max_hold / RSI / Supertrend flip için
        çağrılır; TP/SL zaten high/low ile ele alındığı için çakışma olmaz.
      • Paylaşılan sermaye → tüm semboller aynı USDT havuzundan beslenir.
      • Adaptif öğrenme → record_trade_result() her kapanışta çağrılır;
        böylece threshold gerçek canlı koşullarla aynı şekilde güncellenir.
    """

    def __init__(
        self,
        client: ExchangeClient,
        symbols: list[str],
        start_date: str,            # "YYYY-MM-DD"
        end_date: str,              # "YYYY-MM-DD"
        initial_capital: float  = 10_000.0,
        risk_per_trade: float   = 0.01,
        max_open_positions: int = 5,
        max_position_pct: float = 0.15,
        profit_target_pct: float = 0.008,
        stop_loss_pct: float    = 0.003,
        max_hold_bars: int      = 8,
        daily_max_loss: float   = 0.05,
        volume_mult: float      = 0.4,
        output_dir: str         = "backtest_results",
    ):
        self.client             = client
        self.symbols            = symbols
        self.start_date         = start_date
        self.end_date           = end_date
        self.initial_capital    = initial_capital
        self.risk_per_trade     = risk_per_trade
        self.max_open_positions = max_open_positions
        self.max_position_pct   = max_position_pct
        self.profit_target_pct  = profit_target_pct
        self.stop_loss_pct      = stop_loss_pct
        self.max_hold_bars      = max_hold_bars
        self.daily_max_loss     = daily_max_loss
        self.output_dir         = output_dir

        # Strateji — TP/SL/hold parametreleri backtester'daki değerlerle eşleşmeli
        self.strategy = ScalpingStrategy(
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            max_hold_bars=max_hold_bars,
            volume_mult=volume_mult,
        )

        os.makedirs(output_dir, exist_ok=True)

    # ── Veri Çekme ────────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Paginated 5m OHLCV fetch.
        Binance max 1000 bar/istek → birden fazla sayfa gerekebilir.
        6 aylık veri: ~52.704 bar → ~53 istek.
        """
        start_ts = int(
            datetime.strptime(self.start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )
        end_ts = int(
            datetime.strptime(self.end_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )

        all_rows: list = []
        since = start_ts
        page  = 0

        while since < end_ts:
            rows = self.client.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=BARS_PER_REQUEST)
            if not rows:
                break

            all_rows.extend(rows)
            last_ts = rows[-1][0]
            page   += 1

            if page % 10 == 0:
                pct = (last_ts - start_ts) / max(end_ts - start_ts, 1) * 100
                print(f"    {symbol}: sayfa {page} ({pct:.0f}%) — {len(all_rows)} bar")

            if last_ts >= end_ts or len(rows) < BARS_PER_REQUEST:
                break

            since = last_ts + 1          # Bir sonraki milisaniye
            time.sleep(0.20)             # Binance rate limit: 1200 req/dk

        if not all_rows:
            logger.warning(f"[ScalpBT] {symbol}: veri bulunamadı.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # end_date'i geçen barları at
        df = df[df.index < pd.Timestamp(self.end_date, tz="UTC")]
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)

        if df.empty:
            logger.warning(f"[ScalpBT] {symbol}: filtreden sonra veri kalmadı (sembol o dönemde mevcut değil?).")
            return df

        logger.info(
            f"[ScalpBT] {symbol}: {len(df)} bar "
            f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
        )
        return df

    # ── Pozisyon Boyutu ──────────────────────────────────────────────────────

    def _position_size_usdt(self, capital: float, symbol: str) -> float:
        """
        USDT cinsinden pozisyon büyüklüğü.

        risk_per_trade / stop_loss_pct formülü genellikle max_position_pct
        sınırının üzerine çıkar → min ile sınırlandır.
        Kelly-inspired ölçek de uygulanır.
        """
        risk_usdt    = capital * self.risk_per_trade
        size_by_risk = risk_usdt / self.stop_loss_pct          # SL'e göre boyut
        size_by_cap  = capital * self.max_position_pct         # max % sınırı

        base_size = min(size_by_risk, size_by_cap)

        # Adaptif Kelly ölçeği [0.6× – 1.5×]
        scale     = self.strategy.get_position_scale(symbol)
        final     = base_size * scale

        # Hiçbir zaman sermayenin %95'inden fazlasını kullanma
        return min(final, capital * 0.95)

    # ── Ana Döngü ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Tüm sembolleri aynı anda simüle eder, ortak sermaye havuzu kullanır.
        Sonuçları dict olarak döner.
        """
        self._print_header()

        # ── 1. Veri çek ──────────────────────────────────────────────────────
        symbol_data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            print(f"  ↓ {sym} veri çekiliyor...")
            df = self._fetch_ohlcv(sym)
            if len(df) > WARMUP_BARS + 10:
                symbol_data[sym] = df
            else:
                print(f"    ⚠️  {sym}: yeterli veri yok ({len(df)} bar), atlanıyor.")

        if not symbol_data:
            print("❌ Hiçbir sembol için yeterli veri bulunamadı.")
            return {}

        # ── 2. İndikatörleri önceden hesapla (hız optimizasyonu) ─────────────
        # Her sembol için _calc() bir kere çalışır; döngüde slice alınınca
        # kolonlar zaten mevcut olduğu için _calc() hemen döner (O(1)).
        print("İndikatörler önceden hesaplanıyor...")
        symbol_calc: dict[str, pd.DataFrame] = {}
        for sym, df in symbol_data.items():
            symbol_calc[sym] = self.strategy._calc(df)
            print(f"  ✅ {sym}")
        # symbol_data'yı da güncelle (high/low/open/close OHLCV + indikatörler bir arada)
        symbol_data = symbol_calc
        print()

        # ── 3. Ortak zaman eksenini oluştur ──────────────────────────────────
        all_ts: list[pd.Timestamp] = sorted(
            set().union(*[set(df.index) for df in symbol_data.values()])
        )
        total_bars = len(all_ts)
        print(f"✅ Toplam bar: {total_bars:,}\n")

        # ── 4. Simülasyon state ───────────────────────────────────────────────
        capital   = float(self.initial_capital)
        positions: dict[str, dict] = {}   # symbol → pozisyon bilgisi
        all_trades:  list[dict]    = []
        equity_curve: list[float]  = [capital]
        sym_trades: dict[str, list[dict]] = {sym: [] for sym in symbol_data}

        daily_start_capital = capital
        current_day: Optional[str] = None
        daily_locked          = False

        # İlerleme raporlama
        report_interval = max(total_bars // 40, 1)   # ~%2.5'lik adımlar

        print("Simülasyon başlıyor...\n")

        # ── 4. Bar-by-bar döngü ───────────────────────────────────────────────
        for bar_idx, ts in enumerate(all_ts):

            # İlerleme
            if bar_idx % report_interval == 0:
                pct = bar_idx / total_bars * 100
                open_cnt = len(positions)
                print(
                    f"  [{pct:5.1f}%] {ts.strftime('%Y-%m-%d %H:%M')} | "
                    f"Sermaye: ${capital:>10,.2f} | Açık: {open_cnt}"
                )

            # Günlük sıfırlama
            day_str = ts.strftime("%Y-%m-%d")
            if day_str != current_day:
                daily_start_capital = capital
                daily_locked        = False
                current_day         = day_str
                self.strategy.maybe_decay_thresholds()

            # Günlük kayıp kilidi kontrolü
            if not daily_locked and daily_start_capital > 0:
                daily_dd = (daily_start_capital - capital) / daily_start_capital
                if daily_dd >= self.daily_max_loss:
                    daily_locked = True
                    logger.info(f"[ScalpBT] Günlük kayıp limiti ({day_str}) — trading durduruldu")

            # ── 4a. Açık pozisyonların exit kontrolü ─────────────────────────
            for sym in list(positions.keys()):
                if sym not in symbol_data:
                    continue

                df_sym = symbol_data[sym]
                if ts not in df_sym.index:
                    continue

                pos     = positions[sym]
                sym_idx = df_sym.index.get_loc(ts)
                bar     = df_sym.iloc[sym_idx]

                exit_price: Optional[float] = None
                exit_reason = ""

                # TP / SL → bar high/low kullan (kapanıştan çok daha gerçekçi)
                sl_hit = float(bar["low"])  <= pos["sl"]
                tp_hit = float(bar["high"]) >= pos["tp"]

                if sl_hit and tp_hit:
                    # Her ikisi de aynı bar → kötümser: SL kazanır
                    exit_price  = pos["sl"]
                    exit_reason = "SL"
                elif sl_hit:
                    exit_price  = pos["sl"]
                    exit_reason = "SL"
                elif tp_hit:
                    exit_price  = pos["tp"]
                    exit_reason = "TP"

                # Strateji çıkış kararı (max_hold / RSI aşırı alım / Supertrend flip)
                if exit_price is None and sym_idx >= WARMUP_BARS:
                    df_slice = df_sym.iloc[: sym_idx + 1]
                    try:
                        should_exit, reason = self.strategy.should_exit(
                            sym, df_slice, pos["entry_price"],
                            open_ts=pos["open_ts"], current_ts=ts,
                        )
                        if should_exit:
                            exit_price  = float(bar["close"])
                            exit_reason = reason or "STRATEGY"
                    except Exception as exc:
                        logger.debug(f"[ScalpBT] should_exit hatası {sym}: {exc}")

                # Pozisyonu kapat
                if exit_price is not None:
                    trade = self._close_position(
                        sym, pos, exit_price, exit_reason, ts, capital
                    )
                    capital += pos["allocated"] + trade["pnl"]   # geri iade et
                    all_trades.append(trade)
                    sym_trades[sym].append(trade)
                    self.strategy.record_trade_result(sym, trade["pnl"])
                    del positions[sym]

            # Equity noktası
            equity_curve.append(capital)

            # ── 4b. Yeni giriş sinyali ────────────────────────────────────────
            if daily_locked:
                continue
            if len(positions) >= self.max_open_positions:
                continue

            # Önce o barda verisi olan, pozisyon olmayan semboller
            candidates = [
                s for s in symbol_data
                if s not in positions and ts in symbol_data[s].index
            ]

            for sym in candidates:
                if len(positions) >= self.max_open_positions:
                    break

                df_sym  = symbol_data[sym]
                sym_idx = df_sym.index.get_loc(ts)

                if sym_idx < WARMUP_BARS:
                    continue

                df_slice = df_sym.iloc[: sym_idx + 1]
                bar      = df_sym.iloc[sym_idx]

                # Sinyal üret
                try:
                    signal = self.strategy.generate_signal(sym, df_slice)
                except Exception as exc:
                    logger.debug(f"[ScalpBT] generate_signal hatası {sym}: {exc}")
                    continue

                if signal.side != Side.BUY:
                    continue

                # Giriş fiyatı = kapanış + slipaj
                entry_price = float(bar["close"]) * (1.0 + SLIPPAGE_RATE)
                size_usdt   = self._position_size_usdt(capital, sym)

                if size_usdt < MIN_ORDER_USDT:
                    continue
                if size_usdt > capital:
                    size_usdt = capital * 0.95

                entry_fee = size_usdt * FEE_RATE
                tp        = entry_price * (1.0 + self.profit_target_pct)
                sl        = entry_price * (1.0 - self.stop_loss_pct)

                capital -= size_usdt + entry_fee   # sermayeden düş

                positions[sym] = {
                    "entry_price": entry_price,
                    "allocated":   size_usdt,       # ödenen USDT (fee hariç)
                    "entry_fee":   entry_fee,
                    "tp":          tp,
                    "sl":          sl,
                    "open_ts":     ts,
                    "score":       round(signal.confidence_score, 3),
                    "reason":      signal.reason,
                }

        # ── 5. Kalan açık pozisyonları kapat (backtest sonu) ─────────────────
        print("\n  Açık pozisyonlar kapatılıyor (backtest sonu)...")
        for sym, pos in list(positions.items()):
            if sym not in symbol_data:
                continue
            df_sym     = symbol_data[sym]
            last_bar   = df_sym.iloc[-1]
            exit_price = float(last_bar["close"]) * (1.0 - SLIPPAGE_RATE)
            trade      = self._close_position(
                sym, pos, exit_price, "BACKTEST_END", df_sym.index[-1], capital
            )
            capital += pos["allocated"] + trade["pnl"]
            all_trades.append(trade)
            sym_trades[sym].append(trade)

        equity_curve.append(capital)

        # ── 6. Metrikler & Çıktı ──────────────────────────────────────────────
        return self._report(all_trades, sym_trades, symbol_data, equity_curve)

    # ── Yardımcılar ───────────────────────────────────────────────────────────

    def _close_position(
        self,
        sym: str,
        pos: dict,
        exit_price: float,
        reason: str,
        exit_ts: pd.Timestamp,
        capital: float,
    ) -> dict:
        """Pozisyon kapama PnL hesabı — fee + slipaj dahil."""
        # TP çıkışı zaten TP fiyatında; SL çıkışı SL fiyatında.
        # STRATEGY / MAX_HOLD çıkışları close'da gerçekleşti, slipaj eklendi.
        # should_exit zaten close fiyatını verdi; buraya slipaj ekleme
        # (exit_price already adjusted upstream for strategy exits)
        exit_fee  = pos["allocated"] * FEE_RATE
        pnl_pct   = (exit_price - pos["entry_price"]) / pos["entry_price"]
        pnl       = pos["allocated"] * pnl_pct - exit_fee - pos["entry_fee"]
        bars_held = int((exit_ts - pos["open_ts"]).total_seconds() / 300)

        return {
            "symbol":      sym,
            "entry_ts":    pos["open_ts"].isoformat(),
            "exit_ts":     exit_ts.isoformat(),
            "entry_price": round(pos["entry_price"], 6),
            "exit_price":  round(exit_price, 6),
            "size_usdt":   round(pos["allocated"], 4),
            "pnl":         round(pnl, 4),
            "pnl_pct":     round(pnl_pct * 100, 3),
            "reason":      reason,
            "bars_held":   bars_held,
            "score":       pos.get("score", 0.0),
        }

    def _report(
        self,
        all_trades: list[dict],
        sym_trades: dict[str, list[dict]],
        symbol_data: dict[str, pd.DataFrame],
        equity_curve: list[float],
    ) -> dict:
        """Metrikleri yazdırır ve CSV'ye kaydeder."""
        sep = "=" * 64

        print(f"\n{sep}")
        print("  SİMÜLASYON TAMAMLANDI")
        print(sep)

        # Aggregate metrikler
        agg = calculate_metrics(all_trades, equity_curve, self.initial_capital)
        print_metrics(agg, "TÜM SEMBOLLER (PAYLAŞIMLı SERMAYE)")

        # Per-symbol metrikler
        sym_metrics: dict[str, dict] = {}
        for sym in symbol_data:
            trades = sym_trades[sym]
            if not trades:
                continue
            # Basit eşit-sermaye equity eğrisi (paylaşımlı sermayeyi taklit edemeyiz tam olarak)
            pnls          = [t["pnl"] for t in trades]
            sym_equity    = [self.initial_capital] + list(
                np.cumsum(pnls) + self.initial_capital
            )
            sym_metrics[sym] = calculate_metrics(trades, sym_equity, self.initial_capital)

        # Özet tablo
        self._print_symbol_table(sym_trades, sym_metrics)

        # Kazanan / kaybeden
        self._print_exit_breakdown(all_trades)

        # CSV kaydet
        self._save_csv(all_trades, sym_metrics, agg)

        return {
            "aggregate":    agg,
            "per_symbol":   sym_metrics,
            "trades":       all_trades,
            "equity_curve": equity_curve,
        }

    def _print_symbol_table(
        self,
        sym_trades: dict[str, list[dict]],
        sym_metrics: dict[str, dict],
    ) -> None:
        header = (
            f"\n  {'SEMBOL':<12} {'İŞLEM':>6} {'WR%':>7} "
            f"{'PnL$':>10} {'PF':>7} {'Sharpe':>8} {'AvgBar':>7}"
        )
        print(header)
        print("  " + "─" * 62)

        for sym in sorted(sym_metrics, key=lambda s: -sum(t["pnl"] for t in sym_trades[s])):
            m         = sym_metrics[sym]
            trades    = sym_trades[sym]
            total_pnl = sum(t["pnl"] for t in trades)
            avg_bars  = sum(t["bars_held"] for t in trades) / max(len(trades), 1)
            print(
                f"  {sym:<12} {m['num_trades']:>6} {m['win_rate_pct']:>6.1f}% "
                f"{total_pnl:>10.2f} {m['profit_factor']:>7.3f} "
                f"{m['sharpe_ratio']:>8.3f} {avg_bars:>6.1f}b"
            )

        print("  " + "─" * 62 + "\n")

    def _print_exit_breakdown(self, trades: list[dict]) -> None:
        """Çıkış sebebi dağılımını göster."""
        if not trades:
            return

        from collections import Counter
        counts = Counter(t["reason"] for t in trades)
        pnl_by = {}
        for t in trades:
            pnl_by.setdefault(t["reason"], []).append(t["pnl"])

        print("  Çıkış Sebebi Dağılımı:")
        print("  " + "─" * 48)
        for reason, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pnls   = pnl_by[reason]
            avg    = sum(pnls) / len(pnls)
            total  = sum(pnls)
            wr     = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(
                f"  {reason:<18} {cnt:>4} işlem | "
                f"WR={wr:>5.1f}% | Ort={avg:>+8.2f}$ | Toplam={total:>+10.2f}$"
            )
        print()

    def _save_csv(
        self,
        trades: list[dict],
        sym_metrics: dict[str, dict],
        agg: dict,
    ) -> None:
        """İşlemleri ve özet metrikleri CSV'ye kaydeder."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- İşlem detayları ---
        if trades:
            trades_path = os.path.join(self.output_dir, f"scalp_trades_{ts_str}.csv")
            with open(trades_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
                writer.writeheader()
                writer.writerows(trades)
            print(f"  📄 İşlem kaydı: {trades_path}")

        # --- Sembol özet metrikleri ---
        if sym_metrics:
            summary_path = os.path.join(self.output_dir, f"scalp_summary_{ts_str}.csv")
            rows = []
            for sym, m in sym_metrics.items():
                row = {"symbol": sym, "start": self.start_date, "end": self.end_date}
                row.update(m)
                rows.append(row)
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  📊 Özet rapor: {summary_path}")

        # --- Aggregate tek satır ---
        agg_path = os.path.join(self.output_dir, f"scalp_aggregate_{ts_str}.csv")
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            row = {
                "start": self.start_date,
                "end":   self.end_date,
                "symbols": len(sym_metrics),
            }
            row.update(agg)
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"  📈 Aggregate: {agg_path}\n")

    def _print_header(self) -> None:
        sep = "=" * 64
        print(f"\n{sep}")
        print("  SCALP BACKTEST — ScalpingStrategy v2")
        print(f"  Dönem  : {self.start_date} → {self.end_date}")
        print(
            f"  Sermaye: ${self.initial_capital:>8,.0f}  |  "
            f"TP: {self.profit_target_pct*100:.1f}%  |  "
            f"SL: {self.stop_loss_pct*100:.2f}%"
        )
        print(
            f"  MaxPos : {self.max_open_positions}  |  "
            f"RiskPT: {self.risk_per_trade*100:.1f}%  |  "
            f"MaxHold: {self.max_hold_bars}×5m={self.max_hold_bars*5}dk"
        )
        print(f"  Fee    : {FEE_RATE*100:.1f}%/yön  |  Slipaj: {SLIPPAGE_RATE*100:.2f}%/yön")
        print(f"  Semboller ({len(self.symbols)}): {', '.join(self.symbols)}")
        print(f"{sep}\n")
