"""
BIST canlı veri soyutlama katmanı.

Mimarisi:
  BistDataSource (soyut) ─── YFinancePollingSource  (paper / geliştirme)
                         └── StreamingDataSource      (canlı broker API stub)

Aynı LiveRunner her iki kaynakla da çalışır — sadece BistDataSource değişir.

Bir broker API'ı (Matriks, GarantiBT, vs.) entegre edildiğinde:
    class MatriksSource(BistDataSource):
        def get_history(...): ...
        def subscribe(...): ...
    runner = BistLiveRunner(cfg, data_source=MatriksSource(...))
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Callable, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


# ── Ortak event tipi ──────────────────────────────────────────────────────────

class BarEvent(TypedDict):
    """Her mum kapanışında tetiklenen event."""
    symbol: str
    timestamp: pd.Timestamp    # UTC, bar kapanış zamanı
    open: float
    high: float
    low: float
    close: float
    volume: float


BarCallback = Callable[[BarEvent], None]


# ── Soyut base ────────────────────────────────────────────────────────────────

class BistDataSource(ABC):
    """
    Tüm veri kaynakları bu sınıfı implemente eder.
    LiveRunner sadece bu arayüze bağlıdır.
    """

    @abstractmethod
    def get_history(self, symbol: str, bars: int = 300) -> pd.DataFrame:
        """
        Son `bars` adet OHLCV verisini döndürür.
        Sütunlar: open, high, low, close, volume (küçük harf)
        Index: DatetimeIndex (UTC, timezone-aware)
        Yeni mum henüz kapanmadıysa dahil edilmez.
        """

    @abstractmethod
    def subscribe(self, symbols: list[str], on_bar: BarCallback) -> None:
        """Her yeni mum kapanışında on_bar(event) çağrılacak şekilde kaydeder."""

    @abstractmethod
    def start(self) -> None:
        """Arka plan veri döngüsünü başlatır (blocking veya thread)."""

    @abstractmethod
    def stop(self) -> None:
        """Veri döngüsünü durdurur."""


# ── yfinance Polling ──────────────────────────────────────────────────────────

class YFinancePollingSource(BistDataSource):
    """
    yfinance'ten belirli aralıkla veri çeker (paper trading / geliştirme).

    Her bar_interval'da bir yeni mum var mı kontrol eder.
    Yeni mum kapandıysa tüm abone callback'leri tetikler.

    Kısıtlamalar:
      - 5m data: son 60 gün
      - 1d data: sınırsız
      - Fiyatlar 15 dakika gecikmeli (yfinance ücretsiz kota)
    """

    def __init__(
        self,
        interval: str = "1d",
        poll_seconds: int = 60,
        usd_mode: str = "convert_series",
        cache_enabled: bool = False,
    ):
        self.interval      = interval
        self.poll_seconds  = poll_seconds
        self.usd_mode      = usd_mode
        self.cache_enabled = cache_enabled

        self._symbols:   list[str] = []
        self._callback:  BarCallback | None = None
        self._last_ts:   dict[str, pd.Timestamp | None] = {}
        self._stop_evt   = threading.Event()
        self._thread:    threading.Thread | None = None

    # ── BistDataSource impl ──────────────────────────────────────────────────

    def get_history(self, symbol: str, bars: int = 300) -> pd.DataFrame:
        """Son bars adet OHLCV döndürür (USD cinsinden)."""
        from bist.data.yfinance_provider import YFinanceProvider
        from bist.data.usdtry_provider import USDTRYProvider
        from bist.adapters.price_converter import PriceConverter

        days_back = max(bars * 2, 400)
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        provider  = YFinanceProvider(cache_enabled=self.cache_enabled)
        usdtry_p  = USDTRYProvider(cache_enabled=self.cache_enabled)
        converter = PriceConverter(mode=self.usd_mode)

        df_try = provider.fetch(symbol, start=start, end=end, interval=self.interval, force_refresh=True)
        usdtry = usdtry_p.fetch(start=start, end=end)
        df_usd = converter.convert_ohlcv(df_try, usdtry)
        return df_usd.iloc[-bars:]

    def subscribe(self, symbols: list[str], on_bar: BarCallback) -> None:
        self._symbols  = list(symbols)
        self._callback = on_bar
        self._last_ts  = {s: None for s in symbols}

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="YFinancePoll")
        self._thread.start()
        logger.info(f"[YFinancePollingSource] Başladı — interval={self.interval}, poll={self.poll_seconds}s")
        try:
            self._thread.join()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("[YFinancePollingSource] Durduruldu.")

    # ── Arka plan döngüsü ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            for sym in self._symbols:
                if self._stop_evt.is_set():
                    break
                try:
                    self._check_symbol(sym)
                except Exception as e:
                    logger.error(f"[YFinancePollingSource] {sym} poll hatası: {e}")
            self._stop_evt.wait(self.poll_seconds)

    def _check_symbol(self, sym: str) -> None:
        """Sembol için son mumu kontrol et; yeniyse callback tetikle."""
        df = self.get_history(sym, bars=10)
        if df.empty:
            return

        last_ts    = pd.Timestamp(df.index[-1])
        prev_ts    = self._last_ts.get(sym)
        self._last_ts[sym] = last_ts

        if prev_ts is None or last_ts <= prev_ts:
            return  # Yeni mum yok

        row = df.iloc[-1]
        event: BarEvent = {
            "symbol":    sym,
            "timestamp": last_ts,
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row["volume"]),
        }
        logger.info(f"[YFinancePollingSource] Yeni mum: {sym} @ {last_ts} close={row['close']:.4f}")
        if self._callback:
            self._callback(event)


# ── Streaming Stub ────────────────────────────────────────────────────────────

class StreamingDataSource(BistDataSource):
    """
    Gerçek zamanlı broker/data-provider entegrasyonu için stub.

    Kullanım (Matriks, GarantiBT API, vb.):
      1. Bu sınıfı sub-class edin
      2. get_history() ve _connect_to_broker() metodlarını doldurun
      3. Broker'dan gelen her mum verisinde _emit_bar() çağırın

    Örnek:
        class MatriksSource(StreamingDataSource):
            def get_history(self, symbol, bars=300):
                return matriks_api.get_ohlcv(symbol, bars)

            def _connect_to_broker(self):
                matriks_api.on_bar = self._on_matriks_bar
                matriks_api.subscribe(self._symbols)

            def _on_matriks_bar(self, raw):
                self._emit_bar({
                    "symbol": raw["ticker"],
                    "timestamp": pd.Timestamp(raw["ts"], tz="UTC"),
                    "open": raw["o"], "high": raw["h"],
                    "low": raw["l"], "close": raw["c"],
                    "volume": raw["v"],
                })
    """

    def __init__(self) -> None:
        self._symbols:  list[str] = []
        self._callback: BarCallback | None = None
        self._stop_evt  = threading.Event()

    def get_history(self, symbol: str, bars: int = 300) -> pd.DataFrame:
        raise NotImplementedError("Broker-specific history endpoint implementasyonu gerekli.")

    def subscribe(self, symbols: list[str], on_bar: BarCallback) -> None:
        self._symbols  = list(symbols)
        self._callback = on_bar

    def start(self) -> None:
        self._stop_evt.clear()
        logger.info(f"[StreamingDataSource] Broker bağlantısı kuruluyor...")
        self._connect_to_broker()
        self._stop_evt.wait()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("[StreamingDataSource] Durduruldu.")

    def _connect_to_broker(self) -> None:
        """Sub-class'ta broker WebSocket / REST stream bağlantısı kurulur."""
        raise NotImplementedError

    def _emit_bar(self, event: BarEvent) -> None:
        """Broker'dan gelen mum verisini callback'e iletir."""
        if self._callback:
            self._callback(event)
