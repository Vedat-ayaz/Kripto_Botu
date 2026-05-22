"""yfinance tabanlı BIST OHLCV veri sağlayıcısı. Semboller .IS eki ile çekilir."""
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "cache"


class YFinanceProvider:
    """
    BIST hisse OHLCV verisi çeker. Sonuçları parquet'a cache'ler.

    Kullanım:
        provider = YFinanceProvider()
        df = provider.fetch("GARAN", start="2020-01-01", end="2026-01-01", interval="1d")
        # df columns: open, high, low, close, volume  (hepsi küçük harf)
        # df index: DatetimeIndex (UTC)
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        OHLCV DataFrame döndürür. Sütunlar küçük harf, index DatetimeIndex (UTC).
        interval="1h": yfinance max ~730 gün geriye gider.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance kurulu değil. 'pip install yfinance' komutunu çalıştırın."
            ) from exc

        if not force_refresh:
            cached = self._load_cache(symbol, interval, start, end)
            if cached is not None:
                return cached

        ticker = f"{symbol}.IS"
        logger.info(f"[YFinanceProvider] {ticker} çekiliyor: {start} → {end} ({interval})")

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"[YFinanceProvider] {ticker} için veri gelmedi ({start} → {end}). "
                "Sembol hatalı olabilir veya yfinance erişimi kısıtlı."
            )

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df["close"].notna() & (df["close"] > 0)]

        if self.cache_enabled:
            self._save_cache(symbol, interval, df)

        return df

    def fetch_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Çoklu sembol çeker. Başarısız olanları atlar ve uyarı loglar."""
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch(sym, start, end, interval, force_refresh)
            except Exception as e:
                logger.warning(f"[YFinanceProvider] {sym} atlandı: {e}")
        return result

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return _CACHE_DIR / f"{symbol}_{interval}.parquet"

    def _load_cache(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame | None:
        path = self._cache_path(symbol, interval)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            req_start = pd.Timestamp(start, tz="UTC")
            req_end = pd.Timestamp(end, tz="UTC")
            # Son 7 gün içindeyse yenile
            stale = req_end >= (pd.Timestamp.now(tz="UTC") - timedelta(days=7))
            if stale:
                return None
            if df.index.min() <= req_start and df.index.max() >= req_end - timedelta(days=7):
                logger.debug(f"[YFinanceProvider] Cache hit: {path.name}")
                return df.loc[req_start:req_end]
        except Exception as e:
            logger.warning(f"[YFinanceProvider] Cache okunamadı ({path.name}): {e}")
        return None

    def _save_cache(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, interval)
        try:
            df.to_parquet(path)
            logger.debug(f"[YFinanceProvider] Cache kaydedildi: {path.name}")
        except Exception as e:
            logger.warning(f"[YFinanceProvider] Cache kaydedilemedi: {e}")
