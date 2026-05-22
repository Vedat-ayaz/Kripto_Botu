"""USD/TRY kur verisi sağlayıcısı — Yahoo Finance 'USDTRY=X' ticker."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "cache"
_YF_TICKER = "USDTRY=X"


class USDTRYProvider:
    """
    USD/TRY döviz kuru çeker ve cache'ler.

    Kullanım:
        rate = USDTRYProvider().fetch(start="2020-01-01", end="2026-01-01")
        # pd.Series[float], DatetimeIndex, name="usdtry"
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(self, start: str, end: str, interval: str = "1d") -> pd.Series:
        """
        USD/TRY seri döndürür. Hafta sonu / tatil günleri forward-fill yapılır.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance kurulu değil. 'pip install yfinance' komutunu çalıştırın."
            ) from exc

        cache_path = _CACHE_DIR / f"USDTRY_{interval}.parquet"
        cached = self._load_cache(cache_path, start, end)
        if cached is not None:
            return cached

        logger.info(f"[USDTRYProvider] {_YF_TICKER} çekiliyor {start} → {end}")
        raw = yf.download(
            _YF_TICKER,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"[USDTRYProvider] Veri gelmedi: {start} → {end}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        close = raw["Close"].squeeze()
        close.index = pd.to_datetime(close.index, utc=True)
        close = close.replace(0, np.nan).ffill().bfill()
        close.name = "usdtry"

        if self.cache_enabled:
            close.to_frame().to_parquet(cache_path)

        return close

    def fetch_as_df(self, start: str, end: str) -> pd.DataFrame:
        """DataFrame döndürür: kolonlar = ['date', 'usdtry']"""
        s = self.fetch(start, end)
        return s.reset_index().rename(columns={"index": "date", "Datetime": "date", "Date": "date"})

    def _load_cache(self, path: Path, start: str, end: str) -> pd.Series | None:
        if not self.cache_enabled or not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index, utc=True)
            req_start = pd.Timestamp(start, tz="UTC")
            req_end = pd.Timestamp(end, tz="UTC")
            if s.index.min() <= req_start and s.index.max() >= req_end - pd.Timedelta(days=7):
                logger.debug(f"[USDTRYProvider] Cache hit: {path.name}")
                return s.loc[req_start:req_end]
        except Exception as e:
            logger.warning(f"[USDTRYProvider] Cache okunamadı: {e}")
        return None
