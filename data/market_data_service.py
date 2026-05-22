import logging
from typing import Optional
import pandas as pd

from .exchange_client import ExchangeClient
from .candle_repository import CandleRepository

logger = logging.getLogger(__name__)


def ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    """ccxt'den gelen raw OHLCV listesini DataFrame'e çevirir."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    # Son mum henüz kapanmamış olabilir, onu düşürüyoruz
    df = df.iloc[:-1]
    return df


class MarketDataService:
    """
    Borsa üzerinden OHLCV verisi çeker ve CandleRepository'e kaydeder.
    Backtest modunda harici DataFrame da beslenebilir.
    """

    def __init__(self, client: Optional[ExchangeClient], repo: CandleRepository):
        self.client = client
        self.repo = repo

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Borsadan veri çeker, temizler ve depoya kaydeder."""
        if self.client is None:
            raise RuntimeError("ExchangeClient bağlı değil.")
        raw = self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not raw:
            raise ValueError(f"Veri boş döndü: {symbol} {timeframe}")
        df = ohlcv_to_dataframe(raw)
        self.repo.save(symbol, timeframe, df)
        logger.info(f"[MarketData] {symbol} {timeframe}: {len(df)} mum alındı.")
        return df

    def get_candles(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Depodaki son mum verisini döner."""
        return self.repo.get(symbol, timeframe)

    def load_from_dataframe(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Backtest için harici DataFrame doğrudan besler."""
        self.repo.save(symbol, timeframe, df)
        logger.info(f"[MarketData] {symbol} {timeframe}: {len(df)} mum dışarıdan yüklendi.")
