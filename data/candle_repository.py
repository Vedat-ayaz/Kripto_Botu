from collections import defaultdict
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CandleRepository:
    """
    In-memory mum verisi deposu.
    Her (symbol, timeframe) çifti için son çekilen verileri tutar.
    """

    def __init__(self) -> None:
        # { (symbol, timeframe): pd.DataFrame }
        self._store: dict[tuple, pd.DataFrame] = defaultdict(pd.DataFrame)

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """DataFrame'i depolar."""
        self._store[(symbol, timeframe)] = df.copy()
        logger.debug(f"[CandleRepo] {symbol} {timeframe}: {len(df)} mum güncellendi.")

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Depolanmış DataFrame'i döner, yoksa None."""
        df = self._store.get((symbol, timeframe))
        if df is None or df.empty:
            return None
        return df.copy()

    def has_data(self, symbol: str, timeframe: str) -> bool:
        return (symbol, timeframe) in self._store and not self._store[(symbol, timeframe)].empty

    def clear(self, symbol: Optional[str] = None) -> None:
        if symbol:
            keys = [k for k in self._store if k[0] == symbol]
            for k in keys:
                del self._store[k]
        else:
            self._store.clear()
