from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY   = "BUY"
    SELL  = "SELL"
    HOLD  = "HOLD"
    SHORT = "SHORT"       # Açığa satış girişi (bear trend)
    COVER = "COVER"       # Short pozisyon kapatma sinyali


@dataclass
class Signal:
    symbol: str
    side: Side
    reason: str
    timestamp: datetime
    price: float
    confidence_score: float = 0.0  # 0.0 - 1.0 arası, bilgi amaçlı
    atr: Optional[float] = None
    adx: Optional[float] = None
    rsi: Optional[float] = None

    def is_actionable(self) -> bool:
        """BUY, SELL veya SHORT sinyali mi?"""
        return self.side in (Side.BUY, Side.SELL, Side.SHORT)

    def __str__(self) -> str:
        return (
            f"Signal({self.side.value} {self.symbol} @ {self.price:.4f} | "
            f"score={self.confidence_score:.2f} | {self.reason})"
        )
