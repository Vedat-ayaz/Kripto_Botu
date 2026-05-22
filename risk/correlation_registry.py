"""
Cross-symbol correlation awareness — shared open positions registry.

Per-symbol Backtester'lar bu modülü import edip aynı dict'i paylaşır.
Böylece "BTC ve 4 alt aynı anda LONG" gibi yüksek-korelasyon durumlarında
yeni alt entry'leri otomatik küçültülür → MaxDD düşer.

Akademik kaynak: Carver, Systematic Trading (2015), Ch. 4 — Instrument
Diversification Multiplier (IDM). Crypto'da BTC-alt korelasyon ~0.7-0.9,
8 paralel pozisyon ≈ 2-3 etkin (independent) pozisyon.

MVP yaklaşımı (heuristic): rolling correlation matrix yerine tek skaler
"açık alt sayısı" kullan. Yeterli kanıt elde edilince Carver IDM'e geç.
"""
from typing import Dict

# Symbol → True (açık pozisyon var) eşlemesi.
# Backtester her open/close'da günceller. Per-symbol backtester'lar
# aynı modül-seviye dict'i paylaşır (process-level shared state).
_open_positions: Dict[str, bool] = {}


def register_open(symbol: str) -> None:
    """Pozisyon açıldığında çağrılır."""
    _open_positions[symbol] = True


def register_close(symbol: str) -> None:
    """Pozisyon kapandığında çağrılır."""
    _open_positions.pop(symbol, None)


def get_open_symbols() -> list[str]:
    """Şu an açık pozisyona sahip semboller."""
    return [s for s, v in _open_positions.items() if v]


def open_count() -> int:
    """Açık pozisyon sayısı (registry'den)."""
    return len(_open_positions)


def reset() -> None:
    """Backtest başlangıcında çağrılır — registry'yi temizle."""
    _open_positions.clear()
