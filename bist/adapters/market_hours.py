"""BIST piyasa saati yönetimi — İstanbul zamanı (UTC+3, DST yok)."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# Türkiye 2016'dan beri yaz saati uygulamıyor → kalıcı UTC+3
BIST_TZ = ZoneInfo("Europe/Istanbul")
BIST_OPEN  = time(10, 0)   # 10:00 TR
BIST_CLOSE = time(18, 0)   # 17:55 matching + 18:00 continuous session kapanışı

# Türk resmi tatilleri (kısmi liste; paper trader guard için)
_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 1),
    date(2024, 4, 10), date(2024, 4, 11), date(2024, 4, 12),
    date(2024, 6, 16), date(2024, 6, 17), date(2024, 6, 18),
    date(2024, 4, 23), date(2024, 5, 1), date(2024, 5, 19),
    date(2024, 7, 15), date(2024, 8, 30), date(2024, 10, 29),
    # 2025
    date(2025, 1, 1),
    date(2025, 3, 30), date(2025, 3, 31), date(2025, 4, 1),
    date(2025, 6, 6), date(2025, 6, 7), date(2025, 6, 8),
    date(2025, 4, 23), date(2025, 5, 1), date(2025, 5, 19),
    date(2025, 7, 15), date(2025, 8, 30), date(2025, 10, 29),
    # 2026
    date(2026, 1, 1),
    date(2026, 3, 20), date(2026, 3, 21), date(2026, 3, 22),
    date(2026, 4, 23), date(2026, 5, 1), date(2026, 5, 19),
    date(2026, 7, 15), date(2026, 8, 30), date(2026, 10, 29),
}


def is_trading_day(d: date) -> bool:
    """Pazartesi–Cuma ve Türk resmi tatili değilse True."""
    return d.weekday() < 5 and d not in _HOLIDAYS


def is_market_open(dt: datetime) -> bool:
    """
    Verilen datetime'da BIST açık mı?
    Naive datetime UTC kabul edilir.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local = dt.astimezone(BIST_TZ)
    if not is_trading_day(local.date()):
        return False
    return BIST_OPEN <= local.time() < BIST_CLOSE


def get_next_open(dt: datetime) -> datetime:
    """Bir sonraki BIST açılış saatini Istanbul zamanında döndürür (tz-aware)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local = dt.astimezone(BIST_TZ)

    candidate = local.replace(hour=10, minute=0, second=0, microsecond=0)
    if local >= candidate.replace(hour=18):
        candidate += timedelta(days=1)
    elif local >= candidate:
        pass  # Bugün piyasa saatindeyiz, aynı gün açılışı bul
    # candidate şimdi bugün 10:00 veya yarın 10:00

    while not is_trading_day(candidate.date()):
        candidate += timedelta(days=1)
        candidate = candidate.replace(hour=10, minute=0, second=0, microsecond=0)

    return candidate


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday DataFrame'den piyasa dışı barları temizler.
    Daily DataFrame'de yalnızca işlem günlerini bırakır.
    Frekansı index'ten otomatik algılar.
    """
    if df.empty:
        return df

    freq = pd.infer_freq(df.index)
    is_intraday = freq is not None and freq not in ("B", "D", "W", "M", "Q", "A", "BMS")

    if is_intraday:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        local = idx.tz_convert(BIST_TZ)
        mask = (
            (local.weekday < 5)
            & pd.Series(local.date, index=df.index).apply(lambda d: d not in _HOLIDAYS).values
            & (local.time >= pd.Timestamp("10:00").time())
            & (local.time < pd.Timestamp("18:00").time())
        )
        return df[mask]
    else:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        local_dates = idx.tz_convert(BIST_TZ).date
        mask = [is_trading_day(d) for d in local_dates]
        return df[mask]
