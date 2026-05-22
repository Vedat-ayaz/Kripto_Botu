"""BIST lot ve tick boyutu yönetimi."""


def get_tick_size(price: float) -> float:
    """BIST minimum fiyat adımı. 2022 sonrası birleşik kural: 0.01 TL."""
    return 0.01


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """Fiyatı geçerli tick boyutuna yuvarlar."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def round_to_lot(quantity_float: float) -> int:
    """
    Tam lot'a (tam adet) aşağı yuvarlar. BIST minimum lot = 1 adet.
    0.0'dan küçük sonuç için 0 döner.
    """
    return max(0, int(quantity_float))


def calculate_position_value_try(quantity: int, price_try: float) -> float:
    """TRY cinsinden pozisyon değeri."""
    return quantity * price_try


def calculate_commission(value_try: float, commission_rate: float = 0.0005) -> float:
    """
    Toplam komisyon hesaplar (BSMV dahil).
    commission_rate: broker oranı per side (varsayılan %0.05).
    Geri döndürür: toplam komisyon TRY.
    """
    commission = value_try * commission_rate
    bsmv = commission * 0.05
    borsa_payi = value_try * 0.0000001
    return commission + bsmv + borsa_payi


def calculate_round_trip_cost(value_try: float, commission_rate: float = 0.0005) -> float:
    """Alış + Satış toplam round-trip maliyet (TRY)."""
    return calculate_commission(value_try, commission_rate) * 2
