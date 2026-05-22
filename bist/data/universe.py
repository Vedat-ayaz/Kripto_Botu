"""BIST hisse evreni — .IS eki olmadan sembol listeleri."""

BIST_30: list[str] = [
    # Bankalar (6)
    "GARAN", "AKBNK", "ISCTR", "YKBNK", "HALKB", "VAKBN",
    # Holding (3)
    "KCHOL", "SAHOL", "DOHOL",
    # Sanayi / Petrokimya (3)
    "EREGL", "TUPRS", "PETKM",
    # Otomotiv (2)
    "FROTO", "TOASO",
    # Havayolu (2)
    "THYAO", "PGSUS",
    # Perakende / Gıda (3)
    "BIMAS", "MGROS", "ULKER",
    # Telekom (2)
    "TCELL", "TTKOM",
    # Savunma (1)
    "ASELS",
    # İnşaat / Enerji (1)
    "ENKAI",
    # Cam / Kimya (1)
    "SISE",
    # Madencilik / Metal (2)
    "KRDMD", "KOZAL",
    # Beyaz eşya (2)
    "ARCLK", "VESTL",
    # GYO (1)
    "EKGYO",
    # Havalimanı / Turizm (1)
    "TAVHL",
]  # Toplam: 30

BIST_100: list[str] = [
    # Bankalar (10)
    "GARAN", "AKBNK", "ISCTR", "YKBNK", "HALKB",
    "VAKBN", "QNBFB", "TSKB", "ALBRK", "ODEAB",
    # Holding (6)
    "KCHOL", "SAHOL", "DOHOL", "AGHOL", "KOZAL", "ISFIN",
    # Sanayi / Petrokimya (8)
    "EREGL", "TUPRS", "PETKM", "AYGAZ", "GUBRF", "TRKCM", "AKCNS", "HEKTS",
    # Otomotiv / Makine (6)
    "FROTO", "TOASO", "TTRAK", "KARSAN", "DOAS", "MAVI",
    # Havayolu / Turizm (5)
    "THYAO", "PGSUS", "TAVHL", "CLEBI", "UCAK",
    # Perakende / Gıda (8)
    "BIMAS", "MGROS", "ULKER", "CCOLA", "AEFES", "TATGD", "SOKM", "BRYAT",
    # Telekom / Teknoloji (6)
    "TCELL", "TTKOM", "LOGO", "LINK", "DESPC", "NETAS",
    # Savunma (3)
    "ASELS", "ROKET", "KFEIN",
    # İnşaat / GYO (6)
    "ENKAI", "EKGYO", "ISGYO", "TOKER", "KREAS", "YKGYO",
    # Cam / Kimya / Plastik (5)
    "SISE", "CIMSA", "BFREN", "SELVA", "ZOREN",
    # Beyaz eşya / Elektronik (4)
    "ARCLK", "VESTL", "GENTS", "SERVE",
    # Madencilik / Metal (5)
    "KRDMD", "GLYHO", "IPEKE", "SNPAM", "INDES",
    # Sigorta / Finans (6)
    "AKGRT", "RAYSG", "ANHYT", "AVHOL", "KCAER", "SRVGY",
    # Enerji (3)
    "ODAS", "AKSEN", "ENKAI",
    # Diğer (7)
    "SASA", "BRISA", "KARSN", "PRKME", "SKBNK", "OTKAR", "TCMB",
]
# Enerji grubundaki ENKAI duplikatını dedup ile temizle
BIST_100 = list(dict.fromkeys(BIST_100))
# Şu an 87 sembol — 13 tane ek gerçek BIST 100 üyesi ekle
_extra_13 = [
    "NTHOL", "BERA",  "PEGYO", "CANTE", "METRO",
    "TURSG", "ESEN",  "ASUZU", "PARSN", "VKGYO",
    "BANVT", "TEKTU", "AKENR",
]
BIST_100 = BIST_100 + _extra_13
# Final dedup (güvenlik)
BIST_100 = list(dict.fromkeys(BIST_100))

SECTOR_CLUSTERS: dict[str, list[str]] = {
    "banks":     ["GARAN", "AKBNK", "ISCTR", "YKBNK", "HALKB", "VAKBN", "QNBFB", "TSKB", "ALBRK", "ODEAB"],
    "holding":   ["KCHOL", "SAHOL", "DOHOL", "AGHOL", "KOZAL"],
    "industry":  ["EREGL", "TUPRS", "PETKM", "AYGAZ", "GUBRF"],
    "auto":      ["FROTO", "TOASO", "TTRAK", "KARSAN", "DOAS", "OTKAR"],
    "aviation":  ["THYAO", "PGSUS", "TAVHL", "CLEBI"],
    "retail":    ["BIMAS", "MGROS", "ULKER", "CCOLA", "AEFES"],
    "telecom":   ["TCELL", "TTKOM", "LOGO", "NETAS"],
    "defense":   ["ASELS", "ROKET"],
    "reit":      ["EKGYO", "ISGYO", "YKGYO", "TOKER"],
    "glass":     ["SISE", "TRKCM", "CIMSA", "HEKTS"],
    "appliance": ["ARCLK", "VESTL"],
    "mining":    ["KRDMD", "GLYHO", "IPEKE"],
    "insurance": ["AKGRT", "RAYSG", "ANHYT", "AVHOL"],
    "energy":    ["ENKAI", "ODAS", "AKSEN", "ZOREN"],
    "food":      ["TATGD", "SOKM", "BRYAT"],
    "tech":      ["DESPC", "LINK", "LOGO"],
}


def get_universe(size: str = "30") -> list[str]:
    """Sembol listesi döndürür (.IS eki yok)."""
    if size == "100":
        return BIST_100
    return BIST_30


def get_yf_symbols(symbols: list[str]) -> list[str]:
    """Yahoo Finance formatına çevirir: 'GARAN' → 'GARAN.IS'"""
    return [f"{s}.IS" for s in symbols]


def get_sector(symbol: str) -> str | None:
    """Verilen sembolün sektör kümesini döndürür."""
    for sector, members in SECTOR_CLUSTERS.items():
        if symbol in members:
            return sector
    return None
