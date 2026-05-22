"""TRY/USD fiyat dönüşüm katmanı."""
import pandas as pd


class PriceConverter:
    """
    BIST hisse fiyatlarını (TRY) USD'ye çevirir.

    mode="convert_series":
        OHLCV / USDTRY oranı → strateji USD bazlı çalışır.
        FX trendi sinyale dahil olur (USD yatırımcı bakışı).

    mode="convert_pnl":
        TRY fiyatları değişmez; strateji TRY bazlı.
        P&L entry/exit anında USD'ye çevrilir.
    """

    def __init__(self, mode: str = "convert_series"):
        if mode not in ("convert_series", "convert_pnl"):
            raise ValueError(f"Geçersiz mod: {mode!r}. 'convert_series' veya 'convert_pnl' olmalı.")
        self.mode = mode

    def convert_ohlcv(
        self,
        df_try: pd.DataFrame,
        usdtry: pd.Series,
    ) -> pd.DataFrame:
        """
        OHLCV DataFrame döndürür.
        convert_series: OHLC kolonları aligned USDTRY'ye bölünür. Volume değişmez.
        convert_pnl: df_try değişmeden döner.

        Hizalama: usdtry, df_try.index'e göre reindex + ffill + bfill.
        """
        if self.mode == "convert_pnl":
            return df_try.copy()

        rate = usdtry.reindex(df_try.index, method="ffill").bfill()

        df_usd = df_try.copy()
        for col in ("open", "high", "low", "close"):
            if col in df_usd.columns:
                df_usd[col] = df_usd[col] / rate

        return df_usd

    def convert_price(self, price_try: float, usdtry_rate: float) -> float:
        """Tek bir TRY fiyatını USD'ye çevirir."""
        if self.mode == "convert_series":
            return price_try / usdtry_rate
        return price_try

    def convert_pnl_to_usd(self, pnl_try: float, usdtry_at_exit: float) -> float:
        """Gerçekleşen TRY P&L'yi USD'ye çevirir (convert_pnl modu için)."""
        return pnl_try / usdtry_at_exit
