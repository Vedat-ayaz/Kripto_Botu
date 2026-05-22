import os
import time
import logging
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)


class ExchangeClient:
    """
    ccxt üzerinden borsa bağlantısı sağlar.
    Testnet ve live mod destekler.
    Rate-limit ve bağlantı hatalarında retry yapar.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # saniye

    def __init__(self, exchange_name: str, testnet: bool = True):
        self.exchange_name = exchange_name
        self.testnet = testnet
        self.exchange: Optional[ccxt.Exchange] = None

    def connect(self) -> None:
        """Borsa bağlantısını başlatır."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
        except AttributeError:
            raise ValueError(f"Desteklenmeyen borsa: {self.exchange_name}")

        if self.testnet:
            api_key = os.getenv("TESTNET_API_KEY", "")
            api_secret = os.getenv("TESTNET_API_SECRET", "")
        else:
            api_key = os.getenv("EXCHANGE_API_KEY", "")
            api_secret = os.getenv("EXCHANGE_API_SECRET", "")

        _PLACEHOLDERS = {"your_api_key_here", "your_api_secret_here",
                         "your_testnet_api_key_here", "your_testnet_api_secret_here", ""}

        params: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        # Yalnızca gerçek key varsa ekle; boş veya placeholder değerler borsa tarafından reddedilir
        if api_key and api_key not in _PLACEHOLDERS:
            params["apiKey"] = api_key
        if api_secret and api_secret not in _PLACEHOLDERS:
            params["secret"] = api_secret

        self.exchange = exchange_class(params)

        if self.testnet:
            # Binance testnet URL override
            if self.exchange_name == "binance":
                self.exchange.set_sandbox_mode(True)
            logger.info(f"[ExchangeClient] Testnet modunda bağlandı: {self.exchange_name}")
        else:
            logger.info(f"[ExchangeClient] LIVE modunda bağlandı: {self.exchange_name}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 500) -> list:
        """OHLCV verisi çeker. Rate-limit hatalarında retry yapar."""
        for attempt in range(self.MAX_RETRIES):
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                return data
            except ccxt.RateLimitExceeded as e:
                wait = self.RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limit aşıldı, {wait}s bekleniyor... ({attempt+1}/{self.MAX_RETRIES})")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                wait = self.RETRY_DELAY * (attempt + 1)
                logger.error(f"Ağ hatası: {e}. {wait}s bekleniyor...")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                logger.error(f"Borsa hatası: {e}")
                raise
        raise RuntimeError(f"fetch_ohlcv başarısız: {symbol} {timeframe} ({self.MAX_RETRIES} deneme)")

    def fetch_ticker(self, symbol: str) -> dict:
        """Anlık fiyat ticker'ı döner."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.exchange.fetch_ticker(symbol)
            except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                wait = self.RETRY_DELAY * (attempt + 1)
                logger.warning(f"Ticker hatası, {wait}s bekleniyor: {e}")
                time.sleep(wait)
        raise RuntimeError(f"fetch_ticker başarısız: {symbol}")

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Market emir gönderir. Sadece live modda gerçek emir oluşturur."""
        logger.info(f"[ORDER] {side.upper()} {amount} {symbol} @ market")
        return self.exchange.create_market_order(symbol, side, amount)

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        """Limit emir gönderir."""
        logger.info(f"[ORDER] {side.upper()} {amount} {symbol} @ {price}")
        return self.exchange.create_limit_order(symbol, side, amount, price)

    def fetch_balance(self) -> dict:
        """Bakiye sorgular."""
        return self.exchange.fetch_balance()

    def fetch_open_orders(self, symbol: str) -> list:
        return self.exchange.fetch_open_orders(symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self.exchange.cancel_order(order_id, symbol)

    @property
    def is_connected(self) -> bool:
        return self.exchange is not None
