import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class TelegramNotifier:
    """
    Opsiyonel Telegram bildirim sistemi.
    Kritik bot olaylarını Telegram'a gönderir.
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, enabled: bool = False):
        self.enabled = enabled
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        if self.enabled and (not self.token or not self.chat_id):
            logger.warning("Telegram etkin ama TOKEN veya CHAT_ID eksik. Bildirimler devre dışı.")
            self.enabled = False

    def send(self, message: str) -> bool:
        """Telegram'a mesaj gönderir. Hata durumunda False döner, botu durdurmaz."""
        if not self.enabled:
            return False

        if not _REQUESTS_AVAILABLE:
            logger.warning("Telegram: requests kütüphanesi bulunamadı.")
            return False

        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            resp = _requests.post(self._base_url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram mesajı gönderilemedi: {e}")
            return False

    def bot_started(self, mode: str) -> None:
        self.send(f"🟢 <b>Bot Başladı</b>\nMod: <code>{mode}</code>")

    def new_signal(self, symbol: str, side: str, price: float, reason: str) -> None:
        self.send(
            f"📡 <b>Yeni Sinyal</b>\n"
            f"Parite: <code>{symbol}</code>\n"
            f"Yön: <b>{side}</b>\n"
            f"Fiyat: {price:.4f}\n"
            f"Sebep: {reason}"
        )

    def position_opened(self, symbol: str, entry: float, stop: float, size: float) -> None:
        self.send(
            f"✅ <b>Pozisyon Açıldı</b>\n"
            f"Parite: <code>{symbol}</code>\n"
            f"Giriş: {entry:.4f}\n"
            f"Stop: {stop:.4f}\n"
            f"Miktar: {size:.6f}"
        )

    def position_closed(self, symbol: str, pnl: float, reason: str) -> None:
        emoji = "💚" if pnl >= 0 else "🔴"
        self.send(
            f"{emoji} <b>Pozisyon Kapandı</b>\n"
            f"Parite: <code>{symbol}</code>\n"
            f"PnL: {pnl:+.2f} USDT\n"
            f"Sebep: {reason}"
        )

    def stop_loss_triggered(self, symbol: str, price: float, pnl: float) -> None:
        self.send(
            f"🛑 <b>Stop-Loss Tetiklendi</b>\n"
            f"Parite: <code>{symbol}</code>\n"
            f"Fiyat: {price:.4f}\n"
            f"PnL: {pnl:+.2f} USDT"
        )

    def daily_loss_limit_hit(self, daily_pnl: float, limit: float) -> None:
        self.send(
            f"⚠️ <b>Günlük Zarar Limiti Aşıldı!</b>\n"
            f"Günlük PnL: {daily_pnl:+.2f} USDT\n"
            f"Limit: -{limit:.2f} USDT\n"
            f"İşlemler durduruldu."
        )

    def api_error(self, error: str) -> None:
        self.send(f"❌ <b>API Bağlantı Hatası</b>\n<code>{error[:200]}</code>")
