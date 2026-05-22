import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from execution.position_manager import Position, PositionManager
from risk.risk_manager import RiskManager
from strategy.signal import Signal, Side

if TYPE_CHECKING:
    from data.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Emir oluşturma ve gönderme katmanı.
    Paper modda sanal simülasyon, live modda gerçek API çağrısı yapar.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        live_mode: bool = False,
        exchange_client: Optional["ExchangeClient"] = None,
        trailing_stop_multiplier: float = 2.5,
    ):
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.live_mode = live_mode
        self.client = exchange_client
        self.trailing_stop_multiplier = trailing_stop_multiplier

        if self.live_mode and self.client is None:
            raise ValueError("Live modda ExchangeClient zorunludur!")

    def process_signal(self, signal: Signal, atr: float) -> Optional[Position]:
        """
        BUY sinyalini alır, risk kontrolleri yapar, pozisyon açar.
        Paper modda sanal, live modda gerçek emir.
        """
        if signal.side != Side.BUY:
            return None

        symbol = signal.symbol
        entry_price = signal.price

        # Risk kontrolü
        allowed, reason = self.risk_manager.can_open_trade(
            symbol, entry_price, atr, self.position_manager
        )
        if not allowed:
            logger.info(f"[OrderManager] {symbol} için işlem reddedildi: {reason}")
            return None

        # Pozisyon büyüklüğü ve stop hesapla
        stop_price = self.risk_manager.calculate_stop_price(entry_price, atr)
        trailing_stop = self.risk_manager.calculate_trailing_stop(
            entry_price, atr, self.trailing_stop_multiplier
        )
        position_size = self.risk_manager.calculate_position_size(entry_price, atr)

        if position_size <= 0:
            logger.warning(f"[OrderManager] {symbol}: Hesaplanan pozisyon boyutu sıfır veya negatif.")
            return None

        logger.info(
            f"[OrderManager] {'LIVE' if self.live_mode else 'PAPER'} BUY | "
            f"{symbol} | entry={entry_price:.4f} | size={position_size:.6f} | "
            f"stop={stop_price:.4f} | trailing={trailing_stop:.4f}"
        )

        if self.live_mode:
            success = self._send_live_order(symbol, "buy", position_size, entry_price)
            if not success:
                return None
        # Paper modda sadece logla, gerçek emir gönderme

        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            stop_price=stop_price,
            trailing_stop_price=trailing_stop,
            opened_at=signal.timestamp,
        )
        self.position_manager.open_position(position)
        return position

    def process_pyramid_add(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        thresholds_atr: list[float],
        size_pcts: list[float],
        max_adds: int = 2,
        stop_atr_multiplier: float = 2.0,
    ) -> Optional[Position]:
        """
        Açık pozisyona pyramid lot ekler (Stage 1 — Turtle/Clenow).
        Risk kontrolü + boyut hesabı yapar, gerekirse trailing'i yukarı çeker.
        Döner: güncellenmiş Position veya None (eklenmediyse).
        """
        allowed, level, reason = self.risk_manager.can_pyramid_add(
            symbol, current_price, atr, self.position_manager, thresholds_atr, max_adds,
        )
        if not allowed:
            return None

        pos = self.position_manager.get_position(symbol)
        if pos is None:
            return None

        add_size = self.risk_manager.calculate_pyramid_size(
            pos.initial_size, level, size_pcts,
        )
        if add_size <= 0:
            return None

        # Min order size kontrolü (RiskManager varsa)
        if add_size * current_price < self.risk_manager.min_order_size:
            logger.debug(
                f"[OrderManager] {symbol} pyramid #{level} atlandı: "
                f"size={add_size*current_price:.2f} < min {self.risk_manager.min_order_size}"
            )
            return None

        if self.live_mode:
            success = self._send_live_order(symbol, "buy", add_size, current_price)
            if not success:
                return None

        # Pyramid trailing'i mevcut fiyata göre yenile (yalnız yukarı çeker)
        new_trailing = self.risk_manager.calculate_trailing_stop(
            current_price, atr, self.trailing_stop_multiplier,
        )
        updated = self.position_manager.add_to_position(
            symbol, add_size, current_price,
            new_trailing_stop=new_trailing,
            atr=atr,
            pyramid_stop_atr_multiplier=stop_atr_multiplier,
        )
        if updated:
            logger.info(
                f"[OrderManager] {'LIVE' if self.live_mode else 'PAPER'} PYRAMID #{level} | "
                f"{symbol} | add_size={add_size:.6f} @ {current_price:.4f}"
            )
        return updated

    def close_position(self, symbol: str, current_price: float, reason: str) -> Optional[Position]:
        """Açık pozisyonu kapatır (sinyal bazlı veya manuel)."""
        pos = self.position_manager.get_position(symbol)
        if pos is None:
            return None

        if self.live_mode:
            # Gerçek satış emri gönder
            success = self._send_live_order(symbol, "sell", pos.position_size, current_price)
            if not success:
                logger.error(f"[OrderManager] {symbol} satış emri gönderilemedi!")
                # Pozisyonu yine de kapatalım, kullanıcıyı logdan bilgilendireceğiz

        return self.position_manager.close_position(symbol, current_price, reason)

    def _send_live_order(self, symbol: str, side: str, amount: float, price: float) -> bool:
        """
        Gerçek borsa emri gönderir.
        Partial fill ihtimaline karşı loglama yapılır.
        """
        try:
            order = self.client.create_market_order(symbol, side, amount)
            filled = order.get("filled", 0)
            requested = order.get("amount", amount)

            if filled < requested:
                logger.warning(
                    f"[OrderManager] Partial fill! {symbol} {side} "
                    f"talep={requested:.6f} dolan={filled:.6f}"
                )

            logger.info(
                f"[OrderManager] LIVE order filled | {symbol} {side} "
                f"amount={filled:.6f} | orderId={order.get('id')}"
            )
            return True
        except Exception as e:
            logger.error(f"[OrderManager] {symbol} {side} emri başarısız: {e}")
            return False
