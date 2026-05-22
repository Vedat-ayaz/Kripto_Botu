from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    symbol: str
    entry_price: float           # İLK giriş fiyatı (BE/stop trigger referansı, değişmez)
    position_size: float         # TOPLAM coin miktarı (pyramid eklemelerle artar)
    stop_price: float            # ATR tabanlı sabit stop
    trailing_stop_price: float   # trailing stop
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    close_price: Optional[float] = None
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None

    # ── Pyramiding (Turtle/Clenow) alanları ────────────────────────────────
    # initial_size: ilk lot büyüklüğü (pyramid adds'ları boyutlandırmak için referans)
    # avg_entry_price: ağırlıklı ortalama giriş — PnL hesabı için kullanılır
    # pyramid_adds_count: kaç ek lot eklendi (max 2 öneriliyor)
    # last_add_price: son eklemenin fiyatı (sonraki tetikleyici için referans)
    initial_size: float = 0.0
    avg_entry_price: float = 0.0
    pyramid_adds_count: int = 0
    last_add_price: float = 0.0

    # ── Partial Exits (R-multiple kademeli realize) — Stage 2 ──────────────
    # initial_stop_price: ilk giriş anındaki stop (R hesabı için DEĞİŞMEZ referans)
    # partial_exits_done: kaç kademeli realize yapıldı
    # realized_pnl_partial: partial exit'lerden gelen kümülatif kâr (raporlama için)
    initial_stop_price: float = 0.0
    partial_exits_done: int = 0
    realized_pnl_partial: float = 0.0

    def __post_init__(self) -> None:
        # Geriye dönük uyumluluk: yeni alanlar verilmediyse entry'den türet
        if self.initial_size == 0.0:
            self.initial_size = self.position_size
        if self.avg_entry_price == 0.0:
            self.avg_entry_price = self.entry_price
        if self.last_add_price == 0.0:
            self.last_add_price = self.entry_price
        # R hesabı için orijinal stop'u sakla (stop_price BE/trailing ile değişir)
        if self.initial_stop_price == 0.0:
            self.initial_stop_price = self.stop_price

    @property
    def cost_basis(self) -> float:
        """Pozisyonun toplam USDT maliyeti (ağırlıklı ortalama × toplam adet)."""
        return self.avg_entry_price * self.position_size

    def update_unrealized_pnl(self, current_price: float) -> None:
        # Pyramiding ile avg_entry_price hareketli olur → ağırlıklı ortalamayı kullan
        self.unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size

    def add_pyramid_lot(self, add_size: float, add_price: float) -> None:
        """
        Pyramid ekleme: mevcut pozisyona yeni lot ekler.
        Ağırlıklı ortalama giriş fiyatını yeniler.
        entry_price (İLK giriş) değişmez — BE trigger ve stop referansı olarak korunur.
        """
        new_total = self.position_size + add_size
        if new_total <= 0:
            return
        self.avg_entry_price = (
            (self.avg_entry_price * self.position_size) + (add_price * add_size)
        ) / new_total
        self.position_size = new_total
        self.pyramid_adds_count += 1
        self.last_add_price = add_price

    def update_trailing_stop(
        self,
        current_price: float,
        atr: float,
        multiplier: float,
    ) -> bool:
        """
        Fiyat yükselirse trailing stop'u yukarı çeker.
        ATR tabanlı: new_trailing = current_price - atr * multiplier.
        Yeni stop eski stop'tan düşük olursa güncelleme yapılmaz.
        """
        new_trailing = current_price - atr * multiplier
        if new_trailing > self.trailing_stop_price:
            self.trailing_stop_price = new_trailing
            return True
        return False

    def close(self, price: float, reason: str) -> None:
        self.close_price = price
        self.closed_at = datetime.now(timezone.utc)
        self.status = PositionStatus.CLOSED
        # Pyramid pozisyonlarda PnL = (exit - avg_entry) × toplam_size
        self.realized_pnl = (price - self.avg_entry_price) * self.position_size
        self.unrealized_pnl = 0.0
        self.close_reason = reason
        pyramid_note = f" | pyramids={self.pyramid_adds_count}" if self.pyramid_adds_count else ""
        logger.info(
            f"[Position] CLOSED {self.symbol} | entry={self.entry_price:.4f} "
            f"avg={self.avg_entry_price:.4f} close={price:.4f} | "
            f"pnl={self.realized_pnl:+.4f} USDT | reason={reason}{pyramid_note}"
        )


class PositionManager:
    """
    Açık pozisyonları takip eder.
    Stop-loss ve trailing stop kontrolü de burada yapılır.
    """

    def __init__(self, trailing_stop_atr_multiplier: float = 2.5):
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self._positions: dict[str, Position] = {}

    def open_position(self, position: Position) -> None:
        self._positions[position.symbol] = position
        logger.info(
            f"[PositionManager] OPEN {position.symbol} | "
            f"entry={position.entry_price:.4f} | size={position.position_size:.6f} | "
            f"stop={position.stop_price:.4f} | trailing={position.trailing_stop_price:.4f}"
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        pos = self._positions.get(symbol)
        return pos if pos and pos.status == PositionStatus.OPEN else None

    def has_open_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    def update_positions(self, current_prices: dict[str, float], atrs: dict[str, float]) -> list[tuple[Position, str]]:
        """
        Tüm açık pozisyonları günceller.
        Stop tetiklenirse pozisyonu CLOSED olarak işaretler.
        Döner: [(position, close_reason), ...] — bu turda kapananlar
        """
        closed = []
        for symbol, pos in list(self._positions.items()):
            if pos.status != PositionStatus.OPEN:
                continue

            price = current_prices.get(symbol)
            if price is None:
                continue

            atr = atrs.get(symbol, 0.0)
            pos.update_unrealized_pnl(price)
            pos.update_trailing_stop(price, atr, self.trailing_stop_atr_multiplier)

            # Breakeven stop: fiyat 2×ATR kazanç sağladıysa, sabit stop'u
            # avg_entry'ye taşı → "yukarı gidip geri dönen" işlemler sıfırda kapanır.
            # 1×ATR çok erken tetiklenip kazanan işlemleri kesiyor;
            # 2×ATR trende gerçekten girmiş olduğumuzu garanti eder.
            # Pyramid pozisyonlarda avg_entry kullan → BE gerçekten birleşik
            # pozisyon için sıfır olur (ilk girişten daha geç tetiklenir).
            be_ref = pos.avg_entry_price
            if atr > 0 and (price - be_ref) >= atr * 2.0:
                be_stop = be_ref * 1.003  # komisyon + spread tamponu
                if be_stop > pos.stop_price:
                    pos.stop_price = be_stop
                    logger.debug(
                        f"[PositionManager] {symbol} breakeven stop aktif: {be_stop:.4f} "
                        f"(+2×ATR tetikledi, kâr={price - be_ref:.4f}, "
                        f"pyramids={pos.pyramid_adds_count})"
                    )

            # ATR tabanlı sabit stop
            if price <= pos.stop_price:
                pos.close(price, "stop_loss")
                closed.append((pos, "stop_loss"))
                continue

            # Trailing stop
            if price <= pos.trailing_stop_price:
                pos.close(price, "trailing_stop")
                closed.append((pos, "trailing_stop"))
                continue

        return closed

    def add_to_position(
        self,
        symbol: str,
        add_size: float,
        add_price: float,
        new_trailing_stop: Optional[float] = None,
        atr: Optional[float] = None,
        pyramid_stop_atr_multiplier: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Mevcut açık pozisyona pyramid lot ekler.
        - position_size artar, avg_entry_price güncellenir
        - entry_price (İLK giriş) DEĞİŞMEZ → BE/stop referansı korunur
        - new_trailing_stop verilirse trailing aşağı çekilmez (yalnız yukarı çekilir)
        - pyramid_stop_atr_multiplier verilirse stop yukarı çekilir
          (Turtle kuralı: stop = max(stop, add_price − mult × ATR))
          Pyramid'in büyük kaybı önleyen en kritik adımı.
        Döner: güncellenen Position veya None (açık pozisyon yoksa)
        """
        pos = self.get_position(symbol)
        if pos is None or add_size <= 0:
            return None

        old_size = pos.position_size
        old_avg = pos.avg_entry_price
        old_stop = pos.stop_price
        pos.add_pyramid_lot(add_size, add_price)

        # Pyramid ekleme trailing'i yukarı çekebilir (current price yüksek)
        # — fakat asla aşağı çekme. Aşağı çekme update_trailing_stop tarafından zaten engelli.
        if new_trailing_stop is not None and new_trailing_stop > pos.trailing_stop_price:
            pos.trailing_stop_price = new_trailing_stop

        # ── Stop-Tightening (Pyramid Risk Yönetimi) ──────────────────────────
        # Her pyramid sonrası iki kural birlikte uygulanır, en koruyucu seçilir:
        #
        #   A) Turtle: stop = add_price − mult × ATR (kâr arttıkça stop yukarı)
        #   B) BE+ floor: stop = avg_entry × 1.003 (pyramid'li trade asla loss olmaz)
        #
        # NOT: Kademeli (BE+ sadece pyramid #2'den) varyant denendi → +126%
        # tek-aşamalı varyant +160% verdi. Tek-aşamalı korumayı seçtik:
        # pyramid'li trade'in kayıp riski tamamen elimine olur.
        new_stop_candidates = [pos.stop_price]
        if pyramid_stop_atr_multiplier is not None and atr is not None and atr > 0:
            turtle_stop = add_price - pyramid_stop_atr_multiplier * atr
            new_stop_candidates.append(turtle_stop)
        be_floor = pos.avg_entry_price * 1.003
        new_stop_candidates.append(be_floor)
        pos.stop_price = max(new_stop_candidates)

        logger.info(
            f"[PositionManager] PYRAMID ADD #{pos.pyramid_adds_count} {symbol} | "
            f"add_size={add_size:.6f} @ {add_price:.4f} | "
            f"size {old_size:.6f}→{pos.position_size:.6f} | "
            f"avg_entry {old_avg:.4f}→{pos.avg_entry_price:.4f} | "
            f"stop {old_stop:.4f}→{pos.stop_price:.4f} | "
            f"trailing={pos.trailing_stop_price:.4f}"
        )
        return pos

    def partial_close(
        self,
        symbol: str,
        exit_pct: float,
        exit_price: float,
        reason: str,
    ) -> Optional[tuple[Position, float, float]]:
        """
        Pozisyonun bir kısmını kapatır (R-multiple kademeli realize).
        - exit_pct: mevcut size'ın hangi oranı kapanacak (0.0 - 1.0)
        - exit_price: çıkış fiyatı (slippage uygulanmış)
        - Pozisyon AÇIK kalır, sadece size azalır
        - PnL = (exit_price - avg_entry_price) × exit_size

        Döner: (position, exit_size, partial_pnl) veya None
        """
        pos = self.get_position(symbol)
        if pos is None or exit_pct <= 0 or exit_pct >= 1.0:
            return None

        exit_size = pos.position_size * exit_pct
        if exit_size <= 0:
            return None

        # partial_pnl GROSS — komisyon caller tarafından düşülür ve net realized_pnl_partial
        # alanına yazılır. Bu, gross/net ayrımını tek noktada tutar.
        partial_pnl = (exit_price - pos.avg_entry_price) * exit_size
        pos.position_size -= exit_size
        pos.partial_exits_done += 1

        logger.info(
            f"[PositionManager] PARTIAL EXIT #{pos.partial_exits_done} {symbol} | "
            f"exit_pct={exit_pct:.0%} size={exit_size:.6f} @ {exit_price:.4f} | "
            f"partial_pnl={partial_pnl:+.4f} | kalan size={pos.position_size:.6f}"
        )
        return pos, exit_size, partial_pnl

    def close_position(self, symbol: str, price: float, reason: str) -> Optional[Position]:
        """Manuel pozisyon kapatma (sinyal bazlı çıkış)."""
        pos = self.get_position(symbol)
        if pos:
            pos.close(price, reason)
            return pos
        return None

    def report_open_positions(self) -> None:
        """Bot kapanmadan önce açık pozisyonları loglar."""
        if not self.open_positions:
            logger.info("[PositionManager] Açık pozisyon yok.")
            return
        logger.warning(f"[PositionManager] {self.open_count} açık pozisyon var!")
        for pos in self.open_positions:
            logger.warning(
                f"  -> {pos.symbol} | entry={pos.entry_price:.4f} | "
                f"size={pos.position_size:.6f} | upnl={pos.unrealized_pnl:+.4f}"
            )

    def all_closed_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == PositionStatus.CLOSED]
