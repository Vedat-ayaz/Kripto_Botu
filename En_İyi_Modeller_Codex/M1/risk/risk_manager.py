import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from execution.position_manager import PositionManager

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Her emir açılmadan önce risk kontrollerini yapar.
    Pozisyon büyüklüğü hesaplar, günlük zarar takibi yapar.
    """

    def __init__(
        self,
        account_balance: float,
        risk_per_trade: float = 0.01,
        daily_max_loss: float = 0.03,
        atr_stop_multiplier: float = 2.0,
        max_open_positions: int = 3,
        min_order_size: float = 10.0,
        max_position_pct: float = 0.20,
    ):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.daily_max_loss = daily_max_loss
        self.atr_stop_multiplier = atr_stop_multiplier
        self.max_open_positions = max_open_positions
        self.min_order_size = min_order_size
        # Tek pozisyonun bakiyeye oranı üst sınırı (Clenow/Carver standart yaklaşımı)
        self.max_position_pct = max_position_pct

        self._daily_pnl: float = 0.0
        self._trading_allowed: bool = True
        # Günlük zarar limitini gün-başı bakiyeye göre hesapla (kayan hedef hatası önlenir).
        # Eski kod: limit = account_balance * daily_max_loss  → bakiye düştükçe limit de küçülüyor
        # Yeni kod: limit = _day_start_balance * daily_max_loss → sabit gün limiti
        self._day_start_balance: float = account_balance

    # ------------------------------------------------------------------ #
    #  Günlük PnL takibi
    # ------------------------------------------------------------------ #

    def record_trade_pnl(self, pnl: float) -> None:
        """Kapanan bir işlemin PnL'ini günlük sayaca ekler."""
        self._daily_pnl += pnl
        self.account_balance += pnl
        logger.info(f"[RiskManager] Trade PnL: {pnl:+.4f} | Günlük: {self._daily_pnl:+.4f} | Bakiye: {self.account_balance:.2f}")

        if self._is_daily_loss_exceeded():
            self._trading_allowed = False
            logger.warning(
                f"[RiskManager] GÜNLÜK ZARAR LİMİTİ AŞILDI! "
                f"Günlük PnL: {self._daily_pnl:+.4f} | "
                f"Limit: -{self.account_balance * self.daily_max_loss:.4f}"
            )

    def reset_daily_pnl(self) -> None:
        """Her gün başında çağrılır. Gün-başı bakiyeyi günceller."""
        logger.info(f"[RiskManager] Günlük PnL sıfırlandı. Önceki: {self._daily_pnl:+.4f}")
        self._daily_pnl = 0.0
        self._trading_allowed = True
        # Zarar limiti hesabı için gün-başı bakiyeyi güncelle
        self._day_start_balance = self.account_balance

    # ------------------------------------------------------------------ #
    #  Ana kontrol: can_open_trade
    # ------------------------------------------------------------------ #

    def can_open_trade(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        position_manager: "PositionManager",
    ) -> tuple[bool, str]:
        """
        Yeni işlem açılabilir mi?
        (allowed, reason) döner.
        """
        if not self._trading_allowed:
            return False, "Günlük zarar limiti aşıldı, işlemler durduruldu"

        if self._is_daily_loss_exceeded():
            self._trading_allowed = False
            return False, "Günlük zarar limiti aşıldı"

        if position_manager.open_count >= self.max_open_positions:
            return False, f"Maksimum açık pozisyon sayısına ulaşıldı ({self.max_open_positions})"

        if position_manager.has_open_position(symbol):
            return False, f"{symbol} için zaten açık pozisyon var"

        stop_price = self.calculate_stop_price(entry_price, atr)
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return False, "Stop fiyatı hesaplanamadı (ATR sıfır?)"

        position_size = self.calculate_position_size(entry_price, atr)
        order_value = position_size * entry_price
        if order_value < self.min_order_size:
            return False, f"Emir büyüklüğü minimumun altında ({order_value:.2f} < {self.min_order_size})"

        return True, "OK"

    # ------------------------------------------------------------------ #
    #  Pozisyon büyüklüğü ve stop hesaplama
    # ------------------------------------------------------------------ #

    def calculate_stop_price(self, entry_price: float, atr: float) -> float:
        """ATR tabanlı stop fiyatı."""
        return entry_price - atr * self.atr_stop_multiplier

    def calculate_position_size(self, entry_price: float, atr: float) -> float:
        """
        Literatür standardı: ATR-tabanlı risk sizing + sermaye tahsis limiti.
        (Clenow "Following the Trend", Carver "Systematic Trading")

        1. ATR risk sizing:
           risk_amount = balance * risk_per_trade
           raw_size    = risk_amount / (ATR * stop_multiplier)

        2. Sermaye limiti (tek pozisyon bakiyenin max_position_pct'ini geçemez):
           max_size = (balance * max_position_pct) / entry_price

        3. final_size = min(raw_size, max_size)

        NOT — Volatilite Hedefleme devre dışı:
        ATR-tabanlı sizing zaten volatiliteyi hesaba katar (büyük ATR → küçük lot).
        Kripto'da yüksek vol = yüksek getiri potansiyeli (momentum taşıyıcısı);
        ayrı bir vol-targeting eklenmesi yüksek-getirili altcoinleri kalıcı
        olarak penalize ettiğinden kaldırıldı. (Denendi: SOL/INJ/FET 0.70× ile
        sabitlenip toplam getiri %164→%117'ye geriledi.)
        """
        risk_amount = self.account_balance * self.risk_per_trade
        stop_price = self.calculate_stop_price(entry_price, atr)
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit == 0:
            return 0.0

        raw_size = risk_amount / risk_per_unit
        max_size = (self.account_balance * self.max_position_pct) / entry_price
        final_size = min(raw_size, max_size)

        if final_size < raw_size:
            logger.debug(
                f"[RiskManager] Pozisyon sermaye limitiyle kısıtlandı: "
                f"ATR-sizing={raw_size:.6f} → cap={final_size:.6f} "
                f"(max %{self.max_position_pct*100:.0f} bakiye)"
            )

        return final_size

    def can_pyramid_add(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        position_manager: "PositionManager",
        thresholds_atr: list[float],
        max_adds: int = 2,
    ) -> tuple[bool, int, str]:
        """
        Açık pozisyona pyramid ekleme yapılabilir mi?
        Turtle Trading mantığı: kâr seviyeleri (ATR cinsinden) aşıldıkça lot ekle.

        Döner: (allowed, target_add_level, reason)
          target_add_level: 1-indexed → 1=ilk pyramid, 2=ikinci pyramid
          allowed=False ise target_add_level=0
        """
        if not self._trading_allowed:
            return False, 0, "trading durduruldu"

        pos = position_manager.get_position(symbol)
        if pos is None:
            return False, 0, "açık pozisyon yok"

        if pos.pyramid_adds_count >= max_adds:
            return False, 0, f"max pyramid sayısına ulaşıldı ({max_adds})"

        if atr <= 0:
            return False, 0, "ATR sıfır"

        # Bir sonraki tetikleyici eşik (sıradaki seviye)
        next_level = pos.pyramid_adds_count + 1  # 1, 2, ...
        if next_level > len(thresholds_atr):
            return False, 0, "tetikleyici eşik tanımsız"

        required_atr_gain = thresholds_atr[next_level - 1]
        # Referans: İLK giriş fiyatı (Turtle standardı — son lot fiyatı değil)
        # Bu, kümülatif trendin gerçekten ilerlediğinden emin olmamızı sağlar.
        profit_per_unit = current_price - pos.entry_price
        required_profit = atr * required_atr_gain

        if profit_per_unit < required_profit:
            return False, 0, (
                f"yetersiz kâr: {profit_per_unit:.4f} < gerekli {required_profit:.4f} "
                f"({required_atr_gain}×ATR)"
            )

        return True, next_level, "ok"

    def should_partial_exit(
        self,
        symbol: str,
        current_price: float,
        position_manager: "PositionManager",
        r_multiple_levels: list[float],
        max_exits: int = 2,
    ) -> tuple[bool, int, float, str]:
        """
        Açık pozisyonda R-multiple kâr eşiği aşıldı mı?
        Stage 2 — Partial Exits: kazananı kademeli realize et.

        R = (current_price - entry_price) / (entry_price - initial_stop_price)
        R-multiple kâr birim risk başına ne kadar kazanç sağlandığını ölçer.

        Döner: (allowed, next_level, current_R, reason)
          next_level: 1-indexed → 1=1. realize, 2=2. realize, vs.
          allowed=False ise next_level=0
        """
        pos = position_manager.get_position(symbol)
        if pos is None:
            return False, 0, 0.0, "açık pozisyon yok"

        if pos.partial_exits_done >= max_exits:
            return False, 0, 0.0, f"max partial exit sayısına ulaşıldı ({max_exits})"

        # R (initial risk per unit)
        initial_risk = pos.entry_price - pos.initial_stop_price
        if initial_risk <= 0:
            return False, 0, 0.0, "geçersiz initial risk (entry <= stop)"

        current_r = (current_price - pos.entry_price) / initial_risk

        next_level = pos.partial_exits_done + 1  # 1, 2, ...
        if next_level > len(r_multiple_levels):
            return False, 0, current_r, "eşik tanımsız"

        required_r = r_multiple_levels[next_level - 1]
        if current_r < required_r:
            return False, 0, current_r, (
                f"yetersiz kâr: {current_r:.2f}R < gerekli {required_r:.2f}R"
            )

        return True, next_level, current_r, "ok"

    def calculate_pyramid_size(
        self,
        initial_size: float,
        add_level: int,
        size_pcts: list[float],
    ) -> float:
        """
        Pyramid lot boyutu = initial_size × size_pcts[level-1]
        Örnek: size_pcts=[0.5, 0.25] → 1. add %50, 2. add %25 (toplam %175)
        """
        if add_level < 1 or add_level > len(size_pcts):
            return 0.0
        pct = size_pcts[add_level - 1]
        return initial_size * pct

    def calculate_trailing_stop(
        self,
        entry_price: float,
        atr: float,
        trailing_multiplier: float,
    ) -> float:
        """İlk trailing stop fiyatını hesaplar (entry_price - atr * trailing_multiplier)."""
        return entry_price - atr * trailing_multiplier

    # ------------------------------------------------------------------ #
    #  Yardımcılar
    # ------------------------------------------------------------------ #

    def _is_daily_loss_exceeded(self) -> bool:
        # Gün-başı bakiyeye göre hesapla (kayan hedef hatası önlendi).
        # Eski: account_balance * daily_max_loss → bakiye düştükçe limit küçülüyor,
        #       birkaç küçük kayıptan sonra limit 0'a yaklaşıp trading duraklıyor.
        # Yeni: _day_start_balance → gün boyunca sabit limit.
        limit = self._day_start_balance * self.daily_max_loss
        return self._daily_pnl < -limit

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trading_allowed(self) -> bool:
        return self._trading_allowed

    def summary(self) -> dict:
        return {
            "account_balance": self.account_balance,
            "daily_pnl": self._daily_pnl,
            "trading_allowed": self._trading_allowed,
            "day_start_balance": self._day_start_balance,
            "daily_loss_limit": self._day_start_balance * self.daily_max_loss,
        }
