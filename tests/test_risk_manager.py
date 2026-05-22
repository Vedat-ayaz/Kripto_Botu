import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from risk.risk_manager import RiskManager
from execution.position_manager import PositionManager, Position


def make_position_manager() -> PositionManager:
    return PositionManager(trailing_stop_atr_multiplier=2.5)


def make_risk_manager(balance: float = 10_000.0) -> RiskManager:
    return RiskManager(
        account_balance=balance,
        risk_per_trade=0.01,
        daily_max_loss=0.03,
        atr_stop_multiplier=2.0,
        max_open_positions=3,
        min_order_size=10.0,
    )


class TestRiskManager:
    def test_can_open_trade_basic(self):
        rm = make_risk_manager()
        pm = make_position_manager()
        allowed, reason = rm.can_open_trade("BTC/USDT", 50_000.0, 1_000.0, pm)
        assert allowed is True
        assert reason == "OK"

    def test_rejects_when_max_positions_reached(self):
        rm = make_risk_manager()
        pm = make_position_manager()
        for i in range(3):
            pm.open_position(Position(
                symbol=f"SYM{i}/USDT",
                entry_price=100.0,
                position_size=1.0,
                stop_price=90.0,
                trailing_stop_price=90.0,
            ))
        allowed, reason = rm.can_open_trade("NEW/USDT", 100.0, 5.0, pm)
        assert allowed is False
        assert "Maksimum" in reason

    def test_rejects_duplicate_symbol(self):
        rm = make_risk_manager()
        pm = make_position_manager()
        pm.open_position(Position(
            symbol="BTC/USDT",
            entry_price=50_000.0,
            position_size=0.01,
            stop_price=48_000.0,
            trailing_stop_price=48_000.0,
        ))
        allowed, reason = rm.can_open_trade("BTC/USDT", 51_000.0, 1_000.0, pm)
        assert allowed is False
        assert "açık pozisyon" in reason

    def test_daily_loss_limit_stops_trading(self):
        rm = make_risk_manager(10_000.0)
        # 3% = 300 USDT limit
        rm.record_trade_pnl(-350.0)
        assert rm.trading_allowed is False

        pm = make_position_manager()
        allowed, _ = rm.can_open_trade("BTC/USDT", 100.0, 5.0, pm)
        assert allowed is False

    def test_daily_pnl_reset(self):
        rm = make_risk_manager()
        rm.record_trade_pnl(-350.0)
        assert rm.trading_allowed is False
        rm.reset_daily_pnl()
        assert rm.trading_allowed is True
        assert rm.daily_pnl == 0.0

    def test_position_size_formula(self):
        rm = make_risk_manager(10_000.0)
        entry = 50_000.0
        atr = 1_000.0
        # ATR sizing: risk=100, stop_dist=2000 → raw=0.05 coin → value=2500 USDT
        # Cap: 10000 * 0.20 / 50000 = 0.04 coin → value=2000 USDT (limit devreye girer)
        size = rm.calculate_position_size(entry, atr)
        assert abs(size - 0.04) < 1e-9  # sermaye limiti ATR sizing'i kıstı

    def test_position_size_atr_binding(self):
        # ATR sizing, sermaye limitinden küçük olduğunda ATR sizing geçerli
        rm = make_risk_manager(10_000.0)
        entry = 100.0
        atr = 2.0
        # stop_dist = 4.0, risk=100, raw_size=25 coin, value=2500 USDT
        # cap = 10000*0.20/100 = 20 coin, value=2000 USDT (limit devreye girer)
        # Fiyat çok küçük olduğu için limit yine devreye girecek
        # Farklı senaryo: ATR büyük → raw_size küçük → ATR binding
        entry2 = 100.0
        atr2 = 20.0
        # stop_dist = 40.0, risk=100, raw=2.5 coin, value=250 USDT
        # cap = 20 coin → ATR sizing geçerli
        size2 = rm.calculate_position_size(entry2, atr2)
        assert abs(size2 - 2.5) < 1e-9  # ATR sizing geçerli, limit devreye girmedi

    def test_stop_price_formula(self):
        rm = make_risk_manager()
        stop = rm.calculate_stop_price(50_000.0, 1_000.0)
        assert stop == 48_000.0

    def test_rejects_tiny_order(self):
        # risk_amount = 50 * 0.01 = 0.5 USDT
        # stop_distance = 1.0 * 2.0 = 2.0 USDT
        # size = 0.5 / 2.0 = 0.25 coin
        # order_value = 0.25 * 1.0 = 0.25 USDT  <  10 USDT (min)
        rm = RiskManager(
            account_balance=50.0,
            risk_per_trade=0.01,
            daily_max_loss=0.03,
            atr_stop_multiplier=2.0,
            max_open_positions=3,
            min_order_size=10.0,
        )
        pm = make_position_manager()
        allowed, reason = rm.can_open_trade("SYM/USDT", 1.0, 1.0, pm)
        assert allowed is False
        assert "minimumun" in reason
