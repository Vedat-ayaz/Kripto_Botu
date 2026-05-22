import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.signal import Signal, Side


def make_risk_manager(balance: float = 10_000.0) -> RiskManager:
    return RiskManager(
        account_balance=balance,
        risk_per_trade=0.01,
        daily_max_loss=0.03,
        atr_stop_multiplier=2.0,
        max_open_positions=3,
        min_order_size=10.0,
        max_position_pct=0.20,
    )


def test_manual_close_does_not_record_pnl_twice():
    risk_manager = make_risk_manager()
    position_manager = PositionManager(trailing_stop_atr_multiplier=2.5)
    order_manager = OrderManager(
        position_manager=position_manager,
        risk_manager=risk_manager,
        live_mode=False,
        trailing_stop_multiplier=2.5,
    )

    signal = Signal(
        symbol="BTC/USDT",
        side=Side.BUY,
        reason="test entry",
        timestamp=datetime.now(timezone.utc),
        price=100.0,
    )
    position = order_manager.process_signal(signal, atr=5.0)

    assert position is not None
    assert risk_manager.account_balance == 10_000.0

    closed = order_manager.close_position("BTC/USDT", current_price=110.0, reason="test exit")

    assert closed is not None
    assert closed.realized_pnl == 100.0
    assert risk_manager.account_balance == 10_000.0
    assert risk_manager.daily_pnl == 0.0
