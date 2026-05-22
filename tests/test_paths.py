import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.state import BotStateDB
from main import load_config


def test_load_config_uses_project_root_when_cwd_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = load_config()

    assert "exchange" in cfg
    assert "trading" in cfg


def test_bot_state_db_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "state" / "bot.db"

    db = BotStateDB(str(db_path))
    db.update_bot_status(
        mode="TEST",
        running=True,
        account_balance=1000.0,
        initial_balance=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        trading_allowed=True,
    )

    assert db_path.exists()
    status = db.get_bot_status()
    assert status is not None
    assert status["mode"] == "TEST"
