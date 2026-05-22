"""
BIST 100 USD Trend-Following Bot
Kullanım: python bist/bist_main.py --mode [backtest|paper|live|optimize]

Örnekler:
    python bist/bist_main.py --mode backtest
    python bist/bist_main.py --mode backtest --start 2020-01-01 --end 2026-01-01
    python bist/bist_main.py --mode paper
    python bist/bist_main.py --mode live           # gerçek zamanlı paper (yfinance polling)
    python bist/bist_main.py --mode live --interval 5m   # 5 dakikalık bar
    python bist/bist_main.py --mode backtest --config bist/config_bist.yaml
    python bist/bist_main.py --mode optimize --start 2020-01-01 --end 2025-01-01
    WF_SAMPLES=10 python bist/bist_main.py --mode optimize --start 2022-01-01 --end 2024-01-01
"""
import argparse
import logging
import os
import sys

import yaml

# Project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest(cfg: dict, start: str, end: str) -> None:
    from bist.bist_backtester import run_bist_backtest
    run_bist_backtest(cfg, start_date=start, end_date=end)


def run_paper(cfg: dict) -> None:
    from bist.bist_paper_trader import BistPaperTrader
    trader = BistPaperTrader(cfg)
    trader.start()


def run_live(cfg: dict, interval: str = "1d", poll_seconds: int = 900) -> None:
    """
    Gerçek zamanlı sinyal motoru.
    Varsayılan: yfinance polling (15dk gecikmeli, paper modunda).
    Broker API hazır olduğunda StreamingDataSource sub-class'ı buraya geçirilir.
    """
    from bist.live.data_source import YFinancePollingSource
    from bist.live.live_runner import BistLiveRunner

    data_source = YFinancePollingSource(
        interval     = interval,
        poll_seconds = poll_seconds,
        usd_mode     = cfg.get("data", {}).get("usd_mode", "convert_series"),
        cache_enabled = False,   # Live modda cache kullanma
    )
    runner = BistLiveRunner(cfg, data_source=data_source, live_mode=False)
    runner.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="BIST 100 USD Trend-Following Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "optimize"], default="backtest")
    parser.add_argument("--config", default=os.path.join(_ROOT, "bist", "config_bist.yaml"))
    parser.add_argument("--start",    default=None, help="Backtest başlangıç tarihi (YYYY-MM-DD)")
    parser.add_argument("--end",      default=None, help="Backtest bitiş tarihi (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Live mod bar aralığı: 1d, 1h, 5m")
    parser.add_argument("--poll",     default=900,  type=int, help="Live mod polling süresi (saniye)")
    parser.add_argument("--log",      default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {})

    start = args.start or bt_cfg.get("start_date", "2020-01-01")
    end   = args.end   or bt_cfg.get("end_date",   "2026-01-01")

    if args.mode == "backtest":
        run_backtest(cfg, start, end)
    elif args.mode == "paper":
        run_paper(cfg)
    elif args.mode == "live":
        run_live(cfg, interval=args.interval, poll_seconds=args.poll)
    elif args.mode == "optimize":
        from bist.optimizer.walk_forward import run_walk_forward
        from bist.optimizer.config_writer import update_config_with_params

        result = run_walk_forward(
            cfg,
            start=args.start or "2020-01-01",
            end=args.end or "2025-01-01",
            n_samples=int(os.environ.get("WF_SAMPLES", "80")),
        )
        update_config_with_params(args.config, result["best_params"])
        print("\n  Kalibrasyon tamamlandi. Yeni parametrelerle backtest calistirin:")
        print(f"  python bist/bist_main.py --mode backtest --start {args.start} --end {args.end}")


if __name__ == "__main__":
    main()
