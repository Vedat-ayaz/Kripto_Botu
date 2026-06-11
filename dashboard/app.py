"""
Kripto Bot - Mission Control
Calistirma: streamlit run dashboard/app.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.state import BotStateDB

# UNIVERSE coin listesi — market ekranı için (ccxt kurulu olmasa da fallback)
try:
    from crypto_portfolio_test import UNIVERSE as _CRYPTO_UNIVERSE
except Exception:
    _CRYPTO_UNIVERSE = [
        "BNB/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "TRX/USDT",
        "DOT/USDT", "LINK/USDT", "LTC/USDT", "ATOM/USDT",
        "NEAR/USDT", "UNI/USDT", "APT/USDT", "INJ/USDT",
        "FET/USDT", "ARB/USDT", "OP/USDT", "ETC/USDT",
        "HBAR/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT",
        "SUI/USDT", "TIA/USDT", "TON/USDT", "JUP/USDT", "WIF/USDT",
    ]


st.set_page_config(
    page_title="Kripto Bot Mission Control",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PALETTE = {
    "bg": "#f4f7fb",
    "bg_alt": "#ecf2f9",
    "card": "#ffffff",
    "card_alt": "#f8fbff",
    "ink": "#10233f",
    "muted": "#667892",
    "border": "#d7e2ef",
    "primary": "#0f6fff",
    "primary_soft": "#e8f0ff",
    "success": "#12b886",
    "success_soft": "#e7faf4",
    "warning": "#ff9f1c",
    "warning_soft": "#fff4e5",
    "danger": "#ff5a5f",
    "danger_soft": "#ffe9ea",
    "sky": "#56ccf2",
}

STATE_DIR = PROJECT_ROOT / "live" / "state"
M4_STATE_PATH = STATE_DIR / "m4_state.json"
M5_STATE_PATH = STATE_DIR / "m5_state.json"
M6_STATE_PATH = STATE_DIR / "m6_state.json"
M7_STATE_PATH = STATE_DIR / "m7_state.json"
M8_STATE_PATH = STATE_DIR / "m8_state.json"
ORTAK_STATE_PATH = STATE_DIR / "ortak_state.json"
LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc
DB = BotStateDB()


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;600;700;800&display=swap');

          html, body, [class*="css"] {{
            font-family: 'Manrope', sans-serif;
          }}

          .stApp {{
            color: {PALETTE["ink"]};
            background:
              radial-gradient(circle at 0% 0%, rgba(15,111,255,0.12), transparent 24%),
              radial-gradient(circle at 100% 0%, rgba(18,184,134,0.10), transparent 20%),
              linear-gradient(180deg, {PALETTE["bg"]} 0%, {PALETTE["bg_alt"]} 100%);
          }}

          .block-container {{
            max-width: 1480px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
          }}

          [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
          }}

          section[data-testid="stSidebar"] {{
            display: none;
          }}

          /* Streamlit 1.57 testid'leri = stMetric / stMetricValue / stMetricLabel
             (eski 'metric-container' artık YOK → eski CSS hiç uygulanmıyordu, değer kesiliyordu). */
          div[data-testid="stMetric"], div[data-testid="metric-container"] {{
            background: linear-gradient(180deg, {PALETTE["card"]} 0%, {PALETTE["card_alt"]} 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 12px 10px;
            box-shadow: 0 16px 40px rgba(16, 35, 63, 0.06);
          }}

          div[data-testid="stMetric"] label,
          div[data-testid="metric-container"] label,
          [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
            color: {PALETTE["muted"]} !important;
            font-size: 0.74rem !important;
            letter-spacing: 0.01em;
          }}

          /* 8 metrik tek satıra sığsın: küçük punto + tek satır + kesilme/üç nokta KAPALI. */
          [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
            color: {PALETTE["ink"]} !important;
            font-size: 1.0rem !important;
            font-weight: 800;
            letter-spacing: -0.03em;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
          }}

          .stTabs [data-baseweb="tab-list"] {{
            gap: 0.65rem;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid {PALETTE["border"]};
            border-radius: 999px;
            padding: 0.45rem;
            box-shadow: 0 10px 24px rgba(16, 35, 63, 0.05);
          }}

          .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 0.6rem 1rem;
            font-weight: 700;
            color: {PALETTE["muted"]};
          }}

          .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #edf4ff 0%, #ffffff 100%) !important;
            color: {PALETTE["primary"]} !important;
            border: 1px solid rgba(15, 111, 255, 0.15) !important;
          }}

          .stButton button {{
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, {PALETTE["primary"]} 0%, #3a8bff 100%);
            color: white;
            font-weight: 800;
            padding: 0.68rem 1.15rem;
            box-shadow: 0 12px 24px rgba(15, 111, 255, 0.22);
          }}

          .stButton button:hover {{
            filter: brightness(1.03);
          }}

          .stSelectbox label, .stToggle label, .stRadio label {{
            color: {PALETTE["muted"]} !important;
            font-weight: 700 !important;
          }}

          .stDataFrame {{
            border: 1px solid {PALETTE["border"]};
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 14px 32px rgba(16, 35, 63, 0.04);
          }}

          .stAlert {{
            border-radius: 18px;
          }}

          .mono {{
            font-family: 'IBM Plex Mono', monospace;
          }}

          .hero {{
            padding: 1.35rem 1.5rem;
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.72);
            background:
              linear-gradient(120deg, rgba(255,255,255,0.95) 0%, rgba(244,249,255,0.93) 55%, rgba(232,242,255,0.92) 100%);
            box-shadow: 0 22px 54px rgba(16, 35, 63, 0.10);
          }}

          .hero-title {{
            color: {PALETTE["ink"]};
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            margin-bottom: 0.2rem;
          }}

          .hero-subtitle {{
            color: {PALETTE["muted"]};
            font-size: 0.97rem;
            max-width: 860px;
          }}

          .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
          }}

          .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            border: 1px solid transparent;
          }}

          .badge-neutral {{
            color: {PALETTE["ink"]};
            background: #ffffff;
            border-color: {PALETTE["border"]};
          }}

          .badge-primary {{
            color: {PALETTE["primary"]};
            background: {PALETTE["primary_soft"]};
            border-color: rgba(15, 111, 255, 0.18);
          }}

          .badge-success {{
            color: {PALETTE["success"]};
            background: {PALETTE["success_soft"]};
            border-color: rgba(18, 184, 134, 0.18);
          }}

          .badge-warning {{
            color: {PALETTE["warning"]};
            background: {PALETTE["warning_soft"]};
            border-color: rgba(255, 159, 28, 0.20);
          }}

          .badge-danger {{
            color: {PALETTE["danger"]};
            background: {PALETTE["danger_soft"]};
            border-color: rgba(255, 90, 95, 0.18);
          }}

          .panel {{
            padding: 1.15rem 1.2rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid {PALETTE["border"]};
            box-shadow: 0 18px 40px rgba(16, 35, 63, 0.06);
          }}

          .panel-title {{
            font-size: 0.86rem;
            color: {PALETTE["muted"]};
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.45rem;
          }}

          .panel-big {{
            font-size: 1.55rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            line-height: 1.2;
          }}

          .panel-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.92rem;
            margin-top: 0.45rem;
          }}

          .coin-card {{
            background: linear-gradient(180deg, {PALETTE["card"]} 0%, {PALETTE["card_alt"]} 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 22px;
            padding: 0.95rem 1rem;
            box-shadow: 0 12px 28px rgba(16, 35, 63, 0.05);
            margin-bottom: 0.8rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
          }}

          .coin-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 30px rgba(16, 35, 63, 0.08);
          }}

          .coin-card.selected {{
            border-color: rgba(15, 111, 255, 0.40);
            box-shadow: 0 20px 36px rgba(15, 111, 255, 0.12);
            background: linear-gradient(180deg, #ffffff 0%, #edf4ff 100%);
          }}

          .coin-card-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
          }}

          .coin-symbol {{
            color: {PALETTE["ink"]};
            font-size: 1rem;
            font-weight: 800;
          }}

          .coin-price {{
            color: {PALETTE["ink"]};
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
          }}

          .coin-meta {{
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
          }}

          .section-head {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 0.8rem;
            margin: 0.25rem 0 0.85rem;
          }}

          .section-title {{
            color: {PALETTE["ink"]};
            font-size: 1.2rem;
            font-weight: 800;
          }}

          .section-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.9rem;
          }}

          .mini-list {{
            display: grid;
            gap: 0.6rem;
          }}

          .mini-row {{
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {PALETTE["border"]};
          }}

          .mini-row:last-child {{
            border-bottom: none;
            padding-bottom: 0;
          }}

          .mini-key {{
            color: {PALETTE["muted"]};
            font-weight: 700;
            font-size: 0.86rem;
          }}

          .mini-value {{
            color: {PALETTE["ink"]};
            font-weight: 800;
            font-size: 0.88rem;
            text-align: right;
          }}

          .tv-shell {{
            background: linear-gradient(180deg, #0c1425 0%, #0f1b32 100%);
            border: 1px solid #1e314f;
            border-radius: 28px;
            padding: 1rem 1rem 0.7rem;
            box-shadow: 0 24px 50px rgba(9, 16, 30, 0.28);
          }}

          .tv-topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.15rem 0.2rem 0.9rem;
            color: #dbe7fb;
          }}

          .tv-symbol {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #f7fbff;
          }}

          .tv-sub {{
            font-size: 0.82rem;
            color: #8fa8c9;
            margin-top: 0.15rem;
          }}

          .tv-price {{
            font-size: 1.45rem;
            font-weight: 800;
            color: #f7fbff;
            text-align: right;
          }}

          .tv-stats {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.55rem;
            margin-bottom: 0.9rem;
          }}

          .tv-stat {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(143,168,201,0.18);
            border-radius: 16px;
            padding: 0.7rem 0.8rem;
          }}

          .tv-stat-label {{
            font-size: 0.68rem;
            color: #88a0bf;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
            font-weight: 700;
          }}

          .tv-stat-value {{
            font-size: 0.94rem;
            color: #f4f8ff;
            font-weight: 800;
          }}

          .tv-help {{
            color: #8fa8c9;
            font-size: 0.82rem;
            margin: 0.75rem 0 0.25rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_money(value: float, digits: int = 2) -> str:
    return f"${safe_float(value):,.{digits}f}"


def format_pct(value: float, digits: int = 2, signed: bool = True) -> str:
    fmt = f"{{:{'+' if signed else ''},.{digits}f}}%"
    return fmt.format(safe_float(value))


def compact_money(value: float) -> str:
    value = safe_float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return format_money(value)


def symbol_label(symbol: str) -> str:
    return str(symbol or "").replace("/USDT", "")


def parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if len(text) == 10 and text.count("-") == 2:
            dt = datetime.fromisoformat(text)
            return dt.replace(tzinfo=LOCAL_TZ)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ)
    except ValueError:
        return None


def relative_time_label(dt: datetime | None) -> str:
    if dt is None:
        return "bilinmiyor"
    delta = datetime.now(LOCAL_TZ) - dt
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "az once"
    if seconds < 3600:
        return f"{seconds // 60} dk once"
    if seconds < 86400:
        return f"{seconds // 3600} sa once"
    return f"{seconds // 86400} gun once"


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def signal_badge(side: str) -> str:
    mapping = {
        "BUY": ("BUY", "badge-success"),
        "SELL": ("SELL", "badge-danger"),
        "HOLD": ("HOLD", "badge-warning"),
    }
    label, cls = mapping.get(str(side or "").upper(), ("NO SIGNAL", "badge-neutral"))
    return f'<span class="badge {cls}">{label}</span>'


def style_delta_class(value: float) -> str:
    if value > 0:
        return "badge-success"
    if value < 0:
        return "badge-danger"
    return "badge-neutral"


def style_metric_delta(value: float) -> str:
    if value > 0:
        return "normal"
    if value < 0:
        return "inverse"
    return "off"


def plot_layout(fig: go.Figure, *, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color=PALETTE["ink"], family="IBM Plex Mono"),
        margin=dict(l=12, r=12, t=28, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["muted"], size=11),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["card"],
            bordercolor=PALETTE["border"],
            font=dict(color=PALETTE["ink"]),
        ),
        height=height,
        showlegend=legend,
    )
    fig.update_xaxes(
        gridcolor="rgba(102,120,146,0.14)",
        zeroline=False,
        showline=False,
        tickfont=dict(color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(102,120,146,0.14)",
        zeroline=False,
        showline=False,
        tickfont=dict(color=PALETTE["muted"]),
    )
    return fig


def dark_plot_layout(fig: go.Figure, *, height: int = 560, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0f1b32",
        plot_bgcolor="#0f1b32",
        font=dict(color="#dbe7fb", family="IBM Plex Mono"),
        margin=dict(l=12, r=12, t=26, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8fa8c9", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0c1425",
            bordercolor="#29456f",
            font=dict(color="#f4f8ff"),
        ),
        height=height,
        showlegend=legend,
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="rgba(143,168,201,0.12)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#8fa8c9"),
    )
    fig.update_yaxes(
        gridcolor="rgba(143,168,201,0.12)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#8fa8c9"),
    )
    return fig


@st.cache_data(ttl=120, show_spinner=False)
def load_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_data(ttl=890, show_spinner=False)
def load_model_state(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=890, show_spinner=False)
def load_bot_snapshot() -> dict[str, Any]:
    return {
        "status": DB.get_bot_status(),
        "stats": DB.get_trade_stats(),
        "equity": DB.get_equity_curve(limit=600),
        "open_positions": DB.get_open_positions(),
        "closed": DB.get_closed_trades(limit=250),
        "signals": DB.get_signal_log(limit=160),
        "adapt": DB.get_adaptive_log(limit=40),
        "scalp_learn": DB.get_scalp_learn(),
    }


def ensure_market_selection(watchlist: list[str]) -> str:
    if not watchlist:
        return ""
    current = st.session_state.get("market_detail_symbol")
    if current not in watchlist:
        st.session_state["market_detail_symbol"] = watchlist[0]
    return st.session_state["market_detail_symbol"]


@st.cache_resource(show_spinner=False)
def get_public_exchange(exchange_name: str) -> ccxt.Exchange:
    exchange_cls = getattr(ccxt, exchange_name, ccxt.binance)
    return exchange_cls(
        {
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": "spot"},
        }
    )


@st.cache_data(ttl=25, show_spinner=False)
def fetch_market_snapshot(symbols: tuple[str, ...], exchange_name: str) -> list[dict[str, Any]]:
    exchange = get_public_exchange(exchange_name)
    rows: list[dict[str, Any]] = []
    raw_tickers: dict[str, Any] = {}

    try:
        if exchange.has.get("fetchTickers"):
            raw_tickers = exchange.fetch_tickers(list(symbols))
    except Exception:
        raw_tickers = {}

    for symbol in symbols:
        ticker = raw_tickers.get(symbol)
        if ticker is None:
            try:
                ticker = exchange.fetch_ticker(symbol)
            except Exception as exc:
                ticker = {"error": str(exc)}

        last = safe_float(ticker.get("last"))
        low = safe_float(ticker.get("low"))
        high = safe_float(ticker.get("high"))
        change_pct = safe_float(ticker.get("percentage"))
        quote_volume = safe_float(ticker.get("quoteVolume"))
        open_price = safe_float(ticker.get("open"))
        range_pos = 0.0
        if high > low and last > 0:
            range_pos = (last - low) / (high - low) * 100

        rows.append(
            {
                "symbol": symbol,
                "label": symbol_label(symbol),
                "last": last,
                "change_pct": change_pct,
                "open": open_price,
                "high": high,
                "low": low,
                "quote_volume": quote_volume,
                "range_pos": range_pos,
                "error": ticker.get("error"),
            }
        )

    return rows


def timeframe_to_minutes(timeframe: str) -> int:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
    }
    return mapping.get(timeframe, 60)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, exchange_name: str, days: int, limit: int | None = None) -> pd.DataFrame:
    exchange = get_public_exchange(exchange_name)
    candles_per_day = max(1, int((24 * 60) / timeframe_to_minutes(timeframe)))
    effective_limit = limit or max(120, min(4000, days * candles_per_day + 220))
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=effective_limit)
    except Exception:
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_volume = df["volume"].cumsum().replace(0, pd.NA)
    df["vwap"] = (typical_price * df["volume"]).cumsum() / cum_volume

    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def prepare_db_trades(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()

    rows = []
    for trade in trades:
        rows.append(
            {
                "symbol": trade.get("symbol", ""),
                "label": symbol_label(trade.get("symbol", "")),
                "pnl": safe_float(trade.get("realized_pnl")),
                "entry_price": safe_float(trade.get("entry_price")),
                "exit_price": safe_float(trade.get("exit_price")),
                "position_size": safe_float(trade.get("position_size")),
                "reason": trade.get("close_reason", ""),
                "closed_at": trade.get("closed_at", ""),
                "closed_dt": parse_timestamp(trade.get("closed_at")),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("closed_dt", ascending=False, na_position="last").reset_index(drop=True)
    return df


def prepare_signal_df(signals: list[dict[str, Any]]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()

    rows = []
    for signal in signals:
        rows.append(
            {
                "symbol": signal.get("symbol", ""),
                "label": symbol_label(signal.get("symbol", "")),
                "side": signal.get("side", ""),
                "price": safe_float(signal.get("price")),
                "confidence": safe_float(signal.get("confidence")),
                "rsi": safe_float(signal.get("rsi"), math.nan),
                "adx": safe_float(signal.get("adx"), math.nan),
                "reason": signal.get("reason", ""),
                "timestamp": signal.get("timestamp", ""),
                "ts": parse_timestamp(signal.get("timestamp")),
            }
        )

    return pd.DataFrame(rows)


def build_signal_map(signal_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if signal_df.empty:
        return {}
    latest_rows = signal_df.sort_values("ts", ascending=False, na_position="last")
    latest_rows = latest_rows.drop_duplicates(subset=["symbol"], keep="first")
    return {row["symbol"]: row.to_dict() for _, row in latest_rows.iterrows()}


def prepare_state_trades(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()

    rows = []
    for trade in trades:
        rows.append(
            {
                "symbol": trade.get("symbol", ""),
                "label": symbol_label(trade.get("symbol", "")),
                # is_short → SHORT/LONG etiketi
                "side": ("SHORT" if trade.get("is_short") else "LONG") if "is_short" in trade else trade.get("side", ""),
                "entry_date": trade.get("entry_date", ""),
                "exit_date": trade.get("exit_date", ""),
                "entry_price": safe_float(trade.get("entry_price")),
                "exit_price": safe_float(trade.get("exit_price")),
                # v14: kullanıcı isteği — adet (size) ve maliyet (cost) tabloya geldi
                "size": safe_float(trade.get("size")),
                # cost yoksa entry_price × size'dan hesapla (eski kayıtlar için fallback)
                "cost": safe_float(trade.get("cost")) or (
                    safe_float(trade.get("entry_price")) * safe_float(trade.get("size"))
                ),
                "pnl": safe_float(trade.get("pnl")),
                # pnl_pct: state'de yoksa initial_capital'dan hesapla
                "pnl_pct": safe_float(trade.get("pnl_pct")),
                "reason": trade.get("exit_reason", ""),
                "bars_held": safe_int(trade.get("bars_held")),
                "exit_dt": parse_timestamp(trade.get("exit_date")),
            }
        )

    df = pd.DataFrame(rows)
    sort_key = "exit_dt" if "exit_dt" in df.columns else None
    if sort_key:
        df = df.sort_values(sort_key, ascending=True, na_position="last").reset_index(drop=True)
    df["cum_pnl"] = df["pnl"].cumsum()
    # pnl_pct yoksa entry_price × size'dan hesapla (SHORT için negatif olabilir)
    if "pnl_pct" in df.columns:
        mask = df["pnl_pct"] == 0.0
        cost_basis = (df["entry_price"] * df["size"]).replace(0, float("nan"))
        df.loc[mask, "pnl_pct"] = (df.loc[mask, "pnl"] / cost_basis.loc[mask] * 100)
    return df


def compute_open_position_value(state: dict[str, Any] | None) -> float:
    """Açık pozisyonların toplam güncel değeri ($).

    LONG  → coine giren para hâlâ "sermaye": maliyet + anlık kâr/zarar (≈ güncel piyasa değeri).
    SHORT → bakiyeden kilitlenen marjin + anlık kâr/zarar.
    Bu değer serbest nakde eklenince GERÇEK toplam sermaye (equity) elde edilir; böylece
    pozisyondayken (coinde para varken) toplam sermaye düşük/eksi görünmez.
    """
    if not state:
        return 0.0
    total = 0.0
    for p in state.get("open_positions", []) or []:
        upnl = safe_float(p.get("unrealized_pnl"), 0.0)
        size = safe_float(p.get("size"))
        ep   = safe_float(p.get("entry_price"))
        if p.get("is_short", False):
            locked = safe_float(p.get("margin_locked"), 0.0)
            if locked < 0.01:
                locked = safe_float(p.get("cost"), 0.0) or ep * size
            total += locked + upnl
        else:
            cost = safe_float(p.get("cost"), 0.0)
            if cost < 1.0:
                cost = ep * size
            total += cost + upnl
    return total


def build_model_summary(name: str, state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {
            "name": name,
            "available": False,
            "state": None,
            "trade_df": pd.DataFrame(),
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "expectancy": 0.0,
            "risk_efficiency": 0.0,
            "freshness": "veri yok",
            "fresh_dt": None,
        }

    trade_df = prepare_state_trades(state.get("closed_trades", []))
    initial_capital = safe_float(state.get("initial_capital"), 10_000.0)
    total_pnl = safe_float(state.get("total_pnl"))
    return_pct = safe_float(state.get("total_pnl_pct"))
    drawdown_pct = safe_float(state.get("max_drawdown_pct"))
    win_rate = safe_float(state.get("win_rate"))
    trade_count = safe_int(state.get("total_trades")) or len(state.get("closed_trades", []))
    open_positions = state.get("open_positions", []) or []
    final_balance = safe_float(state.get("final_balance"), initial_capital + total_pnl)
    # GERÇEK toplam sermaye (equity) = serbest nakit + açık pozisyonların değeri.
    # final_balance/balance sadece SERBEST NAKİT'tir (LONG'da coine giren para düşülür);
    # pozisyondaki para da sermaye olduğu için onu geri ekliyoruz → "eksi/düşük sermaye" görünümü çözülür.
    free_cash  = safe_float(state.get("balance"), final_balance)
    open_value = compute_open_position_value(state)
    equity     = free_cash + open_value
    # Equity getirisi = (toplam sermaye / başlangıç − 1). total_pnl_pct SADECE realize kârdır;
    # equity_return realize + açık (unrealized) kârı birlikte içerir → gösterilen Bakiye ile tutarlı.
    equity_return = ((equity / initial_capital - 1) * 100) if initial_capital else 0.0
    run_dt = parse_timestamp(state.get("run_time"))

    if not trade_df.empty:
        wins = trade_df.loc[trade_df["pnl"] > 0, "pnl"]
        losses = trade_df.loc[trade_df["pnl"] < 0, "pnl"]
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        avg_trade = trade_df["pnl"].mean()
        expectancy = trade_df["pnl_pct"].mean()
        trade_df["equity"] = initial_capital + trade_df["cum_pnl"]
    else:
        profit_factor = 0.0
        avg_trade = 0.0
        expectancy = 0.0

    risk_efficiency = return_pct / max(drawdown_pct, 1.0) if drawdown_pct else return_pct

    # ── Sharpe / Omega / reel getiri (günlük equity geçmişinden) ─────────────
    # equity_history: [["YYYY-MM-DD", equity], ...] — live_runner her tick'te yazar.
    # Reel getiri: dolar enflasyonu varsayımı %3/yıl (sabit, gösterge amaçlı).
    INFLATION_ANNUAL = 0.03
    sharpe = None
    omega = None
    real_return_pct = None
    elapsed_days = None
    eq_hist = state.get("equity_history") or []
    try:
        if len(eq_hist) >= 5:
            _s = pd.Series({pd.Timestamp(d): float(v) for d, v in eq_hist}).sort_index()
            _rets = _s.pct_change().dropna()
            if len(_rets) >= 4 and float(_rets.std()) > 0:
                sharpe = float(_rets.mean() / _rets.std()) * math.sqrt(365)
                _gains = float(_rets[_rets > 0].sum())
                _losses = abs(float(_rets[_rets < 0].sum()))
                omega = (_gains / _losses) if _losses > 0 else float("inf")
        _created = parse_timestamp(state.get("created_at"))
        if _created is not None:
            elapsed_days = max((datetime.now(timezone.utc) - _created).days, 1)
            _infl = (1 + INFLATION_ANNUAL) ** (elapsed_days / 365) - 1
            real_return_pct = ((1 + equity_return / 100) / (1 + _infl) - 1) * 100
    except Exception:
        pass

    return {
        "sharpe": sharpe,
        "omega": omega,
        "real_return_pct": real_return_pct,
        "elapsed_days": elapsed_days,
        "name": name,
        "available": True,
        "state": state,
        "trade_df": trade_df,
        "initial_capital": initial_capital,
        "final_balance": final_balance,
        "free_cash": free_cash,
        "open_value": open_value,
        "equity": equity,
        "equity_return": equity_return,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "drawdown_pct": drawdown_pct,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "open_positions": open_positions,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
        "risk_efficiency": risk_efficiency,
        "run_dt": run_dt,
        "freshness": relative_time_label(run_dt),
        "start_date": state.get("start_date", ""),
        "end_date": state.get("end_date", ""),
    }


def build_watchlist(config: dict[str, Any], models: list[dict[str, Any]]) -> list[str]:
    """
    Tüm UNIVERSE coinleri + açık pozisyon coinleri döndürür.
    Market ekranı ve price fetch için kullanılır.
    BTC her zaman başa eklenir (rejim izleme + benchmark).
    """
    # Açık pozisyon/trade'lerden aktif coinler (öncelikli sıralamada)
    active: list[str] = []
    for model in models:
        state = model.get("state") or {}
        active.extend(pos.get("symbol", "") for pos in state.get("open_positions", []))
        active.extend(tr.get("symbol", "") for tr in state.get("closed_trades", [])[:10])

    # BTC + UNIVERSE + aktif coinler (tekrarsız, sıralı)
    full = dedupe(["BTC/USDT"] + _CRYPTO_UNIVERSE + active)
    return full


def market_breadth(market_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in market_rows if not row.get("error") and row.get("last")]
    if not valid:
        return {"gainers": 0, "losers": 0, "flat": 0, "avg_change": 0.0}

    gainers = sum(1 for row in valid if row["change_pct"] > 0)
    losers = sum(1 for row in valid if row["change_pct"] < 0)
    flat = len(valid) - gainers - losers
    avg_change = sum(row["change_pct"] for row in valid) / len(valid)
    return {
        "gainers": gainers,
        "losers": losers,
        "flat": flat,
        "avg_change": avg_change,
    }


def leader_summary(m4: dict[str, Any], m5: dict[str, Any]) -> tuple[str, str]:
    if not m4.get("available") and not m5.get("available"):
        return ("M4 / M5 verisi bekleniyor", "Live runner bir kez calistiktan sonra karsilastirma buraya dusecek.")
    if not m4.get("available"):
        return ("M5 ekranda, M4 bekleniyor", "M5 durum dosyasi var; M4 state olusunca karsilastirma tamamlanacak.")
    if not m5.get("available"):
        return ("M4 ekranda, M5 bekleniyor", "M4 durum dosyasi var; M5 state olusunca karsilastirma tamamlanacak.")

    pnl_gap = safe_float(m5["total_pnl"]) - safe_float(m4["total_pnl"])
    dd_gap = safe_float(m5["drawdown_pct"]) - safe_float(m4["drawdown_pct"])
    wr_gap = safe_float(m5["win_rate"]) - safe_float(m4["win_rate"])

    if abs(pnl_gap) < 10:
        title = "Iki model su an neredeyse basa bas"
    elif pnl_gap > 0:
        title = f"M5 {format_money(abs(pnl_gap))} onde"
    else:
        title = f"M4 {format_money(abs(pnl_gap))} onde"

    detail_parts = [
        f"Win rate farki {wr_gap:+.1f} puan",
        f"drawdown farki {dd_gap:+.1f} puan",
        f"risk/verim skoru M4 {m4['risk_efficiency']:.2f} - M5 {m5['risk_efficiency']:.2f}",
    ]
    return title, " | ".join(detail_parts)


def build_market_board(market_rows: list[dict[str, Any]], signal_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in market_rows:
        signal = signal_map.get(row["symbol"], {})
        signal_side = signal.get("side", "—")
        confidence = safe_float(signal.get("confidence"), math.nan)
        signal_age = relative_time_label(signal.get("ts"))
        board_row = {
            "Coin": row["label"],
            "Fiyat": format_money(row["last"], 4 if row["last"] < 1 else 2),
            "24s %": format_pct(row["change_pct"]),
            "Gunluk Aralik": f"{row['range_pos']:.0f}%",
            "Hacim": compact_money(row["quote_volume"]),
            "Son Sinyal": signal_side if signal_side else "—",
            "Skor": "—" if math.isnan(confidence) else f"{confidence:.2f}",
            "RSI": "—" if math.isnan(safe_float(signal.get("rsi"), math.nan)) else f"{safe_float(signal.get('rsi')):.0f}",
            "ADX": "—" if math.isnan(safe_float(signal.get("adx"), math.nan)) else f"{safe_float(signal.get('adx')):.0f}",
            "Sinyal Yasi": signal_age,
        }
        rows.append(board_row)
    return pd.DataFrame(rows)


def render_hero(
    *,
    exchange_name: str,
    breadth: dict[str, Any],
    m4: dict[str, Any],
    m5: dict[str, Any],
    bot_status: dict[str, Any] | None,
) -> None:
    title, detail = leader_summary(m4, m5)
    mode = (bot_status or {}).get("mode") or "Mission Control"
    last_update = relative_time_label(parse_timestamp((bot_status or {}).get("last_update")))
    hero_html = f"""
        <div class="hero">
          <div class="hero-title">Kripto Bot Control Room</div>
          <div class="hero-subtitle">
            M4 ve M5 modellerini ayni ekranda izle, canli coin akisini takip et,
            risk ve sinyal tarafini tek bakista karar verilebilir hale getir.
          </div>
          <div class="badge-row">
            <span class="badge badge-primary">Exchange {exchange_name.upper()}</span>
            <span class="badge badge-neutral">Aktif mod {mode}</span>
            <span class="badge badge-neutral">DB guncelleme {last_update}</span>
            <span class="badge {style_delta_class(breadth['avg_change'])}">
              Piyasa nefesi {breadth['avg_change']:+.2f}%
            </span>
            <span class="badge badge-success">{breadth['gainers']} yukselen</span>
            <span class="badge badge-danger">{breadth['losers']} dusen</span>
          </div>
        </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    insight_col, risk_col = st.columns(2)
    with insight_col:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Model Yorumu</div>
              <div class="panel-big">{title}</div>
              <div class="panel-copy">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with risk_col:
        dd_ref = max(safe_float(m4.get("drawdown_pct")), safe_float(m5.get("drawdown_pct")))
        risk_title = "Risk sicakligi dusuk"
        risk_copy = "Drawdown tarafinda kontrol korunuyor."
        if dd_ref >= 25:
            risk_title = "Risk sicakligi yuksek"
            risk_copy = "Drawdown 25% ustune cikmis. Pozisyon yogunlugu ve giriş kalitesi dikkat istiyor."
        elif dd_ref >= 15:
            risk_title = "Risk sicakligi orta"
            risk_copy = "Model performansi izlenebilir ama koruma tarafinda dikkat gerekiyor."

        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Risk Durumu</div>
              <div class="panel-big">{risk_title}</div>
              <div class="panel-copy">{risk_copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _model_equity(model: dict[str, Any]) -> float:
    """Model için GERÇEK toplam sermaye (equity = serbest nakit + açık pozisyon değeri).
    equity yoksa final_balance, o da yoksa initial_capital'a düşer (geriye dönük uyumlu)."""
    return safe_float(
        model.get("equity"),
        safe_float(model.get("final_balance"), safe_float(model.get("initial_capital"), 1000.0)),
    )


def render_top_metrics(
    *,
    snapshot: dict[str, Any],
    breadth: dict[str, Any],
    m4: dict[str, Any],
    m5: dict[str, Any],
    m6: dict[str, Any],
    m7: dict[str, Any],
    m8: dict[str, Any],
) -> None:
    open_positions = snapshot.get("open_positions") or []
    closed = snapshot.get("closed") or []
    stats = snapshot.get("stats") or {}
    trade_count = safe_int(stats.get("total"), len(closed))

    # Her bot ayrı $1000 ile başlar. GÖSTERİLEN BAKİYE = equity (serbest nakit + açık pozisyon
    # değeri) → LONG'dayken coine giren para da sayılır; "eksi/düşük sermaye" görünümü çözülür.
    m4_balance = _model_equity(m4)
    m5_balance = _model_equity(m5)
    m6_balance = _model_equity(m6)
    m7_balance = _model_equity(m7)
    m8_balance = _model_equity(m8)
    # Yüzde = equity getirisi (realize + açık K/Z) → gösterilen Bakiye($) ile tutarlı.
    m4_return = safe_float(m4.get("equity_return"), safe_float(m4.get("return_pct")))
    m5_return = safe_float(m5.get("equity_return"), safe_float(m5.get("return_pct")))
    m6_return = safe_float(m6.get("equity_return"), safe_float(m6.get("return_pct")))
    m7_return = safe_float(m7.get("equity_return"), safe_float(m7.get("return_pct")))
    m8_return = safe_float(m8.get("equity_return"), safe_float(m8.get("return_pct")))
    total_balance = m4_balance + m5_balance + m6_balance + m7_balance + m8_balance
    total_initial = (
        safe_float(m4.get("initial_capital"), 1000.0)
        + safe_float(m5.get("initial_capital"), 1000.0)
        + safe_float(m6.get("initial_capital"), 1000.0)
        + safe_float(m7.get("initial_capital"), 1000.0)
        + safe_float(m8.get("initial_capital"), 1000.0)
    )
    total_pnl = total_balance - total_initial

    # 5 model arasından en yüksek equity (toplam sermaye) olanı seç
    _candidates = [("M4", m4), ("M5", m5), ("M6", m6), ("M7", m7), ("M8", m8)]
    best_name, best_dict = max(_candidates, key=lambda x: _model_equity(x[1]))
    best_model = best_name
    best_model_return = safe_float(best_dict.get("equity_return"), safe_float(best_dict.get("return_pct")))

    cols = st.columns(8)
    # Üst satırda 8 metrik → bakiyeler TAM DOLAR (kuruşsuz) gösterilir ki dar sütuna sığsın.
    cols[0].metric("M4 Bakiye", format_money(m4_balance, 0), format_pct(m4_return), delta_color=style_metric_delta(m4_return))
    cols[1].metric("M5 Bakiye", format_money(m5_balance, 0), format_pct(m5_return), delta_color=style_metric_delta(m5_return))
    cols[2].metric("M6 Bakiye", format_money(m6_balance, 0), format_pct(m6_return), delta_color=style_metric_delta(m6_return))
    cols[3].metric("M7 Bakiye", format_money(m7_balance, 0), format_pct(m7_return), delta_color=style_metric_delta(m7_return))
    cols[4].metric("M8 Bakiye", format_money(m8_balance, 0), format_pct(m8_return), delta_color=style_metric_delta(m8_return))
    cols[5].metric("Acik Pozisyon", str(len(open_positions)), f"{trade_count} kapali islem")
    # Piyasa Nefesi: yükselen ▲ yeşil, düşen ▼ kırmızı ayrı renk — st.metric tek renk
    # desteklediğinden custom HTML kart kullanılır (metric kutusu stiliyle uyumlu).
    _avg = breadth["avg_change"]
    _avg_color = "#12b886" if _avg >= 0 else "#ff5a5f"
    cols[6].markdown(
        f"""
        <div data-testid="stMetric" style="
            background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
            border:1px solid #d7e2ef; border-radius:18px;
            padding:12px 10px; box-shadow:0 16px 40px rgba(16,35,63,0.06);
        ">
          <p style="color:#667892;font-size:0.74rem;margin:0 0 4px">Piyasa Nefesi</p>
          <p style="color:#10233f;font-size:1.0rem;font-weight:800;
                    letter-spacing:-0.03em;margin:0 0 6px">{format_pct(_avg)}</p>
          <p style="margin:0;font-size:0.80rem;font-weight:700;letter-spacing:-0.01em">
            <span style="color:#12b886">▲ {breadth['gainers']}</span>
            &nbsp;&nbsp;
            <span style="color:#ff5a5f">▼ {breadth['losers']}</span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[7].metric("Toplam Sermaye", format_money(total_balance, 0), format_money(total_pnl), delta_color=style_metric_delta(total_pnl))


def render_section_header(title: str, copy: str, right: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-head">
          <div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
          </div>
          <div class="section-copy">{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_cards(
    market_rows: list[dict[str, Any]],
    signal_map: dict[str, dict[str, Any]],
    *,
    interactive: bool = False,
    key_prefix: str = "market_card",
) -> None:
    if not market_rows:
        st.info("Canli market verisi alinmadi. Internet ya da exchange erisimi kontrol edilmeli.")
        return

    selected_symbol = st.session_state.get("market_detail_symbol")
    for offset in range(0, len(market_rows), 4):
        cols = st.columns(4)
        for idx, row in enumerate(market_rows[offset : offset + 4]):
            signal = signal_map.get(row["symbol"], {})
            side = signal.get("side", "NO SIGNAL")
            confidence = safe_float(signal.get("confidence"))
            meta_signal = side if side != "NO SIGNAL" else "sinyal yok"
            if side != "NO SIGNAL":
                meta_signal = f"{side} {confidence:.2f}"
            badge_class = style_delta_class(row["change_pct"])
            selected_cls = " selected" if selected_symbol == row["symbol"] else ""
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="coin-card{selected_cls}">
                      <div class="coin-card-top">
                        <div class="coin-symbol">{row["label"]}</div>
                        <span class="badge {badge_class}">{row["change_pct"]:+.2f}%</span>
                      </div>
                      <div class="coin-price mono">{format_money(row["last"], 4 if row["last"] < 1 else 2)}</div>
                      <div class="coin-meta">
                        <span>Hacim {compact_money(row["quote_volume"])}</span>
                        <span>{meta_signal}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if interactive:
                    if st.button(
                        f"{row['label']} grafiğini aç",
                        key=f"{key_prefix}_{row['symbol']}",
                        use_container_width=True,
                    ):
                        st.session_state["market_detail_symbol"] = row["symbol"]


def render_model_comparison_curve(m4: dict[str, Any], m5: dict[str, Any]) -> None:
    fig = go.Figure()
    rendered = False

    for model, color in ((m4, PALETTE["primary"]), (m5, PALETTE["success"])):
        df = model.get("trade_df")
        if df is None or df.empty:
            continue
        x_values = list(range(1, len(df) + 1))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["equity"],
                mode="lines",
                name=model["name"],
                line=dict(color=color, width=3),
                hovertemplate=f"{model['name']}<br>Islem #%{{x}}<br>Equity: $%{{y:,.2f}}<extra></extra>",
            )
        )
        rendered = True

    if not rendered:
        st.info("Karsilastirma egrisi icin kapali islem verisi yok.")
        return

    fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(title="Kapanan islem sirasi")
    plot_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_equity_timeline_with_trades(m4: dict[str, Any], m5: dict[str, Any]) -> None:
    """
    Equity eğrisi — x ekseni ZAMAN (saat bazlı), işlem noktaları üzerinde.
    Yeşil nokta = kârlı işlem, kırmızı nokta = zararlı işlem.
    Hover: coin adı, alış fiyatı, satış fiyatı, PnL.
    """
    fig = go.Figure()
    rendered = False

    for model, line_color in [(m4, PALETTE["primary"]), (m5, PALETTE["success"])]:
        df = model.get("trade_df")
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["exit_dt"]).copy()
        if df.empty:
            continue

        init_cap = model.get("initial_capital", 1000.0)

        # Başlangıç noktası: başlangıç zamanı + init_cap
        first_ts = df["exit_dt"].min()
        x_vals = [first_ts - pd.Timedelta(minutes=1)] + list(df["exit_dt"])
        y_vals = [init_cap] + list(df["equity"])

        # Equity çizgisi
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=model["name"],
            line=dict(color=line_color, width=2.5),
        ))

        # İşlem noktaları
        for is_win in [True, False]:
            subset = df[df["pnl"] > 0] if is_win else df[df["pnl"] <= 0]
            if subset.empty:
                continue
            marker_color  = PALETTE["success"] if is_win else PALETTE["danger"]
            marker_symbol = "circle" if is_win else "circle"
            marker_size   = 10 if is_win else 9

            # Hover için customdata: [coin, giriş$, çıkış$, pnl$, alış_tarihi]
            custom = list(zip(
                subset["label"].tolist(),
                subset["entry_price"].tolist(),
                subset["exit_price"].tolist(),
                subset["pnl"].tolist(),
                subset["entry_date"].tolist(),
            ))

            fig.add_trace(go.Scatter(
                x=subset["exit_dt"],
                y=subset["equity"],
                mode="markers",
                name=f"{'Kâr' if is_win else 'Zarar'} ({model['name']})",
                marker=dict(
                    symbol=marker_symbol,
                    size=marker_size,
                    color=marker_color,
                    line=dict(color="white", width=1.5),
                    opacity=0.90,
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Alış: $%{customdata[1]:.4f}<br>"
                    "Satış: $%{customdata[2]:.4f}<br>"
                    "PnL: <b>$%{customdata[3]:+.2f}</b><br>"
                    "Alış tarihi: %{customdata[4]}<br>"
                    "Satış tarihi: %{x}<extra></extra>"
                ),
                showlegend=False,
            ))
        rendered = True

    if not rendered:
        st.info("Equity grafiği için kapanmış işlem verisi bekleniyor — bot bir işlemi kapattığında burada görünecek.")
        return

    fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(title="Zaman")
    plot_layout(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_open_positions_allocation(
    m4: dict[str, Any],
    m5: dict[str, Any],
    m6: dict[str, Any],
    m7: dict[str, Any],
    m8: dict[str, Any],
) -> None:
    """
    Açık pozisyonlar: hangi coin, kaç dolar yatırıldı.
    Sade kart formatı — model adı, coin adı, $ tutarı.
    """
    all_positions: list[dict] = []
    for model in [m4, m5, m6, m7, m8]:
        if not model.get("available"):
            continue
        for pos in model.get("open_positions", []):
            ep       = safe_float(pos.get("entry_price", 0))
            size     = safe_float(pos.get("size", 0))
            cost     = safe_float(pos.get("cost", 0)) or ep * size
            upnl     = safe_float(pos.get("unrealized_pnl", 0))
            is_short = pos.get("is_short", False)
            all_positions.append({
                "model":    model["name"],
                "sym":      pos.get("symbol", ""),
                "label":    symbol_label(pos.get("symbol", "")),
                "cost":     cost,
                "entry":    ep,
                "upnl":     upnl,
                "is_short": is_short,
            })

    if not all_positions:
        st.markdown(
            '<div class="panel-copy" style="padding:0.5rem 0">Şu an açık pozisyon yok.</div>',
            unsafe_allow_html=True,
        )
        return

    # HTML badge satırı
    badges = []
    for p in all_positions:
        upnl_color = PALETTE["success"] if p["upnl"] >= 0 else PALETTE["danger"]
        upnl_sign  = "+" if p["upnl"] >= 0 else ""
        side_tag   = (
            f'<span style="color:#ef4444;font-size:0.72rem;font-weight:700">▼SHORT</span>&nbsp;'
            if p["is_short"] else
            f'<span style="color:#10b981;font-size:0.72rem;font-weight:700">▲LONG</span>&nbsp;'
        )
        badges.append(
            f'<span class="badge badge-neutral" style="font-size:0.82rem">'
            f'{side_tag}'
            f'<b>{p["label"]}</b>&nbsp;'
            f'<span style="color:{PALETTE["muted"]}">[{p["model"]}]</span>&nbsp;'
            f'${p["cost"]:,.0f}'
            f'&nbsp;<span style="color:{upnl_color}">{upnl_sign}${p["upnl"]:,.2f}</span>'
            f'</span>'
        )

    st.markdown(
        f'<div class="badge-row" style="margin:0.4rem 0 0.8rem">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


def render_market_breadth_chart(market_rows: list[dict[str, Any]]) -> None:
    if not market_rows:
        st.info("Canli market grafigi icin veri yok.")
        return

    df = pd.DataFrame(market_rows).sort_values("change_pct")
    colors = [PALETTE["success"] if value >= 0 else PALETTE["danger"] for value in df["change_pct"]]
    fig = go.Figure(
        go.Bar(
            x=df["change_pct"],
            y=df["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.2f}%" for value in df["change_pct"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="rgba(16,35,63,0.25)")
    fig.update_xaxes(title="24 saatlik degisim")
    plot_layout(fig, height=360, legend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_tradingview_header(
    *,
    symbol: str,
    selected_row: dict[str, Any] | None,
    timeframe: str,
    lower_panel: str,
    latest_signal: dict[str, Any] | None,
    df: pd.DataFrame,
) -> None:
    last_close = safe_float(df["close"].iloc[-1]) if not df.empty else safe_float((selected_row or {}).get("last"))
    prev_close = safe_float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    open_price = safe_float(df["open"].iloc[-1]) if not df.empty else 0.0
    high_price = safe_float(df["high"].iloc[-1]) if not df.empty else 0.0
    low_price = safe_float(df["low"].iloc[-1]) if not df.empty else 0.0
    latest_volume = safe_float(df["volume"].iloc[-1]) if not df.empty else 0.0
    intrabar_change = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0
    day_change = safe_float((selected_row or {}).get("change_pct"))
    signal_text = "Sinyal yok"
    rsi_last = safe_float(df["rsi14"].iloc[-1], math.nan) if not df.empty else math.nan
    vwap_last = safe_float(df["vwap"].iloc[-1], math.nan) if not df.empty else math.nan
    rsi_text = "—" if math.isnan(rsi_last) else f"{rsi_last:.1f}"
    vwap_text = "—" if math.isnan(vwap_last) else format_money(vwap_last, 4 if vwap_last < 1 else 2)
    if latest_signal:
        signal_text = f"{latest_signal.get('side', '—')} · skor {safe_float(latest_signal.get('confidence')):.2f}"

    st.markdown(
        f"""
        <div class="tv-shell">
          <div class="tv-topbar">
            <div>
              <div class="tv-symbol">{symbol_label(symbol)} / USDT</div>
              <div class="tv-sub">Trading deck · {timeframe} · alt panel {lower_panel} · {signal_text}</div>
            </div>
            <div>
              <div class="tv-price mono">{format_money(last_close, 4 if last_close < 1 else 2)}</div>
              <div class="tv-sub" style="text-align:right;color:{'#12b886' if day_change >= 0 else '#ff5a5f'}">
                24s {day_change:+.2f}% · bar {intrabar_change:+.2f}%
              </div>
            </div>
          </div>
          <div class="tv-stats">
            <div class="tv-stat"><div class="tv-stat-label">Open</div><div class="tv-stat-value mono">{format_money(open_price, 4 if open_price < 1 else 2)}</div></div>
            <div class="tv-stat"><div class="tv-stat-label">High</div><div class="tv-stat-value mono">{format_money(high_price, 4 if high_price < 1 else 2)}</div></div>
            <div class="tv-stat"><div class="tv-stat-label">Low</div><div class="tv-stat-value mono">{format_money(low_price, 4 if low_price < 1 else 2)}</div></div>
            <div class="tv-stat"><div class="tv-stat-label">Volume</div><div class="tv-stat-value mono">{compact_money(latest_volume)}</div></div>
            <div class="tv-stat"><div class="tv-stat-label">RSI 14</div><div class="tv-stat-value mono">{rsi_text}</div></div>
            <div class="tv-stat"><div class="tv-stat-label">VWAP</div><div class="tv-stat-value mono">{vwap_text}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_detail_chart(
    symbol: str,
    timeframe: str,
    exchange_name: str,
    signal_map: dict[str, dict[str, Any]],
    selected_row: dict[str, Any] | None,
    *,
    lookback_days: int,
    chart_style: str,
    overlays: list[str],
    lower_panel: str,
) -> None:
    df = fetch_ohlcv(symbol, timeframe, exchange_name, days=lookback_days)
    if df.empty:
        st.info("Secilen coin icin mum verisi alinamadi.")
        return

    latest_signal = signal_map.get(symbol)
    render_tradingview_header(
        symbol=symbol,
        selected_row=selected_row,
        timeframe=timeframe,
        lower_panel=lower_panel,
        latest_signal=latest_signal,
        df=df,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    if chart_style == "Mum":
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Fiyat",
                increasing_line_color="#12b886",
                decreasing_line_color="#ff5a5f",
                increasing_fillcolor="#12b886",
                decreasing_fillcolor="#ff5a5f",
            ),
            row=1,
            col=1,
        )
    elif chart_style == "Alan":
        fill_color = "rgba(15,111,255,0.28)"
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#56ccf2", width=2.5),
                fill="tozeroy",
                fillcolor=fill_color,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#f7fbff", width=2.4),
            ),
            row=1,
            col=1,
        )

    overlay_map = {
        "EMA20": ("ema20", "#ff9f1c"),
        "EMA50": ("ema50", "#0f6fff"),
        "EMA200": ("ema200", "#c084fc"),
        "VWAP": ("vwap", "#56ccf2"),
    }
    for overlay_name, (column, color) in overlay_map.items():
        if overlay_name in overlays:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[column],
                    mode="lines",
                    name=overlay_name,
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

    if latest_signal:
        signal_price = safe_float(latest_signal.get("price"))
        signal_side = latest_signal.get("side", "")
        if "Sinyal" in overlays and signal_price:
            signal_color = PALETTE["success"] if signal_side == "BUY" else PALETTE["danger"] if signal_side == "SELL" else PALETTE["warning"]
            fig.add_hline(
                y=signal_price,
                line_dash="dot",
                line_color=signal_color,
                annotation_text=f"{signal_side} {signal_price:,.4f}",
                annotation_position="top left",
                row=1,
                col=1,
            )

    bar_colors = [
        "rgba(18,184,134,0.60)" if close >= open_ else "rgba(255,90,95,0.58)"
        for open_, close in zip(df["open"], df["close"])
    ]
    if lower_panel == "Hacim":
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Hacim",
                marker_color=bar_colors,
                opacity=0.9,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title="Volume", row=2, col=1)
    elif lower_panel == "RSI":
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["rsi14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="#56ccf2", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,90,95,0.42)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(18,184,134,0.42)", row=2, col=1)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1)
    else:
        macd_colors = [
            "rgba(18,184,134,0.75)" if value >= 0 else "rgba(255,90,95,0.75)"
            for value in df["macd_hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["macd_hist"],
                name="MACD Hist",
                marker_color=macd_colors,
                opacity=0.85,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["macd"],
                mode="lines",
                name="MACD",
                line=dict(color="#56ccf2", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["macd_signal"],
                mode="lines",
                name="Signal",
                line=dict(color="#ff9f1c", width=2),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title="MACD", row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    dark_plot_layout(fig, height=620)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoscale", "toImage"],
        },
    )
    st.markdown(
        '<div class="tv-help">Karttan coin sec, timeframe veya alt paneli degistir, grafiği daha taktiksel oku.</div>',
        unsafe_allow_html=True,
    )


def render_overview_tab(
    *,
    snapshot: dict[str, Any],
    m4: dict[str, Any],
    m5: dict[str, Any],
    m6: dict[str, Any],
    m7: dict[str, Any],
    m8: dict[str, Any],
    market_rows: list[dict[str, Any]],
    signal_map: dict[str, dict[str, Any]],
) -> None:
    render_section_header(
        "Tek Bakista Genel Durum",
        "Equity eğrisi saate göre — işlem noktalarına fare ile git, detay gör.",
        f"M4 {m4.get('freshness', '—')} · M5 {m5.get('freshness', '—')}",
    )

    # Açık pozisyon coin dağılımı (sade — coin adı + $ tutarı)
    st.markdown(
        '<div class="panel-title" style="margin-bottom:0.3rem">Açık Pozisyonlar</div>',
        unsafe_allow_html=True,
    )
    render_open_positions_allocation(m4, m5, m6, m7, m8)

    left, right = st.columns([7, 5])
    with left:
        render_equity_timeline_with_trades(m4, m5)
    with right:
        render_market_breadth_chart(market_rows)

    render_section_header(
        "Canli Piyasa Kartlari",
        "Takip ettigin coinleri hizli okumak icin sade kartlar.",
    )
    render_market_cards(market_rows[:12], signal_map)

    render_section_header(
        "M4 ve M5 Kisa Tablo",
        "Getiri, drawdown ve islem kalitesini yanyana oku.",
    )
    compare_rows = [
        {
            "Metrik": "Toplam PnL",
            "M4": format_money(m4.get("total_pnl", 0.0)),
            "M5": format_money(m5.get("total_pnl", 0.0)),
            "Onde": "M4" if safe_float(m4.get("total_pnl")) > safe_float(m5.get("total_pnl")) else "M5",
        },
        {
            "Metrik": "Getiri",
            "M4": format_pct(m4.get("return_pct", 0.0)),
            "M5": format_pct(m5.get("return_pct", 0.0)),
            "Onde": "M4" if safe_float(m4.get("return_pct")) > safe_float(m5.get("return_pct")) else "M5",
        },
        {
            "Metrik": "Max DD",
            "M4": format_pct(m4.get("drawdown_pct", 0.0), signed=False),
            "M5": format_pct(m5.get("drawdown_pct", 0.0), signed=False),
            "Onde": "M4" if safe_float(m4.get("drawdown_pct"), 999) < safe_float(m5.get("drawdown_pct"), 999) else "M5",
        },
        {
            "Metrik": "Win Rate",
            "M4": format_pct(m4.get("win_rate", 0.0)),
            "M5": format_pct(m5.get("win_rate", 0.0)),
            "Onde": "M4" if safe_float(m4.get("win_rate")) > safe_float(m5.get("win_rate")) else "M5",
        },
        {
            "Metrik": "Profit Factor",
            "M4": "inf" if math.isinf(safe_float(m4.get("profit_factor"))) else f"{safe_float(m4.get('profit_factor')):.2f}",
            "M5": "inf" if math.isinf(safe_float(m5.get("profit_factor"))) else f"{safe_float(m5.get('profit_factor')):.2f}",
            "Onde": "M4" if safe_float(m4.get("profit_factor")) > safe_float(m5.get("profit_factor")) else "M5",
        },
        {
            "Metrik": "Ortalama Islem",
            "M4": format_money(m4.get("avg_trade", 0.0)),
            "M5": format_money(m5.get("avg_trade", 0.0)),
            "Onde": "M4" if safe_float(m4.get("avg_trade")) > safe_float(m5.get("avg_trade")) else "M5",
        },
    ]
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)


def render_market_tab(
    *,
    market_rows: list[dict[str, Any]],
    signal_map: dict[str, dict[str, Any]],
    exchange_name: str,
    watchlist: list[str],
) -> None:
    n_coins = len(market_rows)
    valid   = sum(1 for r in market_rows if not r.get("error") and r.get("last"))
    render_section_header(
        "Canli Piyasa Ekrani",
        f"{valid}/{n_coins} coin fiyatı alındı · Karta tıkla → grafik aşağıda açılır.",
    )
    selected_symbol = ensure_market_selection(watchlist)
    render_market_cards(
        market_rows,
        signal_map,
        interactive=True,
        key_prefix="live_market_pick",
    )
    selected_symbol = ensure_market_selection(watchlist)

    controls_top, controls_mid, controls_right = st.columns([4, 4, 4])
    selected_symbol = controls_top.selectbox(
        "Secili coin",
        options=watchlist,
        format_func=symbol_label,
        key="market_detail_symbol",
    )
    timeframe = controls_mid.radio(
        "Zaman dilimi",
        options=["15m", "1h", "4h", "1d"],
        index=1,
        horizontal=True,
        key="market_detail_timeframe",
    )
    chart_style = controls_right.radio(
        "Grafik tipi",
        options=["Mum", "Çizgi", "Alan"],
        index=0,
        horizontal=True,
        key="market_detail_chart_style",
    )

    range_cols = st.columns([3, 3, 6])
    lookback_days = range_cols[0].number_input(
        "Kac gun gosterilsin",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="market_detail_days",
    )
    estimated_bars = int(lookback_days * (24 * 60) / timeframe_to_minutes(timeframe))
    range_cols[1].metric("Tahmini bar", f"{estimated_bars:,}")
    range_cols[2].caption("Gun sayisini sen belirle; sistem secilen timeframe'e gore gerekli mum sayisini otomatik ceker.")

    overlay_cols = st.columns([7, 5])
    overlays = overlay_cols[0].multiselect(
        "Ust gosterimler",
        options=["EMA20", "EMA50", "EMA200", "VWAP", "Sinyal"],
        default=["EMA20", "EMA50", "VWAP", "Sinyal"],
        key="market_detail_overlays",
    )
    lower_panel = overlay_cols[1].radio(
        "Alt panel",
        options=["Hacim", "RSI", "MACD"],
        index=0,
        horizontal=True,
        key="market_detail_lower_panel",
    )

    selected_row = next((row for row in market_rows if row["symbol"] == selected_symbol), None)
    if selected_row:
        metrics = st.columns(5)
        metrics[0].metric("Anlik Fiyat", format_money(selected_row["last"], 4 if selected_row["last"] < 1 else 2))
        metrics[1].metric("24s Degisim", format_pct(selected_row["change_pct"]), delta_color=style_metric_delta(selected_row["change_pct"]))
        metrics[2].metric("Gunluk Yüksek", format_money(selected_row["high"], 4 if selected_row["high"] < 1 else 2))
        metrics[3].metric("Gunluk Dusuk", format_money(selected_row["low"], 4 if selected_row["low"] < 1 else 2))
        metrics[4].metric("Gunluk Aralik", f"{selected_row['range_pos']:.0f}%")

    render_market_detail_chart(
        selected_symbol,
        timeframe,
        exchange_name,
        signal_map,
        selected_row,
        lookback_days=int(lookback_days),
        chart_style=chart_style,
        overlays=overlays,
        lower_panel=lower_panel,
    )

    render_section_header(
        "Market Board",
        "Canli piyasa snapshot ile sinyal verisini ayni tabloda birlestirir.",
    )
    board_df = build_market_board(market_rows, signal_map)
    st.dataframe(board_df, use_container_width=True, hide_index=True)


def render_model_summary_card(model: dict[str, Any], accent: str, subtitle: str) -> None:
    if not model.get("available"):
        st.info(f"{model['name']} state dosyasi henuz olusmamis.")
        return

    pf_value = model["profit_factor"]
    pf_text = "inf" if math.isinf(pf_value) else f"{pf_value:.2f}"

    # Bakiye dağılımı: Bakiye(equity) = serbest nakit + açık pozisyonların GÜNCEL DEĞERİ.
    # Pozisyon değerleri maliyet + açık K/Z (unrealized) içerir → Serbest nakit + LONG + SHORT = Bakiye.
    state       = model.get("state") or {}
    free_cash   = safe_float(state.get("balance"))
    open_pos    = state.get("open_positions", []) or []
    long_val    = 0.0   # LONG: güncel piyasa değeri (maliyet + açık K/Z)
    short_val   = 0.0   # SHORT: equity katkısı (marjin + açık K/Z)
    open_upnl   = 0.0   # toplam açık (gerçekleşmemiş) K/Z
    n_long = n_short = 0
    for p in open_pos:
        ep   = safe_float(p.get("entry_price"))
        size = safe_float(p.get("size"))
        upnl = safe_float(p.get("unrealized_pnl", 0))
        open_upnl += upnl
        if p.get("is_short", False):
            n_short += 1
            margin = safe_float(p.get("margin_locked", 0))
            if margin < 0.01:
                stop  = safe_float(p.get("stop_price", 0))
                trail = safe_float(p.get("trail_price", 0))
                ref   = max(stop, trail)
                margin = abs(ref - ep) * size if ref > ep else ep * size * 0.02
            short_val += margin + upnl
        else:
            n_long += 1
            cost = safe_float(p.get("cost", 0))
            if cost <= 1.0:
                cost = ep * size
            long_val += cost + upnl

    # Pozisyon satırı — değerler GÜNCEL (maliyet + açık K/Z), böylece nakit + pozisyon = Bakiye
    pos_parts = []
    if n_long > 0:
        pos_parts.append(f"📈 LONG ({n_long}): {format_money(long_val)}")
    if n_short > 0:
        pos_parts.append(f"📉 SHORT ({n_short}): {format_money(short_val)}")
    if not pos_parts:
        pos_parts.append("Pozisyon yok")
    pos_line  = " &nbsp;|&nbsp; ".join(pos_parts)
    upnl_txt  = f" &nbsp;|&nbsp; Açık K/Z: {format_money(open_upnl)}" if open_pos else ""

    st.markdown(
        f"""
        <div class="panel" style="border-top: 6px solid {accent};">
          <div class="panel-title">{model["name"]} · {subtitle}</div>
          <div class="panel-big">{format_pct(model.get("equity_return", model.get("return_pct", 0.0)))}</div>
          <div class="panel-copy">
            Bakiye {format_money(_model_equity(model))} ·
            Max DD {model["drawdown_pct"]:.1f}% ·
            WR {model["win_rate"]:.1f}% ·
            PF {pf_text}
          </div>
          <div class="panel-copy" style="margin-top:0.4rem; font-size:0.82rem; color:#6b7280;">
            💰 Serbest nakit: {format_money(free_cash)} &nbsp;|&nbsp; {pos_line}{upnl_txt}
          </div>
          <div class="panel-copy" style="margin-top:0.35rem; font-size:0.82rem; color:#6b7280;">
            📐 Sharpe: {_fmt_ratio(model.get("sharpe"))} ·
            Omega: {_fmt_ratio(model.get("omega"))} ·
            Reel getiri (enf. %3/yıl): {_fmt_real(model.get("real_return_pct"))}
          </div>
          <div class="badge-row" style="margin-top:0.8rem">
            <span class="badge badge-neutral">{model["trade_count"]} islem</span>
            <span class="badge badge-neutral">{len(open_pos)} acik pozisyon</span>
            <span class="badge badge-primary">{model["freshness"]}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt_ratio(v) -> str:
    """Sharpe/Omega gösterimi — yeterli günlük veri yoksa '—' (≥5 gün gerekir)."""
    if v is None:
        return "—"
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


def _fmt_real(v) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def render_state_trade_table(model: dict[str, Any], limit: int = 12) -> None:
    df = model.get("trade_df")
    if df is None or df.empty:
        st.info("Kapali islem yok.")
        return

    show = df.sort_values("exit_dt", ascending=False, na_position="last").head(limit).copy()
    show["PnL"] = show["pnl"].map(lambda value: format_money(value))
    show["PnL %"] = show["pnl_pct"].map(lambda value: format_pct(value))
    show["Giris"] = show["entry_price"].map(lambda value: format_money(value, 4 if value < 1 else 2))
    show["Cikis"] = show["exit_price"].map(lambda value: format_money(value, 4 if value < 1 else 2))
    show["Bekleme"] = show["bars_held"].map(lambda value: f"{safe_int(value)} bar")
    show["Cikis Gunu"] = show["exit_date"]
    show = show[["label", "side", "PnL", "PnL %", "Giris", "Cikis", "Bekleme", "reason", "Cikis Gunu"]]
    show.columns = ["Coin", "Yon", "PnL", "PnL %", "Giris", "Cikis", "Bekleme", "Sebep", "Tarih"]
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_coin_benchmark_tab(m4: dict[str, Any], m5: dict[str, Any], m6: dict[str, Any], m7: dict[str, Any], m8: dict[str, Any]) -> None:
    """v15: Bot başlatıldığı andan itibaren her coinin fiyat değişimini göster.

    Kullanıcı isteği: restart anından itibaren coinlerin değer artışını inceleme.
    Veri kaynağı: state JSON'daki coin_benchmarks — her modelin kendi backtestinden.
    M5 öncelikli, yoksa M4, yoksa M6.
    """
    render_section_header(
        "Coin Performansi — Bot Baslatilmasindan Bu Yana",
        "Bot baslatildiginda her coinin fiyatini kayit altına aldik. Asagida o andan simdi kadar ne kadar degistiklerini goruyorsunuz.",
    )

    # Veri kaynagi — M5 öncelikli
    benchmarks: list[dict] = []
    source_label = ""
    for model, label in [(m8, "M8"), (m5, "M5"), (m7, "M7"), (m4, "M4"), (m6, "M6")]:
        state = (model or {}).get("state") or {}
        bm = state.get("coin_benchmarks") or []
        if bm:
            benchmarks = bm
            source_label = label
            break

    if not benchmarks:
        st.info("Coin benchmark verisi henüz yok. Bot ilk çalışmasını tamamladığında burada görünecek.")
        return

    # Özet metrikler
    rising = sum(1 for b in benchmarks if b.get("pct_change", 0) > 0)
    falling = sum(1 for b in benchmarks if b.get("pct_change", 0) < 0)
    avg_chg = sum(b.get("pct_change", 0) for b in benchmarks) / len(benchmarks) if benchmarks else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Takip Edilen Coin", len(benchmarks))
    col2.metric("Yükselen 📈", rising)
    col3.metric("Düşen 📉", falling)
    col4.metric("Ortalama Değişim", f"{avg_chg:+.2f}%")

    # Tablo
    rows = []
    for b in benchmarks:
        sym = b.get("symbol", "")
        pct = b.get("pct_change", 0.0)
        start_px = b.get("start_price", 0.0)
        cur_px = b.get("current_price", 0.0)
        rows.append({
            "Coin": symbol_label(sym),
            "Baslangic $": format_money(start_px, 4 if start_px < 1 else 2),
            "Simdi $": format_money(cur_px, 4 if cur_px < 1 else 2),
            "Degisim %": f"{pct:+.2f}%",
            "Durum": "📈 Yukselis" if pct > 3 else ("📉 Dusus" if pct < -3 else "➡ Stabil"),
            "Baslangic Tarihi": b.get("start_date", ""),
        })

    df_bm = pd.DataFrame(rows)
    st.dataframe(df_bm, use_container_width=True, hide_index=True, height=520)

    st.caption(
        f"Veri kaynagi: {source_label} modeli · "
        f"Baslangic: {benchmarks[0].get('start_date', '?')} · "
        f"Bot ilk calistirildigi andan itibaren her guncelleme ile fiyatlar guncellenir."
    )


def render_state_detailed_trade_table(model: dict[str, Any], limit: int = 50) -> None:
    """v14: Kullanıcı isteği — detaylı işlem tablosu.

    Her trade için: Kripto, Yön, Adet, Maliyet, Alış tarihi (dk), Satış tarihi (dk),
    Giriş/Çıkış fiyatı, Kar/Zarar (₺ + %), Durum.
    """
    df = model.get("trade_df")
    if df is None or df.empty:
        st.info("Henuz islem yok.")
        return

    show = df.sort_values("exit_dt", ascending=False, na_position="last").head(limit).copy()

    # Format kolonları
    show["Kripto"] = show["label"]
    show["Yon"] = show["side"]
    # Adet — küçük coinlerde 6 ondalık, büyüklerde 4
    show["Adet"] = show["size"].map(lambda v: f"{v:.6f}" if v < 1 else f"{v:.4f}" if v < 100 else f"{v:.2f}")
    show["Maliyet"] = show["cost"].map(lambda v: format_money(v))
    # Tarihler — dakika cinsi (state JSON'dan saatli geliyor)
    show["Alis"] = show["entry_date"]
    show["Satis"] = show["exit_date"]
    show["Giris $"] = show["entry_price"].map(lambda v: format_money(v, 4 if v < 1 else 2))
    show["Cikis $"] = show["exit_price"].map(lambda v: format_money(v, 4 if v < 1 else 2))
    show["Kar/Zarar"] = show["pnl"].map(lambda v: format_money(v))
    show["K/Z %"] = show["pnl_pct"].map(lambda v: format_pct(v))
    # Durum: kar / zarar metin
    show["Durum"] = show["pnl"].map(lambda v: "✅ Kar" if v > 0 else ("❌ Zarar" if v < 0 else "⚖ Nötr"))

    show = show[["Kripto", "Yon", "Adet", "Maliyet", "Alis", "Satis",
                 "Giris $", "Cikis $", "Kar/Zarar", "K/Z %", "Durum"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)
    st.caption(f"Toplam {len(df)} kapali islem, son {min(limit, len(show))} tanesi gosteriliyor.")


def render_state_open_positions(state: dict[str, Any] | None) -> None:
    positions = (state or {}).get("open_positions") or []
    if not positions:
        st.info("Acik pozisyon yok.")
        return

    rows = []
    for pos in positions:
        entry   = safe_float(pos.get("entry_price"))
        upnl    = safe_float(pos.get("unrealized_pnl"))
        cost    = safe_float(pos.get("cost"))
        # cost küçükse (eski SHORT margin rezervi) entry×size kullan
        if cost < 1.0 and entry > 0:
            cost = entry * safe_float(pos.get("size"))
        upnl_pct = (upnl / cost * 100) if cost > 0 else 0.0
        # Yon: is_short → SHORT/LONG
        is_short = pos.get("is_short", False)
        yon = "SHORT" if is_short else "LONG"
        rows.append(
            {
                "Coin":  symbol_label(pos.get("symbol", "")),
                "Yon":   yon,
                "Giris": format_money(entry, 4 if entry < 1 else 2),
                "Maliyet": format_money(cost),
                "PnL":   format_money(upnl),
                "PnL %": format_pct(upnl_pct),
                "Stop":  format_money(safe_float(pos.get("stop_price")), 4 if safe_float(pos.get("stop_price")) < 1 else 2),
                "Bars":  safe_int(pos.get("bars_held")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_model_symbol_comparison(m4: dict[str, Any], m5: dict[str, Any]) -> None:
    df4 = m4.get("trade_df")
    df5 = m5.get("trade_df")
    if (df4 is None or df4.empty) and (df5 is None or df5.empty):
        st.info("Sembol bazli model karsilastirmasi icin trade verisi yok.")
        return

    g4 = (
        df4.groupby("label")["pnl"].sum().rename("M4").reset_index()
        if df4 is not None and not df4.empty
        else pd.DataFrame(columns=["label", "M4"])
    )
    g5 = (
        df5.groupby("label")["pnl"].sum().rename("M5").reset_index()
        if df5 is not None and not df5.empty
        else pd.DataFrame(columns=["label", "M5"])
    )
    merged = pd.merge(g4, g5, on="label", how="outer").fillna(0.0)
    merged = merged.sort_values("label")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=merged["label"], y=merged["M4"], name="M4", marker_color=PALETTE["primary"]))
    fig.add_trace(go.Bar(x=merged["label"], y=merged["M5"], name="M5", marker_color=PALETTE["success"]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(ticksuffix=" USDT")
    plot_layout(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)


def render_models_tab(m4: dict[str, Any], m5: dict[str, Any], m6: dict[str, Any], m7: dict[str, Any], m8: dict[str, Any], ortak: dict[str, Any] | None = None) -> None:
    render_section_header(
        "M4 / M5 / M6 / M7 / M8 / ORTAK Derin Karsilastirma",
        "Ayni donemde hangi model daha temiz, daha guclu ve daha verimli calismis gorebil.",
    )
    card_m4, card_m5, card_m6, card_m7, card_m8, card_ortak = st.columns(6)
    with card_m4:
        render_model_summary_card(m4, PALETTE["primary"], "Stabil referans")
    with card_m5:
        render_model_summary_card(m5, PALETTE["success"], "Risk dengeli")
    with card_m6:
        render_model_summary_card(m6, PALETTE["warning"], "Agresif M6")
    with card_m7:
        render_model_summary_card(m7, "#e879f9", "Seçici trend (M7)")
    with card_m8:
        render_model_summary_card(m8, "#f97316", "Hacim odaklı (M8)")
    with card_ortak:
        if ortak is not None:
            render_model_summary_card(ortak, "#facc15", "Tahsisçi (M9-sleeve)")
            _ost = (ortak.get("state") or {})
            _alloc = _ost.get("allocated")
            _oi = (_ost.get("oi_log") or [])
            _oi_txt = f"{float(_oi[-1][1]):,.0f} BTC" if _oi else "—"
            st.caption(
                f"Tahsis: {'🟢 AÇIK (M9 aynalanıyor)' if _alloc else '💤 nakitte'} · "
                f"Shadow M9: {_ost.get('shadow_return_pct', '—')}% · BTC OI: {_oi_txt}"
            )

    render_section_header(
        "Sembol Bazli Model Farki",
        "Hangi coinlerde M4 ya da M5 daha iyi calismis hemen ayir.",
    )
    render_model_symbol_comparison(m4, m5)

    trade_m4, trade_m5, trade_m6, trade_m7, trade_m8 = st.columns(5)
    with trade_m4:
        st.markdown("##### M4 Acik Pozisyonlar")
        render_state_open_positions(m4.get("state"))
        st.markdown("##### M4 Son Islemler")
        render_state_trade_table(m4)
    with trade_m5:
        st.markdown("##### M5 Acik Pozisyonlar")
        render_state_open_positions(m5.get("state"))
        st.markdown("##### M5 Son Islemler")
        render_state_trade_table(m5)
    with trade_m6:
        st.markdown("##### M6 Acik Pozisyonlar")
        render_state_open_positions(m6.get("state"))
        st.markdown("##### M6 Son Islemler")
        render_state_trade_table(m6)
    with trade_m7:
        st.markdown("##### M7 Acik Pozisyonlar")
        render_state_open_positions(m7.get("state"))
        st.markdown("##### M7 Son Islemler")
        render_state_trade_table(m7)
    with trade_m8:
        st.markdown("##### M8 Acik Pozisyonlar")
        render_state_open_positions(m8.get("state"))
        st.markdown("##### M8 Son Islemler")
        render_state_trade_table(m8)

    # v14: Detayli islem gecmisi — her model icin ayri tablo
    # Kullanici istegi: kripto, adet, alis/satis tarihi (dakika cinsi), kar/zarar durumu
    render_section_header(
        "Detayli Islem Gecmisi",
        "Her model icin: kripto, adet, alis/satis zamani (dakika cinsinden), maliyet ve kar/zarar durumu.",
    )
    st.markdown("##### M4 — Detayli Islem Gecmisi")
    render_state_detailed_trade_table(m4)
    st.markdown("##### M5 — Detayli Islem Gecmisi")
    render_state_detailed_trade_table(m5)
    st.markdown("##### M6 — Detayli Islem Gecmisi")
    render_state_detailed_trade_table(m6)
    st.markdown("##### M7 — Detayli Islem Gecmisi")
    render_state_detailed_trade_table(m7)
    st.markdown("##### M8 — Detayli Islem Gecmisi")
    render_state_detailed_trade_table(m8)


def render_active_positions(open_positions: list[dict[str, Any]]) -> None:
    if not open_positions:
        st.info("Aktif bot tarafinda acik pozisyon yok.")
        return

    rows = []
    for pos in open_positions:
        cost_basis = safe_float(pos.get("cost_basis"))
        unrealized = safe_float(pos.get("unrealized_pnl"))
        upnl_pct = (unrealized / cost_basis * 100) if cost_basis else 0.0
        entry = safe_float(pos.get("entry_price"))
        rows.append(
            {
                "Coin": symbol_label(pos.get("symbol", "")),
                "Giris": format_money(entry, 4 if entry < 1 else 2),
                "Boyut": f"{safe_float(pos.get('position_size')):.4f}",
                "Maliyet": format_money(cost_basis),
                "Acik PnL": format_money(unrealized),
                "PnL %": format_pct(upnl_pct),
                "Stop": format_money(pos.get("stop_price", 0.0), 4 if safe_float(pos.get("stop_price")) < 1 else 2),
                "Acilis": str(pos.get("opened_at", ""))[:16].replace("T", " "),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_db_equity_chart(snapshot: dict[str, Any]) -> None:
    equity = snapshot.get("equity") or []
    if not equity or len(equity) < 3:
        st.info("Equity verisi henuz yeterli degil.")
        return

    df = pd.DataFrame(equity)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df = df.dropna(subset=["timestamp", "balance"])
    if df.empty:
        st.info("Equity verisi gosterilemiyor.")
        return

    rolling_peak = df["balance"].cummax()
    drawdown = (df["balance"] - rolling_peak) / rolling_peak * 100
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["balance"],
            mode="lines",
            name="Bakiye",
            line=dict(color=PALETTE["primary"], width=3),
            fill="tozeroy",
            fillcolor="rgba(15,111,255,0.10)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=PALETTE["danger"], width=2),
            fill="tozeroy",
            fillcolor="rgba(255,90,95,0.10)",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    plot_layout(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)


def render_active_trade_analytics(closed_df: pd.DataFrame) -> None:
    if closed_df.empty:
        st.info("Kapali islem verisi gelmedigi icin aktif bot analitigi bekliyor.")
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        hist = go.Figure()
        positive = closed_df.loc[closed_df["pnl"] > 0, "pnl"]
        negative = closed_df.loc[closed_df["pnl"] <= 0, "pnl"]
        if not positive.empty:
            hist.add_trace(go.Histogram(x=positive, name="Kazanan", marker_color=PALETTE["success"], opacity=0.78))
        if not negative.empty:
            hist.add_trace(go.Histogram(x=negative, name="Kaybeden", marker_color=PALETTE["danger"], opacity=0.72))
        hist.update_layout(barmode="overlay")
        hist.update_xaxes(ticksuffix=" USDT")
        plot_layout(hist, height=280)
        st.plotly_chart(hist, use_container_width=True)

    with chart_right:
        by_symbol = closed_df.groupby("label")["pnl"].sum().sort_values()
        fig = go.Figure(
            go.Bar(
                x=by_symbol.values,
                y=by_symbol.index,
                orientation="h",
                marker_color=[PALETTE["success"] if value >= 0 else PALETTE["danger"] for value in by_symbol.values],
                text=[f"{value:+.2f}" for value in by_symbol.values],
                textposition="outside",
            )
        )
        fig.add_vline(x=0, line_color="rgba(16,35,63,0.20)")
        fig.update_xaxes(ticksuffix=" USDT")
        plot_layout(fig, height=280, legend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_execution_tab(snapshot: dict[str, Any]) -> None:
    render_section_header(
        "Islem Merkezi",
        "Aktif botun pozisyonlari, son islemleri ve equity akisina odaklan.",
    )
    open_positions = snapshot.get("open_positions") or []
    closed_df = prepare_db_trades(snapshot.get("closed") or [])

    left, right = st.columns([5, 7])
    with left:
        st.markdown("##### Acik Pozisyonlar")
        render_active_positions(open_positions)
    with right:
        st.markdown("##### Son Kapali Islemler")
        if closed_df.empty:
            st.info("Henuz kapali islem yok.")
        else:
            trades_table = closed_df.head(18).copy()
            trades_table["PnL"] = trades_table["pnl"].map(format_money)
            trades_table["Giris"] = trades_table["entry_price"].map(lambda value: format_money(value, 4 if value < 1 else 2))
            trades_table["Cikis"] = trades_table["exit_price"].map(lambda value: format_money(value, 4 if value < 1 else 2))
            trades_table["Zaman"] = trades_table["closed_at"].astype(str).str[:16].str.replace("T", " ")
            trades_table = trades_table[["label", "PnL", "Giris", "Cikis", "reason", "Zaman"]]
            trades_table.columns = ["Coin", "PnL", "Giris", "Cikis", "Sebep", "Zaman"]
            st.dataframe(trades_table, use_container_width=True, hide_index=True)

    st.markdown("##### Equity ve Drawdown")
    render_db_equity_chart(snapshot)

    st.markdown("##### Aktif Bot Analitigi")
    render_active_trade_analytics(closed_df)


def render_config_panel(config: dict[str, Any]) -> None:
    exchange_cfg = config.get("exchange") or {}
    trading_cfg = config.get("trading") or {}
    risk_cfg = config.get("risk") or {}
    info_rows = [
        ("Exchange", exchange_cfg.get("name", "binance")),
        ("Timeframe", trading_cfg.get("timeframe", "1h")),
        ("Izlenen coin", str(len(trading_cfg.get("symbols") or []))),
        ("Risk / islem", f"{safe_float(risk_cfg.get('risk_per_trade')) * 100:.2f}%"),
        ("Gunluk max loss", f"{safe_float(risk_cfg.get('daily_max_loss')) * 100:.2f}%"),
        ("Max acik pozisyon", str(safe_int(risk_cfg.get("max_open_positions"), 0))),
    ]
    html = ['<div class="panel"><div class="panel-title">Konfig Ozeti</div><div class="mini-list">']
    for key, value in info_rows:
        html.append(
            f'<div class="mini-row"><span class="mini-key">{key}</span>'
            f'<span class="mini-value">{value}</span></div>'
        )
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    symbols = trading_cfg.get("symbols") or []
    if symbols:
        chips = " ".join(f'<span class="badge badge-neutral">{symbol_label(symbol)}</span>' for symbol in symbols)
        st.markdown(
            f"""
            <div class="panel" style="margin-top:0.8rem">
              <div class="panel-title">Takip Listesi</div>
              <div class="badge-row" style="margin-top:0">{chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_signal_log(signal_df: pd.DataFrame) -> None:
    if signal_df.empty:
        st.info("Sinyal logu henuz dolmamis.")
        return

    filt_left, filt_mid, filt_right = st.columns(3)
    side_filter = filt_left.selectbox("Yon", ["Tumu", "BUY", "SELL", "HOLD"], key="signal_side_filter")
    symbols = ["Tumu"] + sorted(signal_df["symbol"].dropna().unique().tolist())
    symbol_filter = filt_mid.selectbox("Coin", symbols, format_func=lambda item: "Tumu" if item == "Tumu" else symbol_label(item), key="signal_symbol_filter")
    min_conf = filt_right.slider("Min skor", 0.0, 1.0, 0.0, 0.05, key="signal_conf_filter")

    df = signal_df.copy()
    if side_filter != "Tumu":
        df = df[df["side"] == side_filter]
    if symbol_filter != "Tumu":
        df = df[df["symbol"] == symbol_filter]
    df = df[df["confidence"] >= min_conf]

    if df.empty:
        st.info("Filtreye uyan sinyal kalmadi.")
        return

    show = df.sort_values("ts", ascending=False, na_position="last").head(120).copy()
    show["Zaman"] = show["timestamp"].astype(str).str[:16].str.replace("T", " ")
    show["Coin"] = show["label"]
    show["Fiyat"] = show["price"].map(lambda value: format_money(value, 4 if value < 1 else 2))
    show["Skor"] = show["confidence"].map(lambda value: f"{value:.2f}")
    show["RSI"] = show["rsi"].map(lambda value: "—" if math.isnan(value) else f"{value:.0f}")
    show["ADX"] = show["adx"].map(lambda value: "—" if math.isnan(value) else f"{value:.0f}")
    show["Sebep"] = show["reason"].astype(str).str.slice(0, 72)
    show = show[["Zaman", "Coin", "side", "Fiyat", "Skor", "RSI", "ADX", "Sebep"]]
    show.columns = ["Zaman", "Coin", "Yon", "Fiyat", "Skor", "RSI", "ADX", "Sebep"]
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_signal_distribution(signal_df: pd.DataFrame) -> None:
    if signal_df.empty:
        st.info("Dagilim icin sinyal verisi yok.")
        return

    counts = signal_df["side"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker_colors=[
                PALETTE["success"] if label == "BUY" else PALETTE["danger"] if label == "SELL" else PALETTE["warning"]
                for label in counts.index
            ],
        )
    )
    plot_layout(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)


def render_learning_summary_cards(snapshot: dict[str, Any], signal_df: pd.DataFrame) -> None:
    adapt_log = snapshot.get("adapt") or []
    scalp_learn = snapshot.get("scalp_learn") or []

    total_signals = len(signal_df)
    buy_count = safe_int((signal_df["side"] == "BUY").sum()) if not signal_df.empty else 0
    avg_conf = safe_float(signal_df["confidence"].mean()) if not signal_df.empty else 0.0
    strong_signals = safe_int((signal_df["confidence"] >= 0.60).sum()) if not signal_df.empty else 0
    adapting_symbols = len([row for row in scalp_learn if safe_int(row.get("trade_count")) > 0])

    cards = st.columns(5)
    cards[0].metric("Toplam Sinyal", f"{total_signals}", f"{buy_count} BUY")
    cards[1].metric("Ort. Güven", f"{avg_conf:.2f}", f"{strong_signals} güçlü sinyal")
    cards[2].metric("Adaptasyon Kaydı", str(len(adapt_log)), "walk-forward olay")
    cards[3].metric("Öğrenen Coin", str(adapting_symbols), "scalp öğrenme aktif")
    cards[4].metric("Sinyal Çeşidi", str(signal_df['symbol'].nunique() if not signal_df.empty else 0), "farklı coin")


def render_signal_quality_panel(signal_df: pd.DataFrame) -> None:
    if signal_df.empty:
        st.info("Sinyal kalitesi paneli için veri yok.")
        return

    quality_left, quality_right = st.columns([7, 5])

    with quality_left:
        conf_df = signal_df.copy()
        conf_df["bucket"] = pd.cut(
            conf_df["confidence"],
            bins=[0.0, 0.35, 0.5, 0.65, 1.0],
            labels=["Dusuk", "Izlenir", "Iyi", "Cok guclu"],
            include_lowest=True,
        )
        bucket_counts = conf_df["bucket"].value_counts().reindex(["Dusuk", "Izlenir", "Iyi", "Cok guclu"], fill_value=0)
        fig = go.Figure(
            go.Bar(
                x=bucket_counts.index,
                y=bucket_counts.values,
                marker_color=[PALETTE["danger"], PALETTE["warning"], PALETTE["primary"], PALETTE["success"]],
                text=bucket_counts.values,
                textposition="outside",
            )
        )
        fig.update_yaxes(title="Sinyal adedi")
        plot_layout(fig, height=260, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with quality_right:
        top_symbols = (
            signal_df.groupby("label")
            .agg(
                avg_conf=("confidence", "mean"),
                signal_count=("confidence", "size"),
                last_side=("side", "first"),
            )
            .sort_values(["avg_conf", "signal_count"], ascending=[False, False])
            .head(8)
            .reset_index()
        )
        top_symbols["avg_conf"] = top_symbols["avg_conf"].map(lambda value: f"{safe_float(value):.2f}")
        top_symbols.columns = ["Coin", "Ort. Skor", "Sinyal", "Son Yön"]
        st.markdown("##### En net çalışan coinler")
        st.dataframe(top_symbols, use_container_width=True, hide_index=True)


def render_signal_timeline(signal_df: pd.DataFrame) -> None:
    if signal_df.empty:
        st.info("Sinyal zaman çizgisi için veri yok.")
        return

    tl = signal_df.copy()
    tl = tl.dropna(subset=["ts"])
    if tl.empty:
        st.info("Zaman bilgisi eksik olduğu için çizgi üretilemedi.")
        return

    tl["hour"] = tl["ts"].dt.floor("h")
    hourly = tl.groupby(["hour", "side"]).size().reset_index(name="count")
    fig = go.Figure()
    color_map = {"BUY": PALETTE["success"], "SELL": PALETTE["danger"], "HOLD": PALETTE["warning"]}
    for side in ["BUY", "SELL", "HOLD"]:
        side_df = hourly[hourly["side"] == side]
        if side_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=side_df["hour"],
                y=side_df["count"],
                mode="lines+markers",
                name=side,
                line=dict(color=color_map.get(side, PALETTE["primary"]), width=2.5),
            )
        )
    fig.update_yaxes(title="Saatlik sinyal")
    plot_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_adaptation_story(snapshot: dict[str, Any]) -> None:
    adapt_log = snapshot.get("adapt") or []
    scalp_learn = snapshot.get("scalp_learn") or []

    story_left, story_right = st.columns([7, 5])

    with story_left:
        if adapt_log:
            st.markdown("##### Walk-forward değişim akışı")
            df = pd.DataFrame(adapt_log)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df["old_sharpe"], mode="lines+markers", name="Eski fitness", line=dict(color=PALETTE["warning"], width=2)))
            fig.add_trace(go.Scatter(y=df["new_sharpe"], mode="lines+markers", name="Yeni fitness", line=dict(color=PALETTE["success"], width=3)))
            fig.update_xaxes(title="Adaptasyon sırası")
            fig.update_yaxes(title="Fitness")
            plot_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Walk-forward adaptasyon henüz başlamamış.")

        if scalp_learn:
            df_sl = pd.DataFrame(scalp_learn)
            df_sl = df_sl[df_sl["trade_count"] > 0].sort_values("win_rate", ascending=False).head(10)
            if not df_sl.empty:
                st.markdown("##### Scalp öğrenme lider tablosu")
                show = df_sl[["symbol", "win_rate", "threshold", "pos_scale", "trade_count"]].copy()
                show.columns = ["Coin", "Win Rate", "Eşik", "Kelly", "Trade"]
                show["Win Rate"] = show["Win Rate"].map(lambda value: f"{safe_float(value) * 100:.1f}%")
                show["Eşik"] = show["Eşik"].map(lambda value: f"{safe_float(value):.3f}")
                show["Kelly"] = show["Kelly"].map(lambda value: f"{safe_float(value):.2f}x")
                st.dataframe(show, use_container_width=True, hide_index=True)

    with story_right:
        html = ['<div class="panel"><div class="panel-title">Bu Sayfayı Nasıl Oku</div><div class="mini-list">']
        guide_rows = [
            ("1. Sistem sağlığı", "Toplam sinyal, güven seviyesi ve adaptasyon yoğunluğu"),
            ("2. Kalite", "Hangi coinler net, hangi sinyaller daha güvenli"),
            ("3. Zamanlama", "Sinyaller gün içinde hangi saatlerde kümeleniyor"),
            ("4. Öğrenme", "Fitness artıyor mu, scalp tarafı hangi coinlerde iyileşiyor"),
            ("5. Ayarlar", "Sistemin hangi parametrelerle çalıştığını altta gör"),
        ]
        for key, value in guide_rows:
            html.append(
                f'<div class="mini-row"><span class="mini-key">{key}</span>'
                f'<span class="mini-value" style="text-align:left">{value}</span></div>'
            )
        html.append("</div></div>")
        st.markdown("".join(html), unsafe_allow_html=True)

        if adapt_log:
            latest = adapt_log[0]
            old_score = safe_float(latest.get("old_sharpe"))
            new_score = safe_float(latest.get("new_sharpe"))
            improvement = 0.0 if old_score == 0 else ((new_score - old_score) / abs(old_score)) * 100
            st.markdown(
                f"""
                <div class="panel" style="margin-top:0.8rem">
                  <div class="panel-title">Son Adaptasyon Özeti</div>
                  <div class="panel-big">{format_pct(improvement)}</div>
                  <div class="panel-copy">
                    {latest.get("symbol", "GENEL")} için fitness {old_score:.3f} → {new_score:.3f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif scalp_learn:
            df_sl = pd.DataFrame(scalp_learn)
            active = df_sl[df_sl["trade_count"] > 0]
            avg_wr = safe_float(active["win_rate"].mean()) * 100 if not active.empty else 0.0
            st.markdown(
                f"""
                <div class="panel" style="margin-top:0.8rem">
                  <div class="panel-title">Scalp Öğrenme Özeti</div>
                  <div class="panel-big">{avg_wr:.1f}%</div>
                  <div class="panel-copy">
                    Öğrenen coinlerin ortalama EMA win rate seviyesi.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_adaptation_panels(snapshot: dict[str, Any]) -> None:
    adapt_log = snapshot.get("adapt") or []
    scalp_learn = snapshot.get("scalp_learn") or []

    if adapt_log:
        render_section_header(
            "Walk-Forward Adaptasyon",
            "Aktif parametre degisimlerinin fitness sonucunu gor.",
        )
        df = pd.DataFrame(adapt_log)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["old_sharpe"], mode="lines+markers", name="Eski", line=dict(color=PALETTE["warning"], width=2)))
        fig.add_trace(go.Scatter(y=df["new_sharpe"], mode="lines+markers", name="Yeni", line=dict(color=PALETTE["success"], width=3)))
        fig.update_xaxes(title="Adaptasyon adimi")
        fig.update_yaxes(title="Fitness")
        plot_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

        table_rows = []
        for item in adapt_log[:20]:
            old_score = safe_float(item.get("old_sharpe"))
            new_score = safe_float(item.get("new_sharpe"))
            improvement = 0.0 if old_score == 0 else ((new_score - old_score) / abs(old_score)) * 100
            table_rows.append(
                {
                    "Zaman": str(item.get("timestamp", ""))[:16].replace("T", " "),
                    "Sembol": item.get("symbol", ""),
                    "Eski": f"{old_score:.3f}",
                    "Yeni": f"{new_score:.3f}",
                    "Iyilesme": format_pct(improvement),
                    "Trade": safe_int(item.get("trade_count")),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if scalp_learn:
        render_section_header(
            "Scalp Ogrenme Durumu",
            "Per-symbol win rate, threshold ve Kelly skala ozetini takip et.",
        )
        df = pd.DataFrame(scalp_learn)
        df = df[df["trade_count"] > 0].copy()
        if not df.empty:
            df = df.sort_values("win_rate")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df["win_rate"] * 100,
                    y=df["symbol"],
                    orientation="h",
                    marker_color=[PALETTE["success"] if value >= 0.5 else PALETTE["danger"] for value in df["win_rate"]],
                    text=(df["win_rate"] * 100).round(0).astype(int).astype(str) + "%",
                    textposition="outside",
                )
            )
            fig.add_vline(x=50, line_dash="dot", line_color="rgba(16,35,63,0.25)")
            fig.update_xaxes(title="Win rate %")
            plot_layout(fig, height=320, legend=False)
            st.plotly_chart(fig, use_container_width=True)

            table = df[["symbol", "win_rate", "threshold", "pos_scale", "trade_count"]].copy()
            table.columns = ["Sembol", "Win Rate", "Esik", "Kelly", "Trade"]
            table["Win Rate"] = table["Win Rate"].map(lambda value: f"{value * 100:.1f}%")
            table["Esik"] = table["Esik"].map(lambda value: f"{safe_float(value):.3f}")
            table["Kelly"] = table["Kelly"].map(lambda value: f"{safe_float(value):.2f}x")
            st.dataframe(table, use_container_width=True, hide_index=True)


def render_intelligence_tab(snapshot: dict[str, Any], config: dict[str, Any]) -> None:
    render_section_header(
        "Zeka ve Ogrenme",
        "Sistemin ne gordugunu, ne ogrendigini ve neden o sekilde davrandigini daha mantikli bir akista oku.",
    )

    signal_df = prepare_signal_df(snapshot.get("signals") or [])
    render_learning_summary_cards(snapshot, signal_df)

    render_section_header(
        "Sinyal Kalitesi",
        "Önce sinyallerin gücünü ve hangi coinlerde daha temiz çalıştığını gör.",
    )
    render_signal_quality_panel(signal_df)

    render_section_header(
        "Sinyal Zamanlaması",
        "Sinyallerin gün içinde hangi saatlerde kümelendiğini takip et.",
    )
    render_signal_timeline(signal_df)

    render_section_header(
        "Öğrenme Hikayesi",
        "Walk-forward ve scalp öğrenmenin gerçekten iyileşip iyileşmediğini oku.",
    )
    render_adaptation_story(snapshot)

    deep_left, deep_right = st.columns([7, 5])
    with deep_left:
        st.markdown("##### Detaylı Sinyal Logu")
        render_signal_log(signal_df)
    with deep_right:
        st.markdown("##### Sinyal Dağılımı")
        render_signal_distribution(signal_df)
        st.markdown("##### Sistem Ayarları")
        render_config_panel(config)

    render_adaptation_panels(snapshot)


def main() -> None:
    inject_styles()

    top_left, top_mid, top_right = st.columns([6, 3, 3])
    with top_left:
        st.caption("Veri kaynaklari: `dashboard/bot_state.db` + `live/state/*.json` + public exchange feed")
    with top_mid:
        refresh_enabled = st.toggle("Otomatik yenile", value=True, key="auto_refresh_toggle")
    with top_right:
        refresh_seconds = st.selectbox("Yenileme araligi", options=[30, 60, 300, 900], index=3, key="auto_refresh_seconds")

    action_left, action_right = st.columns([2, 10])
    with action_left:
        if st.button("Cache temizle ve yenile"):
            st.cache_data.clear()
            st.rerun()
    with action_right:
        st.caption(f"Lokal saat: {datetime.now(LOCAL_TZ).strftime('%d.%m.%Y %H:%M:%S')}")

    run_every = f"{refresh_seconds}s" if refresh_enabled else None

    @st.fragment(run_every=run_every)
    def render_live_body() -> None:
        config = load_config()
        snapshot = load_bot_snapshot()
        m4 = build_model_summary("M4", load_model_state(str(M4_STATE_PATH)))
        m5 = build_model_summary("M5", load_model_state(str(M5_STATE_PATH)))
        m6 = build_model_summary("M6", load_model_state(str(M6_STATE_PATH)))
        m7 = build_model_summary("M7", load_model_state(str(M7_STATE_PATH)))
        m8 = build_model_summary("M8", load_model_state(str(M8_STATE_PATH)))
        ortak = build_model_summary("ORTAK", load_model_state(str(ORTAK_STATE_PATH)))
        exchange_name = ((config.get("exchange") or {}).get("name") or "binance").lower()
        watchlist = build_watchlist(config, [m4, m5, m6, m7, m8])
        market_rows = fetch_market_snapshot(tuple(watchlist), exchange_name) if watchlist else []
        breadth = market_breadth(market_rows)
        signal_df = prepare_signal_df(snapshot.get("signals") or [])
        signal_map = build_signal_map(signal_df)

        render_top_metrics(snapshot=snapshot, breadth=breadth, m4=m4, m5=m5, m6=m6, m7=m7, m8=m8)

        tabs = st.tabs(
            [
                "Genel Bakis",
                "Canli Piyasa",
                "Modeller",
                "Coin Takip",
                "Islem Merkezi",
                "Zeka ve Ogrenme",
            ]
        )

        with tabs[0]:
            render_overview_tab(
                snapshot=snapshot,
                m4=m4,
                m5=m5,
                m6=m6,
                m7=m7,
                m8=m8,
                market_rows=market_rows,
                signal_map=signal_map,
            )
        with tabs[1]:
            render_market_tab(
                market_rows=market_rows,
                signal_map=signal_map,
                exchange_name=exchange_name,
                watchlist=watchlist,
            )
        with tabs[2]:
            render_models_tab(m4, m5, m6, m7, m8, ortak)
        with tabs[3]:
            render_coin_benchmark_tab(m4, m5, m6, m7, m8)
        with tabs[4]:
            render_execution_tab(snapshot)
        with tabs[5]:
            render_intelligence_tab(snapshot, config)

    render_live_body()


if __name__ == "__main__":
    main()
