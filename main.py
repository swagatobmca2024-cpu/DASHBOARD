"""
╔══════════════════════════════════════════════════════════════╗
║         APEX TRADING TERMINAL  — v2.0                       ║
║         Institutional-grade Portfolio & Market Dashboard    ║
╠══════════════════════════════════════════════════════════════╣
║  Requirements:                                              ║
║    streamlit>=1.32.0    plotly>=5.20.0    yfinance>=0.2.38  ║
║    pandas>=2.1.0        numpy>=1.26.0     requests>=2.31.0  ║
║    python-dateutil>=2.8.2                                   ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional, Any

# ─── third-party ──────────────────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Apex Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Apex Trading Terminal v2.0 — Institutional-grade Portfolio Dashboard"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IST = ZoneInfo("Asia/Kolkata")
US_EASTERN = ZoneInfo("America/New_York")

CACHE_TTL_QUOTE = 30        # seconds for live quote cache
CACHE_TTL_HISTORY = 300     # seconds for history cache
CACHE_TTL_NEWS = 600        # seconds for news cache

SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Energy", "Materials",
    "Real Estate", "Communication Services", "Utilities", "Unknown",
]

INITIAL_PORTFOLIO = [
    {"ticker": "AAPL",  "quantity": 10, "avg_cost": 150.00},
    {"ticker": "TSLA",  "quantity":  5, "avg_cost": 200.00},
    {"ticker": "GOOGL", "quantity":  8, "avg_cost": 120.00},
    {"ticker": "AMZN",  "quantity":  6, "avg_cost": 130.00},
    {"ticker": "MSFT",  "quantity": 12, "avg_cost": 280.00},
]

INITIAL_WATCHLIST = ["NVDA", "META", "NFLX", "DIS", "AMD"]
INITIAL_CASH      = 25_000.00

NEWS_PROVIDERS = {
    "Yahoo Finance RSS": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
}

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("apex")

# ══════════════════════════════════════════════════════════════════════════════
#  THEME / CSS
# ══════════════════════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

/* ── CSS Variables ── */
:root {
  --bg-base:       #0a0c10;
  --bg-surface:    #10141c;
  --bg-card:       #141820;
  --bg-hover:      #1a2030;
  --border:        #1e2535;
  --border-active: #2e3d5a;
  --accent:        #00c4ff;
  --accent-dim:    rgba(0,196,255,0.12);
  --accent-glow:   rgba(0,196,255,0.35);
  --green:         #00e676;
  --green-dim:     rgba(0,230,118,0.12);
  --red:           #ff4444;
  --red-dim:       rgba(255,68,68,0.12);
  --amber:         #ffa726;
  --text-primary:  #e8edf5;
  --text-secondary:#8899bb;
  --text-dim:      #4a5568;
  --mono:          'IBM Plex Mono', monospace;
  --sans:          'DM Sans', sans-serif;
  --display:       'Bebas Neue', sans-serif;
}

/* ── App Shell ── */
.stApp {
  background: var(--bg-base);
  font-family: var(--sans);
  color: var(--text-primary);
}
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Terminal Header ── */
.apex-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 1.2rem;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
  margin: -1rem -1.5rem 1.2rem -1.5rem;
  position: sticky;
  top: 0;
  z-index: 100;
}
.apex-logo {
  font-family: var(--display);
  font-size: 1.6rem;
  letter-spacing: 3px;
  color: var(--accent);
  text-shadow: 0 0 20px var(--accent-glow);
  line-height: 1;
}
.apex-logo span { color: var(--text-secondary); font-size: 0.85rem; font-family: var(--mono); letter-spacing: 1px; }
.apex-status-bar {
  display: flex;
  gap: 1.2rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-secondary);
  align-items: center;
}
.status-pill {
  padding: 0.2rem 0.6rem;
  border-radius: 3px;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 1px;
}
.pill-green { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,230,118,0.25); }
.pill-red   { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,68,68,0.25); }
.pill-amber { background: rgba(255,167,38,0.12); color: var(--amber); border: 1px solid rgba(255,167,38,0.25); }

/* ── Metric Cards ── */
.metric-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.7rem;
  margin-bottom: 1rem;
}
.m-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.m-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}
.m-card:hover { border-color: var(--border-active); }
.m-label {
  font-size: 0.65rem;
  font-family: var(--mono);
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 0.35rem;
}
.m-value {
  font-family: var(--mono);
  font-size: 1.45rem;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1;
}
.m-delta {
  font-family: var(--mono);
  font-size: 0.72rem;
  margin-top: 0.3rem;
}
.delta-pos { color: var(--green); }
.delta-neg { color: var(--red); }
.delta-neu { color: var(--text-secondary); }

/* ── Tab Navigation ── */
.tab-nav {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
}
.tab-btn {
  padding: 0.55rem 1.2rem;
  font-size: 0.78rem;
  font-family: var(--mono);
  font-weight: 500;
  letter-spacing: 0.5px;
  cursor: pointer;
  border: none;
  background: transparent;
  color: var(--text-dim);
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  text-transform: uppercase;
}
.tab-btn:hover  { color: var(--text-secondary); background: var(--bg-hover); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); background: var(--accent-dim); }

/* ── Table Styling ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px !important; }
[data-testid="stDataFrame"] > div { background: var(--bg-card) !important; }
thead th { font-family: var(--mono) !important; font-size: 0.7rem !important;
           text-transform: uppercase !important; letter-spacing: 1px !important;
           color: var(--text-dim) !important; background: var(--bg-surface) !important; }
tbody td { font-family: var(--mono) !important; font-size: 0.82rem !important; }

/* ── Holdings Row ── */
.holding-row {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.65rem 1rem;
  margin-bottom: 0.4rem;
  display: grid;
  grid-template-columns: 80px 1fr 1fr 1fr 1fr 1fr 1fr;
  align-items: center;
  font-family: var(--mono);
  font-size: 0.8rem;
  transition: all 0.15s;
}
.holding-row:hover { border-color: var(--border-active); background: var(--bg-hover); }
.holding-ticker { font-weight: 600; font-size: 0.9rem; color: var(--accent); }
.holding-header {
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-dim);
  padding: 0.3rem 1rem;
  display: grid;
  grid-template-columns: 80px 1fr 1fr 1fr 1fr 1fr 1fr;
}

/* ── Trade Panel ── */
.trade-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.2rem;
}
.trade-panel h4 {
  font-family: var(--mono);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-dim);
  margin: 0 0 1rem 0;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}

/* ── News Card ── */
.news-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.6rem;
  transition: all 0.2s;
}
.news-card:hover { border-color: var(--border-active); border-left-color: var(--accent); background: var(--bg-hover); }
.news-headline { font-size: 0.88rem; font-weight: 500; color: var(--text-primary); margin-bottom: 0.3rem; line-height: 1.4; }
.news-meta { font-family: var(--mono); font-size: 0.65rem; color: var(--text-dim); display: flex; gap: 1rem; }
.news-meta .source { color: var(--accent); }
.news-sentiment-pos { color: var(--green); font-weight: 600; }
.news-sentiment-neg { color: var(--red); font-weight: 600; }
.news-sentiment-neu { color: var(--text-dim); }

/* ── Trade Log ── */
.trade-log-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.55rem 0.9rem;
  margin-bottom: 0.3rem;
  font-family: var(--mono);
  font-size: 0.78rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.trade-buy  { border-left: 3px solid var(--green); }
.trade-sell { border-left: 3px solid var(--red); }
.badge-buy  { background: var(--green-dim); color: var(--green); padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.65rem; font-weight: 600; }
.badge-sell { background: var(--red-dim);   color: var(--red);   padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.65rem; font-weight: 600; }

/* ── Streamlit Widget Overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  font-family: var(--mono) !important;
  border-radius: 4px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent-dim) !important;
}
label[data-testid="stWidgetLabel"] > div > p {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  color: var(--text-dim) !important;
}

/* ── Buttons (general) ── */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  transition: all 0.15s !important;
}
/* Tab buttons — inactive */
.stButton > button[kind="secondary"] {
  background: var(--bg-surface) !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--border) !important;
  border-bottom: 2px solid transparent !important;
}
.stButton > button[kind="secondary"]:hover {
  color: var(--text-secondary) !important;
  background: var(--bg-hover) !important;
  border-color: var(--border-active) !important;
}
/* Tab buttons — active */
.stButton > button[kind="primary"] {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  box-shadow: 0 0 10px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
}

/* Buy button */
.buy-btn > div > button { background: var(--green) !important; color: #000 !important; border: none !important; }
.buy-btn > div > button:hover { background: #33ff88 !important; box-shadow: 0 0 12px rgba(0,230,118,0.4) !important; }
.sell-btn > div > button { background: var(--red) !important; color: #fff !important; border: none !important; }
.sell-btn > div > button:hover { background: #ff6666 !important; box-shadow: 0 0 12px rgba(255,68,68,0.4) !important; }

/* ── Plotly overrides ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .plot-container {
  background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-active); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Alert / Info boxes ── */
.stAlert { background: var(--bg-card) !important; border-radius: 4px !important; }
[data-testid="stNotification"] { background: var(--bg-card) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 5px !important;
}
[data-testid="stExpanderToggleIcon"] { color: var(--accent) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Metric widget (native) ── */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.7rem 0.9rem;
}
[data-testid="stMetricLabel"] p { font-family: var(--mono) !important; font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 1.3rem !important; color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.75rem !important; }

/* ── P&L positive/negative ── */
.pnl-pos { color: var(--green) !important; }
.pnl-neg { color: var(--red) !important; }

/* ── Section Headers ── */
.section-header {
  font-family: var(--mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-dim);
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.8rem;
}

/* ── Watchlist ── */
.wl-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.9rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 0.4rem;
  font-family: var(--mono);
  font-size: 0.82rem;
  transition: all 0.15s;
}
.wl-row:hover { border-color: var(--border-active); }
.wl-ticker { font-weight: 600; font-size: 0.92rem; color: var(--text-primary); min-width: 70px; }
.wl-price  { color: var(--text-secondary); }

/* ── Tooltip / info ── */
.tooltip-text { font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim); }

/* ── Footer ── */
.apex-footer {
  border-top: 1px solid var(--border);
  padding: 0.6rem 0;
  font-family: var(--mono);
  font-size: 0.62rem;
  color: var(--text-dim);
  display: flex;
  justify-content: space-between;
  margin-top: 2rem;
}

/* ── Action buttons (non-tab secondary buttons that need accent colour) ── */
/* Target by key suffix patterns via Streamlit data-testid not available,
   so we style all secondary buttons that sit inside known wrapper divs.
   Generic secondary buttons in forms look like bordered accent buttons. */
[data-testid="stForm"] .stButton > button[kind="secondary"],
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:not([data-tab-btn]) {
  border-color: var(--border-active) !important;
  color: var(--text-secondary) !important;
}
[data-testid="stForm"] .stButton > button[kind="secondary"]:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY DARK TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1218",
    font=dict(family="IBM Plex Mono, monospace", color="#8899bb", size=11),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", tickcolor="#4a5568", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", tickcolor="#4a5568", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#141820", font_color="#e8edf5", font_size=12,
                    font_family="IBM Plex Mono", bordercolor="#2e3d5a"),
    margin=dict(l=0, r=0, t=36, b=0),
)

# ══════════════════════════════════════════════════════════════════════════════
#  TIMEZONE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def now_ist() -> datetime:
    return datetime.now(IST)

def now_est() -> datetime:
    return datetime.now(US_EASTERN)

def fmt_ist(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S IST  %d %b %Y")

def get_market_status() -> Tuple[str, str]:
    """Return (status, label) — OPEN/PRE/AFTER/CLOSED + human string."""
    est = now_est()
    if est.weekday() >= 5:
        return "CLOSED", "WEEKEND"
    t = est.time()
    from datetime import time as dtime
    if dtime(4,  0) <= t < dtime(9, 30):  return "PRE",    "PRE-MARKET"
    if dtime(9, 30) <= t < dtime(16,  0): return "OPEN",   "MARKET OPEN"
    if dtime(16, 0) <= t < dtime(20,  0): return "AFTER",  "AFTER-HOURS"
    return "CLOSED", "CLOSED"

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults: Dict[str, Any] = {
        "portfolio":          [h.copy() for h in INITIAL_PORTFOLIO],
        "watchlist":          INITIAL_WATCHLIST.copy(),
        "cash":               INITIAL_CASH,
        "trade_log":          [],           # list[dict] — every executed trade
        "use_mock":           False,
        "active_tab":         "PORTFOLIO",
        "last_refresh":       now_ist(),
        "mc_results":         None,         # (days_arr, sim_matrix, base_value)
        "mc_params":          {"days": 252, "sims": 1000},
        "target_alloc":       {
            "Technology": 35, "Healthcare": 15, "Financials": 15,
            "Consumer Discretionary": 15, "Communication Services": 10, "Cash": 10,
        },
        "news_filter_ticker": "ALL",
        "selected_ticker":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — fetch_quote / fetch_history / fetch_news
# ══════════════════════════════════════════════════════════════════════════════

def _mock_quote(ticker: str) -> Dict:
    rng = np.random.default_rng(abs(hash(ticker)) % 2**31)
    base = rng.uniform(40, 600)
    chg  = rng.normal(0, base * 0.015)
    return dict(
        ticker=ticker, price=base + chg, prev_close=base,
        change=chg, change_pct=chg / base * 100,
        volume=int(rng.integers(500_000, 40_000_000)),
        market_cap=int(base * rng.integers(100_000_000, 8_000_000_000)),
        pe=float(rng.uniform(10, 50)), pb=float(rng.uniform(1, 10)),
        eps=float(rng.uniform(0.5, 15)), div_yield=float(rng.uniform(0, 3)),
        week52_high=base * rng.uniform(1.05, 1.45),
        week52_low=base * rng.uniform(0.55, 0.95),
        sector=rng.choice(SECTORS),
        name=f"{ticker} Inc.",
        beta=float(rng.uniform(0.4, 2.2)),
    )

@st.cache_data(ttl=CACHE_TTL_QUOTE, show_spinner=False)
def fetch_quote(ticker: str, mock: bool = False) -> Dict:
    if mock:
        return _mock_quote(ticker)
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="2d", interval="1d")

        if hist.empty:
            return _mock_quote(ticker)

        price      = float(info.get("currentPrice") or hist["Close"].iloc[-1])
        prev_close = float(info.get("previousClose") or
                          (hist["Close"].iloc[-2] if len(hist) >= 2 else price))

        chg     = price - prev_close
        chg_pct = (chg / prev_close * 100) if prev_close else 0.0

        return dict(
            ticker=ticker,
            price=price,
            prev_close=prev_close,
            change=chg,
            change_pct=chg_pct,
            volume=int(info.get("volume") or hist["Volume"].iloc[-1] or 0),
            market_cap=int(info.get("marketCap") or 0),
            pe=float(info.get("trailingPE") or 0),
            pb=float(info.get("priceToBook") or 0),
            eps=float(info.get("trailingEps") or 0),
            div_yield=float(info.get("dividendYield") or 0) * 100,
            week52_high=float(info.get("fiftyTwoWeekHigh") or price),
            week52_low=float(info.get("fiftyTwoWeekLow")  or price),
            sector=str(info.get("sector") or "Unknown"),
            name=str(info.get("longName") or ticker),
            beta=float(info.get("beta") or 1.0),
        )
    except Exception as exc:
        log.warning("fetch_quote(%s): %s — using mock", ticker, exc)
        return _mock_quote(ticker)


def _mock_history(ticker: str, days: int = 180) -> pd.DataFrame:
    rng   = np.random.default_rng(abs(hash(ticker)) % 2**31)
    base  = rng.uniform(40, 600)
    ret   = rng.normal(0.0005, 0.018, days)
    close = base * np.cumprod(1 + ret)
    high  = close * (1 + abs(rng.normal(0, 0.008, days)))
    low   = close * (1 - abs(rng.normal(0, 0.008, days)))
    open_ = close * (1 + rng.normal(0, 0.006, days))
    vol   = rng.integers(500_000, 30_000_000, days).astype(float)
    idx   = pd.date_range(end=pd.Timestamp.now(tz=IST), periods=days, freq="B", tz=IST)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
                        index=idx[-len(close):])


@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo", mock: bool = False) -> pd.DataFrame:
    if mock:
        return _mock_history(ticker)
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return _mock_history(ticker)
        if hist.index.tzinfo is None:
            hist.index = hist.index.tz_localize(US_EASTERN)
        hist.index = hist.index.tz_convert(IST)
        return hist
    except Exception as exc:
        log.warning("fetch_history(%s): %s — using mock", ticker, exc)
        return _mock_history(ticker)


@st.cache_data(ttl=CACHE_TTL_NEWS, show_spinner=False)
def fetch_news(ticker: str, mock: bool = False, max_items: int = 12) -> List[Dict]:
    """Fetch news from yfinance .news property; fallback to mock headlines."""
    if mock:
        return _mock_news(ticker)
    try:
        news_raw = yf.Ticker(ticker).news or []
        articles = []
        for item in news_raw[:max_items]:
            ct = item.get("content", {})
            title = (ct.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            pub_date = ""
            ts = ct.get("pubDate") or item.get("providerPublishTime")
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        pub_date = datetime.fromtimestamp(ts, tz=IST).strftime("%b %d %H:%M IST")
                    else:
                        pub_date = str(ts)[:16]
                except Exception:
                    pass
            provider = ""
            prov = ct.get("provider") or item.get("publisher") or {}
            if isinstance(prov, dict):
                provider = prov.get("displayName") or prov.get("name") or ""
            elif isinstance(prov, str):
                provider = prov
            url = ct.get("canonicalUrl", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            else:
                url = item.get("link") or ""
            sentiment, score = _sentiment(title)
            articles.append(dict(
                title=title, date=pub_date,
                source=provider or "Yahoo Finance",
                url=url, sentiment=sentiment, score=score,
                ticker=ticker,
            ))
        return articles if articles else _mock_news(ticker)
    except Exception as exc:
        log.warning("fetch_news(%s): %s", ticker, exc)
        return _mock_news(ticker)


def _sentiment(text: str) -> Tuple[str, float]:
    """Simple keyword-based sentiment — O(1), no external deps."""
    pos = ["surge", "rally", "gain", "beat", "record", "buy", "upgrade", "profit",
           "rise", "growth", "strong", "bullish", "high", "positive", "soar"]
    neg = ["fall", "drop", "loss", "miss", "downgrade", "sell", "decline", "weak",
           "bearish", "low", "negative", "crash", "plunge", "cut", "risk", "warn"]
    t   = text.lower()
    p   = sum(1 for w in pos if w in t)
    n   = sum(1 for w in neg if w in t)
    score = (p - n) / max(p + n, 1)
    if score >  0.1: return "positive", score
    if score < -0.1: return "negative", score
    return "neutral", score


def _mock_news(ticker: str) -> List[Dict]:
    items = [
        f"{ticker} reports strong quarterly earnings beating estimates",
        f"Analysts upgrade {ticker} to Buy citing robust demand",
        f"{ticker} announces share buyback program worth $2B",
        f"Institutional investors increase stake in {ticker}",
        f"{ticker} faces headwinds from rising interest rates",
        f"Sector rotation may impact {ticker} short-term outlook",
        f"{ticker} CEO comments on AI investment strategy",
        f"Technical analysis: {ticker} tests key support level",
    ]
    result = []
    rng = np.random.default_rng(abs(hash(ticker + "news")) % 2**31)
    idxs = rng.choice(len(items), size=min(6, len(items)), replace=False)
    for i in idxs:
        title = items[i]
        s, sc = _sentiment(title)
        hours_ago = int(rng.integers(1, 48))
        result.append(dict(
            title=title,
            date=f"{hours_ago}h ago",
            source="Market Wire",
            url="#",
            sentiment=s, score=sc,
            ticker=ticker,
        ))
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_holdings(portfolio: List[Dict], mock: bool) -> List[Dict]:
    """Enrich portfolio positions with live quotes and history."""
    enriched = []
    for pos in portfolio:
        q    = fetch_quote(pos["ticker"], mock)
        hist = fetch_history(pos["ticker"], mock=mock)
        cp   = q["price"]
        cost = pos["avg_cost"]
        qty  = pos["quantity"]
        cv   = cp * qty
        cb   = cost * qty
        pl   = cv - cb
        plp  = pl / cb * 100 if cb else 0.0
        day_pl = q["change"] * qty

        # 5-day P&L via history
        h5d_pl = 0.0
        if len(hist) >= 6:
            p5 = float(hist["Close"].iloc[-6])
            h5d_pl = (cp - p5) * qty

        enriched.append(dict(
            ticker=pos["ticker"],
            name=q["name"],
            quantity=qty,
            avg_cost=cost,
            current_price=cp,
            current_value=cv,
            cost_basis=cb,
            pl=pl, pl_pct=plp,
            day_change=q["change"],
            day_change_pct=q["change_pct"],
            day_pl=day_pl,
            five_day_pl=h5d_pl,
            sector=q["sector"],
            history=hist,
            quote=q,
        ))
    return enriched


def portfolio_metrics(holdings: List[Dict], cash: float) -> Dict:
    tv   = sum(h["current_value"] for h in holdings) + cash
    cb   = sum(h["cost_basis"]    for h in holdings)
    eq   = tv - cash
    pl   = eq - cb
    plp  = pl / cb * 100 if cb else 0.0
    day_pl = sum(h["day_pl"] for h in holdings)
    day_plp = day_pl / (eq - day_pl) * 100 if (eq - day_pl) else 0.0
    five_day_pl = sum(h["five_day_pl"] for h in holdings)

    # Weighted Sharpe
    sharpe = 0.0
    if holdings:
        w  = np.array([h["current_value"] for h in holdings])
        w  = w / w.sum() if w.sum() else w
        rets_list = []
        for i, h in enumerate(holdings):
            if len(h["history"]) > 20:
                r = h["history"]["Close"].pct_change().dropna().values
                rets_list.append((r, w[i]))
        if rets_list:
            port_rets = sum(r * wi for r, wi in rets_list)
            if hasattr(port_rets, "__len__") and len(port_rets) > 1:
                mu  = np.mean(port_rets) * 252
                sig = np.std(port_rets)  * np.sqrt(252)
                sharpe = mu / sig if sig else 0.0

    # Beta vs SPY
    beta = sum(h["quote"]["beta"] * h["current_value"] for h in holdings) / eq if eq else 1.0

    return dict(
        total_value=tv, equity=eq, cash=cash,
        cost_basis=cb, total_pl=pl, total_pl_pct=plp,
        day_pl=day_pl, day_pl_pct=day_plp,
        five_day_pl=five_day_pl,
        sharpe=sharpe, beta=beta,
        n_positions=len(holdings),
    )

# ══════════════════════════════════════════════════════════════════════════════
#  TRADE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def execute_buy(ticker: str, quantity: int, price: float) -> Tuple[bool, str]:
    """Execute a BUY order. Updates portfolio + cash. Returns (ok, message)."""
    cost = price * quantity
    if cost > st.session_state.cash:
        return False, f"Insufficient funds — need ${cost:,.2f}, have ${st.session_state.cash:,.2f}"
    if quantity <= 0:
        return False, "Quantity must be > 0"

    # Debit cash
    st.session_state.cash -= cost

    # Update or add position
    portfolio = st.session_state.portfolio
    for pos in portfolio:
        if pos["ticker"] == ticker:
            old_qty  = pos["quantity"]
            old_cost = pos["avg_cost"]
            new_qty  = old_qty + quantity
            new_avg  = (old_cost * old_qty + price * quantity) / new_qty
            pos["quantity"] = new_qty
            pos["avg_cost"]  = round(new_avg, 4)
            break
    else:
        portfolio.append({"ticker": ticker, "quantity": quantity, "avg_cost": round(price, 4)})

    # Log the trade
    st.session_state.trade_log.insert(0, dict(
        ts=now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        action="BUY",
        ticker=ticker,
        quantity=quantity,
        price=price,
        total=cost,
        cash_after=st.session_state.cash,
    ))

    # Invalidate cache for this ticker
    fetch_quote.clear()
    return True, f"BUY {quantity} × {ticker} @ ${price:.2f} = ${cost:,.2f} — cash remaining ${st.session_state.cash:,.2f}"


def execute_sell(ticker: str, quantity: int, price: float) -> Tuple[bool, str]:
    """Execute a SELL order. Updates portfolio + cash. Returns (ok, message)."""
    if quantity <= 0:
        return False, "Quantity must be > 0"

    portfolio = st.session_state.portfolio
    pos_idx = next((i for i, p in enumerate(portfolio) if p["ticker"] == ticker), None)

    if pos_idx is None:
        return False, f"No position found for {ticker}"
    pos = portfolio[pos_idx]
    if quantity > pos["quantity"]:
        return False, f"Cannot sell {quantity} shares — only hold {pos['quantity']}"

    proceeds = price * quantity
    avg_cost  = pos["avg_cost"]
    realized_pl = (price - avg_cost) * quantity

    # Credit cash
    st.session_state.cash += proceeds

    # Reduce / remove position
    pos["quantity"] -= quantity
    if pos["quantity"] == 0:
        st.session_state.portfolio.pop(pos_idx)

    # Log the trade
    st.session_state.trade_log.insert(0, dict(
        ts=now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        action="SELL",
        ticker=ticker,
        quantity=quantity,
        price=price,
        total=proceeds,
        realized_pl=realized_pl,
        cash_after=st.session_state.cash,
    ))

    sign = "+" if realized_pl >= 0 else ""
    return True, (f"SELL {quantity} × {ticker} @ ${price:.2f} = ${proceeds:,.2f} — "
                  f"Realized P/L: {sign}${realized_pl:,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo(
    tickers:    Tuple[str, ...],
    weights:    Tuple[float, ...],
    base_value: float,
    days:       int,
    sims:       int,
    cache_key:  str,
) -> Tuple[np.ndarray, np.ndarray]:
    """GBM-based Monte Carlo. Returns (days_axis, simulations matrix [sims×days])."""
    mock = st.session_state.get("use_mock", False)
    returns_matrix = []
    for t, w in zip(tickers, weights):
        hist = fetch_history(t, mock=mock)
        if len(hist) < 20:
            continue
        r = hist["Close"].pct_change().dropna().values
        returns_matrix.append(r * w)

    if not returns_matrix:
        return np.array([]), np.array([])

    min_len = min(len(r) for r in returns_matrix)
    port_ret = sum(r[-min_len:] for r in returns_matrix)

    mu  = np.mean(port_ret)
    sig = np.std(port_ret)

    rng = np.random.default_rng(42)
    sim = np.zeros((sims, days))
    for i in range(sims):
        dr  = rng.normal(mu, sig, days)
        sim[i] = base_value * np.cumprod(1 + dr)

    return np.arange(1, days + 1), sim

# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _layout(**kw) -> dict:
    d = PLOT_LAYOUT.copy()
    d.update(kw)
    return d


def chart_portfolio_value(holdings: List[Dict]) -> go.Figure:
    """Reconstruct historical portfolio value from individual histories."""
    if not holdings:
        return go.Figure()

    # Align on common dates (business days)
    all_hist  = [(h["ticker"], h["quantity"], h["history"]) for h in holdings if not h["history"].empty]
    if not all_hist:
        return go.Figure()

    # Use the shortest history as the date index
    min_len = min(len(h) for _, _, h in all_hist)
    ref_idx = all_hist[0][2].index[-min_len:]

    values = np.zeros(min_len)
    for _, qty, hist in all_hist:
        closes = hist["Close"].values[-min_len:]
        values += closes * qty

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ref_idx, y=values,
        mode="lines",
        name="Portfolio",
        line=dict(color="#00c4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,196,255,0.06)",
        hovertemplate="%{x|%b %d %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))

    # Drawdown shading
    roll_max = pd.Series(values).cummax()
    dd = (pd.Series(values) - roll_max) / roll_max * 100
    fig.add_trace(go.Scatter(
        x=ref_idx, y=values * (1 + dd / 100),
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(255,68,68,0.05)",
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(**_layout(
        title=dict(text="PORTFOLIO VALUE", font=dict(size=11, color="#4a5568"), x=0),
        height=280,
    ))
    return fig


def chart_candlestick(ticker: str, hist: pd.DataFrame, period_label: str = "6M") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],  close=hist["Close"],
        name=ticker,
        increasing=dict(line=dict(color="#00e676", width=1), fillcolor="rgba(0,230,118,0.6)"),
        decreasing=dict(line=dict(color="#ff4444", width=1), fillcolor="rgba(255,68,68,0.6)"),
    ), row=1, col=1)

    # 20-day & 50-day MA
    hist = hist.copy()
    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"], name="MA20",
                             line=dict(color="#ffa726", width=1.2, dash="dot"),
                             hovertemplate="%{y:.2f}<extra>MA20</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], name="MA50",
                             line=dict(color="#ab47bc", width=1.2, dash="dash"),
                             hovertemplate="%{y:.2f}<extra>MA50</extra>"), row=1, col=1)

    # Volume bars
    colors = ["rgba(0,230,118,0.5)" if c >= o else "rgba(255,68,68,0.5)"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                         marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(**_layout(
        title=dict(text=f"{ticker} · {period_label}", font=dict(size=11, color="#4a5568"), x=0),
        height=420,
        xaxis_rangeslider_visible=False,
    ))
    fig.update_yaxes(row=2, col=1, title_text="Vol", title_font=dict(size=9))
    return fig


def chart_allocation_donut(holdings: List[Dict]) -> go.Figure:
    labels = [h["ticker"] for h in holdings]
    values = [h["current_value"] for h in holdings]
    colors = ["#00c4ff", "#00e676", "#ffa726", "#ab47bc",
              "#ef5350", "#26c6da", "#ffca28", "#66bb6a"][:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0a0c10", width=2)),
        textinfo="label+percent",
        textfont=dict(family="IBM Plex Mono", size=10, color="#8899bb"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=dict(text="ALLOCATION", font=dict(size=11, color="#4a5568"), x=0),
        height=300, showlegend=False,
        annotations=[dict(text="BY STOCK", x=0.5, y=0.5, showarrow=False,
                          font=dict(family="IBM Plex Mono", size=9, color="#4a5568"))],
    ))
    return fig


def chart_sector_donut(holdings: List[Dict]) -> go.Figure:
    smap: Dict[str, float] = {}
    for h in holdings:
        smap[h["sector"]] = smap.get(h["sector"], 0) + h["current_value"]
    labels = list(smap.keys())
    values = list(smap.values())
    colors = px.colors.qualitative.Pastel[:len(labels)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0a0c10", width=2)),
        textinfo="label+percent",
        textfont=dict(family="IBM Plex Mono", size=10, color="#8899bb"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=dict(text="SECTOR EXPOSURE", font=dict(size=11, color="#4a5568"), x=0),
        height=300, showlegend=False,
        annotations=[dict(text="SECTOR", x=0.5, y=0.5, showarrow=False,
                          font=dict(family="IBM Plex Mono", size=9, color="#4a5568"))],
    ))
    return fig


def chart_risk_return(holdings: List[Dict]) -> go.Figure:
    tickers, rets, vols, vals = [], [], [], []
    for h in holdings:
        if len(h["history"]) < 20:
            continue
        r = h["history"]["Close"].pct_change().dropna()
        ann_ret = (h["current_price"] / h["avg_cost"] - 1) * 100
        ann_vol = r.std() * np.sqrt(252) * 100
        tickers.append(h["ticker"])
        rets.append(ann_ret)
        vols.append(ann_vol)
        vals.append(h["current_value"])

    color = rets if rets else [0]
    fig = go.Figure(go.Scatter(
        x=vols, y=rets,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=10, color="#8899bb"),
        marker=dict(
            size=[max(v / 300, 10) for v in vals],
            color=color,
            colorscale=[[0, "#ff4444"], [0.5, "#ffa726"], [1, "#00e676"]],
            showscale=True,
            colorbar=dict(title=dict(text="Rtn%", font=dict(size=9)), tickfont=dict(size=9)),
            line=dict(color="#0a0c10", width=1),
        ),
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=dict(text="RISK ↔ RETURN", font=dict(size=11, color="#4a5568"), x=0),
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Return (%)",
        height=340,
    ))
    # Efficient frontier quadrant lines
    if vols:
        mx = max(vols) * 1.1
        fig.add_vline(x=np.mean(vols), line=dict(color="#1e2535", dash="dot", width=1))
        fig.add_hline(y=0, line=dict(color="#1e2535", dash="dot", width=1))
    return fig


def chart_rolling_vol(holdings: List[Dict]) -> go.Figure:
    fig = go.Figure()
    colors_list = ["#00c4ff", "#00e676", "#ffa726", "#ab47bc", "#ef5350"]
    for i, h in enumerate(holdings[:5]):
        hist = h["history"]
        if len(hist) < 22:
            continue
        rv = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=hist.index, y=rv,
            name=h["ticker"],
            mode="lines",
            line=dict(color=colors_list[i % len(colors_list)], width=1.5),
            hovertemplate="%{x|%b %d}<br>%{y:.2f}%<extra>" + h["ticker"] + "</extra>",
        ))
    fig.update_layout(**_layout(
        title=dict(text="21-DAY ROLLING VOLATILITY (ANNUALISED)", font=dict(size=11, color="#4a5568"), x=0),
        yaxis_title="Vol %",
        height=280,
        showlegend=True,
    ))
    return fig


def chart_pnl_waterfall(holdings: List[Dict]) -> go.Figure:
    tickers = [h["ticker"] for h in holdings]
    pls     = [h["pl"] for h in holdings]
    colors  = ["rgba(0,230,118,0.7)" if p >= 0 else "rgba(255,68,68,0.7)" for p in pls]

    fig = go.Figure(go.Bar(
        x=tickers, y=pls,
        marker_color=colors,
        text=[f"${p:+,.0f}" for p in pls],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10),
        hovertemplate="<b>%{x}</b><br>P/L: $%{y:+,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#2e3d5a", width=1))
    fig.update_layout(**_layout(
        title=dict(text="UNREALISED P/L BY POSITION", font=dict(size=11, color="#4a5568"), x=0),
        yaxis_title="P/L ($)",
        height=280,
    ))
    return fig


def chart_monte_carlo(days_ax: np.ndarray, sims: np.ndarray) -> go.Figure:
    if not len(sims):
        return go.Figure()
    pct = {5: "#ff4444", 25: "#ffa726", 50: "#00c4ff", 75: "#00e676", 95: "#33ff88"}
    fig = go.Figure()

    # Fan fill between 5 and 95
    p5  = np.percentile(sims, 5,  axis=0)
    p95 = np.percentile(sims, 95, axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([days_ax, days_ax[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(0,196,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    for p, color in pct.items():
        fig.add_trace(go.Scatter(
            x=days_ax,
            y=np.percentile(sims, p, axis=0),
            name=f"P{p}",
            mode="lines",
            line=dict(color=color, width=1.5 if p != 50 else 2,
                      dash="solid" if p == 50 else "dot"),
            hovertemplate=f"Day %{{x}}<br>P{p}: $%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(**_layout(
        title=dict(text="MONTE CARLO FORECAST", font=dict(size=11, color="#4a5568"), x=0),
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        height=380,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, orientation="h"),
    ))
    return fig


def chart_rebalance(current: Dict[str, float], target: Dict[str, float]) -> go.Figure:
    cats = sorted(set(list(current.keys()) + list(target.keys())))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=[current.get(c, 0) for c in cats],
        name="Current",
        marker_color="rgba(0,196,255,0.6)",
        text=[f"{current.get(c, 0):.1f}%" for c in cats],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9),
    ))
    fig.add_trace(go.Bar(
        x=cats, y=[target.get(c, 0) for c in cats],
        name="Target",
        marker_color="rgba(0,230,118,0.6)",
        text=[f"{target.get(c, 0):.1f}%" for c in cats],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9),
    ))
    fig.update_layout(**_layout(
        title=dict(text="CURRENT vs TARGET ALLOCATION", font=dict(size=11, color="#4a5568"), x=0),
        barmode="group",
        yaxis_title="Weight (%)",
        height=320,
        showlegend=True,
    ))
    return fig


def sparkline(hist: pd.DataFrame, h: int = 50) -> go.Figure:
    prices = hist["Close"].values[-30:]
    up     = prices[-1] >= prices[0]
    color  = "#00e676" if up else "#ff4444"
    fill   = "rgba(0,230,118,0.10)" if up else "rgba(255,68,68,0.10)"
    fig = go.Figure(go.Scatter(
        x=list(range(len(prices))), y=prices,
        mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=fill,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=h, margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pnl_html(value: float, percent: Optional[float] = None, mono: bool = True) -> str:
    cls   = "pnl-pos" if value >= 0 else "pnl-neg"
    sign  = "+" if value >= 0 else ""
    font  = "font-family:var(--mono);" if mono else ""
    pct_s = f" ({sign}{percent:.2f}%)" if percent is not None else ""
    return f'<span class="{cls}" style="{font}">{sign}${value:,.2f}{pct_s}</span>'


def render_header(mkt_status: str, mkt_label: str):
    pill = "pill-green" if mkt_status == "OPEN" else ("pill-amber" if mkt_status in ("PRE", "AFTER") else "pill-red")
    st.markdown(f"""
<div class="apex-header">
  <div>
    <div class="apex-logo">APEX <span>TRADING TERMINAL</span></div>
  </div>
  <div class="apex-status-bar">
    <span class="status-pill {pill}">{mkt_label}</span>
    <span>{fmt_ist(now_ist())}</span>
    <span style="color:var(--text-dim)">·</span>
    <span>{'LIVE DATA' if not st.session_state.use_mock else '⚡ MOCK MODE'}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_metric_strip(m: Dict):
    def _card(label: str, value: str, delta: Optional[str] = None, delta_cls: str = "delta-neu") -> str:
        d = f'<div class="m-delta {delta_cls}">{delta}</div>' if delta else ""
        return (f'<div class="m-card">'
                f'<div class="m-label">{label}</div>'
                f'<div class="m-value">{value}</div>'
                f'{d}</div>')

    d_cls = "delta-pos" if m["day_pl"] >= 0 else "delta-neg"
    p_cls = "delta-pos" if m["total_pl"] >= 0 else "delta-neg"
    s_cls = "delta-pos" if m["sharpe"] >= 1 else ("delta-neu" if m["sharpe"] >= 0 else "delta-neg")
    sign  = lambda v: "+" if v >= 0 else ""

    st.markdown(f"""
<div class="metric-strip">
  {_card("Total Value",     f"${m['total_value']:,.2f}")}
  {_card("Unrealised P/L",  f"${m['total_pl']:+,.2f}",
          f"{sign(m['total_pl'])}{m['total_pl_pct']:.2f}%", p_cls)}
  {_card("Day P/L",         f"${m['day_pl']:+,.2f}",
          f"{sign(m['day_pl'])}{m['day_pl_pct']:.2f}%", d_cls)}
  {_card("Cash Balance",    f"${m['cash']:,.2f}")}
  {_card("Sharpe Ratio",    f"{m['sharpe']:.3f}",
          f"β {m['beta']:.2f}", s_cls)}
</div>
""", unsafe_allow_html=True)


def tab_nav(tabs: List[str]) -> str:
    """Render tab bar using real Streamlit buttons so clicks trigger reruns."""
    active = st.session_state.active_tab

    # Inject CSS for the tab strip container (visual styling only)
    st.markdown('<div class="tab-nav-spacer"></div>', unsafe_allow_html=True)

    cols = st.columns(len(tabs))
    for col, t in zip(cols, tabs):
        is_active = (t == active)
        # Active tab: primary style (accent colour); inactive: plain secondary
        label = f"▸ {t}" if is_active else t
        if col.button(label, key=f"_tab_{t}", use_container_width=True,
                      type="primary" if is_active else "secondary"):
            st.session_state.active_tab = t
            st.rerun()

    # Thin separator line beneath tabs
    st.markdown(
        '<hr style="margin:0.3rem 0 1rem 0;border-color:var(--border);">',
        unsafe_allow_html=True,
    )
    return st.session_state.active_tab

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def render_portfolio(holdings: List[Dict], metrics: Dict):
    st.markdown('<div class="section-header">HOLDINGS OVERVIEW</div>', unsafe_allow_html=True)

    # Holdings table
    st.markdown('<div class="holding-header">'
                '<span>TICKER</span><span>QTY</span><span>AVG COST</span>'
                '<span>LAST PRICE</span><span>MARKET VALUE</span>'
                '<span>P / L</span><span>DAY Δ</span>'
                '</div>', unsafe_allow_html=True)

    for h in holdings:
        pl_cls  = "delta-pos" if h["pl"]         >= 0 else "delta-neg"
        day_cls = "delta-pos" if h["day_change"]  >= 0 else "delta-neg"
        sign    = lambda v: "+" if v >= 0 else ""
        st.markdown(f"""
<div class="holding-row">
  <span class="holding-ticker">{h["ticker"]}</span>
  <span>{h["quantity"]}</span>
  <span>${h["avg_cost"]:.2f}</span>
  <span>${h["current_price"]:.2f}</span>
  <span>${h["current_value"]:,.2f}</span>
  <span class="{pl_cls}">{sign(h["pl"])}${h["pl"]:,.2f} ({sign(h["pl_pct"])}{h["pl_pct"]:.2f}%)</span>
  <span class="{day_cls}">{sign(h["day_change"])}{h["day_change_pct"]:.2f}%</span>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(chart_portfolio_value(holdings), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(chart_pnl_waterfall(holdings), use_container_width=True, config={"displayModeBar": False})

    # Charts row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_allocation_donut(holdings), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(chart_sector_donut(holdings), use_container_width=True, config={"displayModeBar": False})

    # Individual stock expanders
    st.markdown('<div class="section-header">INDIVIDUAL CHARTS</div>', unsafe_allow_html=True)
    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y"}
    for h in holdings:
        with st.expander(f"▸  {h['ticker']}  ·  ${h['current_price']:.2f}  ·  "
                         f"{'▲' if h['day_change'] >= 0 else '▼'} {abs(h['day_change_pct']):.2f}%"):
            p_label = st.radio("Period", list(period_map.keys()), index=1,
                               horizontal=True, key=f"period_{h['ticker']}")
            hist = fetch_history(h["ticker"], period=period_map[p_label],
                                 mock=st.session_state.use_mock)
            st.plotly_chart(chart_candlestick(h["ticker"], hist, p_label),
                            use_container_width=True, config={"displayModeBar": False})
            # Fundamentals mini-table
            q = h["quote"]
            cols = st.columns(4)
            cols[0].metric("P/E",          f"{q['pe']:.1f}"         if q["pe"]  else "—")
            cols[1].metric("P/B",          f"{q['pb']:.2f}"         if q["pb"]  else "—")
            cols[2].metric("52W High",     f"${q['week52_high']:.2f}")
            cols[3].metric("Div Yield",    f"{q['div_yield']:.2f}%" if q["div_yield"] else "—")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: TRADE
# ══════════════════════════════════════════════════════════════════════════════

def render_trade(holdings: List[Dict]):
    col_form, col_log = st.columns([1, 1.6])

    with col_form:
        st.markdown('<div class="section-header">ORDER TICKET</div>', unsafe_allow_html=True)

        ticker_input = st.text_input("TICKER SYMBOL", value="AAPL",
                                     placeholder="e.g. AAPL", key="trade_ticker").upper().strip()
        action  = st.radio("ORDER TYPE", ["BUY", "SELL"], horizontal=True, key="trade_action")
        qty     = st.number_input("QUANTITY (SHARES)", min_value=1, value=1, step=1, key="trade_qty")

        # Live quote preview
        if ticker_input:
            try:
                q = fetch_quote(ticker_input, st.session_state.use_mock)
                price = q["price"]
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:0.78rem;color:var(--text-secondary);'
                    f'padding:0.5rem 0;">'
                    f'Last price: <b style="color:var(--text-primary)">${price:.2f}</b>'
                    f'  {("▲" if q["change"]>=0 else "▼")}'
                    f'  <span style="color:{"var(--green)" if q["change"]>=0 else "var(--red)"}">'
                    f'  {q["change_pct"]:+.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
                order_value = price * qty
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:0.78rem;color:var(--text-secondary);">'
                    f'Order value: <b style="color:var(--text-primary)">${order_value:,.2f}</b>'
                    f'  ·  Cash: <b>${st.session_state.cash:,.2f}</b></div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                price = st.number_input("PRICE (MANUAL)", min_value=0.01, value=100.0, key="manual_price")

        st.markdown("<br>", unsafe_allow_html=True)

        # Colour-coded action button — uses CSS class wrapper (buy-btn/sell-btn)
        exec_label = (f"⬆  EXECUTE BUY — {qty} × {ticker_input}" if action == "BUY"
                      else f"⬇  EXECUTE SELL — {qty} × {ticker_input}")
        wrapper_cls = "buy-btn" if action == "BUY" else "sell-btn"
        st.markdown(f'<div class="{wrapper_cls}">', unsafe_allow_html=True)
        clicked = st.button(exec_label, key="exec_btn", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if clicked and ticker_input:
            try:
                exec_price = fetch_quote(ticker_input, st.session_state.use_mock)["price"]
            except Exception:
                exec_price = 100.0
            fn = execute_buy if action == "BUY" else execute_sell
            ok, msg = fn(ticker_input, qty, exec_price)
            if ok:
                st.success(msg)
                fetch_quote.clear()
                st.rerun()
            else:
                st.error(msg)

        # Cash management
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">CASH MANAGEMENT</div>', unsafe_allow_html=True)
        new_cash = st.number_input("SET CASH BALANCE ($)", min_value=0.0,
                                   value=float(st.session_state.cash),
                                   step=1000.0, key="cash_setter")
        if st.button("UPDATE CASH", key="update_cash"):
            st.session_state.cash = new_cash
            st.success(f"Cash updated to ${new_cash:,.2f}")

        # Remove position
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">REMOVE POSITION</div>', unsafe_allow_html=True)
        if st.session_state.portfolio:
            rm_ticker = st.selectbox("SELECT POSITION TO REMOVE",
                                     [p["ticker"] for p in st.session_state.portfolio],
                                     key="rm_pos_select")
            if st.button("REMOVE POSITION", key="rm_pos_btn", type="secondary"):
                st.session_state.portfolio = [
                    p for p in st.session_state.portfolio if p["ticker"] != rm_ticker
                ]
                st.success(f"Removed {rm_ticker} from portfolio")
                st.rerun()

    with col_log:
        st.markdown('<div class="section-header">TRADE LOG</div>', unsafe_allow_html=True)

        log_data = st.session_state.trade_log
        if not log_data:
            st.markdown('<span class="tooltip-text">No trades executed yet.</span>',
                        unsafe_allow_html=True)
        else:
            # Summary metrics
            total_trades = len(log_data)
            buys  = sum(1 for t in log_data if t["action"] == "BUY")
            sells = sum(1 for t in log_data if t["action"] == "SELL")
            realized = sum(t.get("realized_pl", 0) for t in log_data)
            cols = st.columns(4)
            cols[0].metric("TOTAL TRADES", total_trades)
            cols[1].metric("BUYS",  buys)
            cols[2].metric("SELLS", sells)
            cols[3].metric("REALISED P/L", f"${realized:+,.2f}",
                           delta=f"{'▲' if realized>=0 else '▼'}",
                           delta_color="normal")

            st.markdown("<br>", unsafe_allow_html=True)

            for trade in log_data:
                cls  = "trade-buy" if trade["action"] == "BUY" else "trade-sell"
                badge_cls = "badge-buy" if trade["action"] == "BUY" else "badge-sell"
                rpl  = trade.get("realized_pl")
                rpl_s = (f' · R-P/L <b style="color:{"var(--green)" if rpl>=0 else "var(--red)"}">'
                         f'${rpl:+,.2f}</b>') if rpl is not None else ""
                st.markdown(f"""
<div class="trade-log-item {cls}">
  <span>
    <span class="{badge_cls}">{trade["action"]}</span>
    &nbsp;<b>{trade["ticker"]}</b>
    &nbsp;{trade["quantity"]} × ${trade["price"]:.2f}
    &nbsp;=&nbsp;${trade["total"]:,.2f}
    {rpl_s}
  </span>
  <span style="color:var(--text-dim);font-size:0.68rem">{trade["ts"]}</span>
</div>""", unsafe_allow_html=True)

            # Download trade log
            df_log = pd.DataFrame(log_data)
            csv    = df_log.to_csv(index=False)
            st.download_button("⬇  EXPORT TRADE LOG (CSV)", data=csv,
                               file_name=f"trade_log_{now_ist().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════

def render_watchlist():
    col_add, col_list = st.columns([1, 2])

    with col_add:
        st.markdown('<div class="section-header">ADD TO WATCHLIST</div>', unsafe_allow_html=True)
        new_ticker = st.text_input("TICKER", key="wl_add_inp", placeholder="e.g. NVDA").upper().strip()
        if st.button("ADD TICKER", key="wl_add_btn", type="secondary"):
            if new_ticker and new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                st.success(f"Added {new_ticker}")
                st.rerun()
            elif new_ticker in st.session_state.watchlist:
                st.warning(f"{new_ticker} already watched")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">REMOVE</div>', unsafe_allow_html=True)
        if st.session_state.watchlist:
            rm = st.selectbox("SELECT TICKER", st.session_state.watchlist, key="wl_rm_sel")
            if st.button("REMOVE", key="wl_rm_btn", type="secondary"):
                st.session_state.watchlist.remove(rm)
                st.rerun()

    with col_list:
        st.markdown('<div class="section-header">WATCHLIST</div>', unsafe_allow_html=True)
        for ticker in st.session_state.watchlist:
            q    = fetch_quote(ticker, st.session_state.use_mock)
            hist = fetch_history(ticker, "1mo", st.session_state.use_mock)
            sign = "▲" if q["change"] >= 0 else "▼"
            chg_color = "var(--green)" if q["change"] >= 0 else "var(--red)"

            with st.container():
                r1, r2, r3 = st.columns([1.5, 1, 1.5])
                with r1:
                    st.markdown(f"""
<div class="wl-row">
  <span class="wl-ticker">{ticker}</span>
  <span class="wl-price">${q["price"]:.2f}</span>
  <span style="color:{chg_color};font-family:var(--mono);font-size:0.78rem">
    {sign} {abs(q["change_pct"]):.2f}%
  </span>
</div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(
                        f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--text-dim);padding-top:0.3rem">'
                        f'Vol: {q["volume"]/1e6:.1f}M<br>P/E: {q["pe"]:.1f}'
                        f'</div>',
                        unsafe_allow_html=True)
                with r3:
                    if not hist.empty:
                        st.plotly_chart(sparkline(hist), use_container_width=True,
                                        config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def render_analytics(holdings: List[Dict], metrics: Dict):
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_risk_return(holdings), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(chart_rolling_vol(holdings), use_container_width=True, config={"displayModeBar": False})

    # Correlation heatmap
    st.markdown('<div class="section-header">RETURN CORRELATION MATRIX</div>', unsafe_allow_html=True)
    ret_df = pd.DataFrame({
        h["ticker"]: h["history"]["Close"].pct_change().dropna()
        for h in holdings if len(h["history"]) > 20
    })
    if ret_df.shape[1] >= 2:
        corr = ret_df.corr()
        z    = corr.values
        annot = [[f"{v:.2f}" for v in row] for row in z]
        heat = go.Figure(go.Heatmap(
            z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
            text=annot, texttemplate="%{text}",
            colorscale=[[0, "#ff4444"], [0.5, "#141820"], [1, "#00e676"]],
            zmid=0, zmin=-1, zmax=1,
            textfont=dict(family="IBM Plex Mono", size=10),
            hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        heat.update_layout(**_layout(height=300))
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})

    # Monte Carlo
    st.markdown('<div class="section-header">MONTE CARLO SIMULATION</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sim_days = st.slider("FORECAST DAYS", 30, 504, st.session_state.mc_params["days"],
                             step=21, key="mc_days")
    with c2:
        sim_sims = st.slider("SIMULATIONS", 200, 5000, st.session_state.mc_params["sims"],
                             step=200, key="mc_sims")
    with c3:
        if st.button("▶  RUN SIMULATION", type="secondary", key="run_mc"):
            with st.spinner("Running Monte Carlo…"):
                tickers = tuple(h["ticker"] for h in holdings)
                total   = sum(h["current_value"] for h in holdings)
                weights = tuple(h["current_value"] / total for h in holdings) if total else tuple(1 / len(holdings) for _ in holdings)
                ck      = f"{sim_days}_{sim_sims}_{now_ist().strftime('%Y%m%d')}_{'_'.join(tickers)}"
                da, sm  = run_monte_carlo(tickers, weights, metrics["equity"], sim_days, sim_sims, ck)
                st.session_state.mc_results  = (da, sm, metrics["equity"])
                st.session_state.mc_params   = {"days": sim_days, "sims": sim_sims}

    if st.session_state.mc_results is not None:
        da, sm, base = st.session_state.mc_results
        st.plotly_chart(chart_monte_carlo(da, sm), use_container_width=True, config={"displayModeBar": False})
        final = sm[:, -1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P5  (Bear)",  f"${np.percentile(final, 5):,.0f}",
                  f"{(np.percentile(final, 5)/base - 1)*100:.1f}%")
        c2.metric("P50 (Base)",  f"${np.percentile(final, 50):,.0f}",
                  f"{(np.percentile(final, 50)/base - 1)*100:.1f}%")
        c3.metric("P95 (Bull)",  f"${np.percentile(final, 95):,.0f}",
                  f"{(np.percentile(final, 95)/base - 1)*100:.1f}%")
        c4.metric("Prob > Base", f"{(final > base).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: NEWS
# ══════════════════════════════════════════════════════════════════════════════

def render_news(holdings: List[Dict]):
    # Ticker filter
    all_tickers = ["ALL"] + [h["ticker"] for h in holdings] + st.session_state.watchlist
    col_f, col_s = st.columns([2, 1])
    with col_f:
        selected = st.selectbox("FILTER BY TICKER", all_tickers, key="news_ticker_filter")
    with col_s:
        sentiment_filter = st.selectbox("SENTIMENT", ["ALL", "POSITIVE", "NEGATIVE", "NEUTRAL"],
                                        key="news_sentiment_filter")

    st.markdown('<div class="section-header">MARKET INTELLIGENCE FEED</div>', unsafe_allow_html=True)

    # Gather news
    tickers_to_fetch = [h["ticker"] for h in holdings] + st.session_state.watchlist[:3]
    if selected != "ALL":
        tickers_to_fetch = [selected]

    all_news: List[Dict] = []
    with st.spinner("Fetching latest news…"):
        for t in tickers_to_fetch[:8]:   # cap to avoid rate limits
            all_news.extend(fetch_news(t, st.session_state.use_mock))

    # Deduplicate by title
    seen = set()
    deduped = []
    for item in all_news:
        key = item["title"][:60]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Sentiment filter
    if sentiment_filter != "ALL":
        deduped = [n for n in deduped if n["sentiment"].upper() == sentiment_filter]

    # Sort: positive/negative first, then neutral
    deduped.sort(key=lambda x: abs(x["score"]), reverse=True)

    if not deduped:
        st.info("No news articles found for the selected filters.")
        return

    # Sentiment summary bar
    pos_n = sum(1 for n in deduped if n["sentiment"] == "positive")
    neg_n = sum(1 for n in deduped if n["sentiment"] == "negative")
    neu_n = sum(1 for n in deduped if n["sentiment"] == "neutral")
    total = len(deduped)

    st.markdown(f"""
<div style="display:flex;gap:1.5rem;font-family:var(--mono);font-size:0.72rem;
     margin-bottom:0.8rem;padding:0.5rem 0;border-bottom:1px solid var(--border);">
  <span class="news-sentiment-pos">▲ BULLISH: {pos_n} ({pos_n/total*100:.0f}%)</span>
  <span class="news-sentiment-neg">▼ BEARISH: {neg_n} ({neg_n/total*100:.0f}%)</span>
  <span class="news-sentiment-neu">— NEUTRAL: {neu_n} ({neu_n/total*100:.0f}%)</span>
</div>
""", unsafe_allow_html=True)

    # Article cards
    for article in deduped[:30]:
        s    = article["sentiment"]
        s_cls = f"news-sentiment-{s}"
        s_lbl = "▲ BULLISH" if s == "positive" else ("▼ BEARISH" if s == "negative" else "— NEUTRAL")
        url   = article.get("url", "#") or "#"
        link  = f'<a href="{url}" target="_blank" style="color:var(--accent);text-decoration:none;">↗ READ</a>' if url != "#" else ""

        st.markdown(f"""
<div class="news-card">
  <div class="news-headline">{article["title"]}</div>
  <div class="news-meta">
    <span class="source">{article["source"]}</span>
    <span>{article["date"]}</span>
    <span class="{s_cls}">{s_lbl}</span>
    <span style="color:var(--accent);font-size:0.68rem;padding:0.1rem 0.4rem;
                 background:var(--accent-dim);border-radius:3px;">{article["ticker"]}</span>
    {link}
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: REBALANCE
# ══════════════════════════════════════════════════════════════════════════════

def render_rebalance(holdings: List[Dict], metrics: Dict):
    total = metrics["equity"] + metrics["cash"]

    # Build current sector allocation
    sector_map: Dict[str, float] = {}
    for h in holdings:
        sector_map[h["sector"]] = sector_map.get(h["sector"], 0.0) + h["current_value"]
    sector_map["Cash"] = metrics["cash"]
    current_alloc = {k: v / total * 100 for k, v in sector_map.items()}

    col_chart, col_target = st.columns([1.5, 1])
    with col_chart:
        st.plotly_chart(chart_rebalance(current_alloc, st.session_state.target_alloc),
                        use_container_width=True, config={"displayModeBar": False})

    with col_target:
        st.markdown('<div class="section-header">SET TARGET WEIGHTS</div>', unsafe_allow_html=True)
        new_target = {}
        for sector in st.session_state.target_alloc:
            new_target[sector] = st.slider(
                sector, 0, 100, int(st.session_state.target_alloc[sector]),
                key=f"t_{sector}"
            )
        total_w = sum(new_target.values())
        if total_w != 100:
            st.warning(f"Weights sum to {total_w}% (should be 100%)")
        if st.button("SAVE TARGETS", type="secondary", key="save_targets"):
            st.session_state.target_alloc = new_target
            st.success("Target allocation saved")

    # Rebalancing table
    st.markdown('<div class="section-header">REBALANCING ACTIONS</div>', unsafe_allow_html=True)
    suggestions = []
    for sector, target_pct in st.session_state.target_alloc.items():
        curr_pct = current_alloc.get(sector, 0.0)
        diff     = target_pct - curr_pct
        if abs(diff) > 1.5:
            action  = "BUY" if diff > 0 else "SELL"
            amount  = abs(diff) * total / 100
            suggestions.append({
                "Sector":     sector,
                "Action":     action,
                "Current %":  f"{curr_pct:.1f}%",
                "Target %":   f"{target_pct:.1f}%",
                "Δ":          f"{diff:+.1f}%",
                "Est. Amount": f"${amount:,.2f}",
            })

    if suggestions:
        df_s = pd.DataFrame(suggestions)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    else:
        st.success("✔  Portfolio is within tolerance of target allocation")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def render_settings():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">DATA SOURCE</div>', unsafe_allow_html=True)
        mock = st.toggle("Use Mock Data (offline mode)", value=st.session_state.use_mock, key="mock_toggle")
        st.session_state.use_mock = mock
        st.markdown(
            '<span class="tooltip-text">Mock mode generates synthetic prices — '
            'useful for demo and offline testing.</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">CACHE CONTROL</div>', unsafe_allow_html=True)
        if st.button("CLEAR ALL CACHES", type="secondary", key="clear_cache"):
            st.cache_data.clear()
            st.success("All data caches cleared")

    with col2:
        st.markdown('<div class="section-header">EXPORT</div>', unsafe_allow_html=True)

        if st.button("EXPORT PORTFOLIO CSV", type="secondary", key="exp_port"):
            rows = [{"ticker": p["ticker"], "quantity": p["quantity"], "avg_cost": p["avg_cost"]}
                    for p in st.session_state.portfolio]
            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("⬇  DOWNLOAD PORTFOLIO", data=csv,
                               file_name=f"portfolio_{now_ist().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

        if st.button("EXPORT TRADE LOG CSV", type="secondary", key="exp_log"):
            csv = pd.DataFrame(st.session_state.trade_log).to_csv(index=False)
            st.download_button("⬇  DOWNLOAD TRADE LOG", data=csv,
                               file_name=f"trade_log_{now_ist().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="color:var(--red)">DANGER ZONE</div>',
                    unsafe_allow_html=True)
        confirm = st.checkbox("Confirm reset — this cannot be undone", key="confirm_reset")
        if st.button("RESET ALL DATA", type="secondary", key="reset_all"):
            if confirm:
                for k in ["portfolio", "watchlist", "cash", "trade_log", "mc_results"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.success("All session data reset")
                st.rerun()
            else:
                st.error("Check the confirmation box first")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _init_state()
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    mkt_status, mkt_label = get_market_status()
    render_header(mkt_status, mkt_label)

    # Sidebar: quick actions + refresh
    with st.sidebar:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.7rem;'
            'text-transform:uppercase;letter-spacing:2px;color:#4a5568;'
            'margin-bottom:1rem;">Quick Actions</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔄  Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.session_state.last_refresh = now_ist()
            st.rerun()

        st.markdown("---")
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:0.7rem;color:#4a5568;">'
            f'Positions: {len(st.session_state.portfolio)}<br>'
            f'Watchlist: {len(st.session_state.watchlist)}<br>'
            f'Trades: {len(st.session_state.trade_log)}<br>'
            f'Mode: {"MOCK" if st.session_state.use_mock else "LIVE"}<br>'
            f'Last refresh: {st.session_state.last_refresh.strftime("%H:%M:%S")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Load data
    with st.spinner("Loading market data…"):
        holdings = build_holdings(st.session_state.portfolio, st.session_state.use_mock)
        metrics  = portfolio_metrics(holdings, st.session_state.cash)

    # Metric strip
    render_metric_strip(metrics)

    # Tab routing
    TABS = ["PORTFOLIO", "TRADE", "WATCHLIST", "ANALYTICS", "NEWS", "REBALANCE", "SETTINGS"]
    active = tab_nav(TABS)

    if active == "PORTFOLIO":
        render_portfolio(holdings, metrics)
    elif active == "TRADE":
        render_trade(holdings)
    elif active == "WATCHLIST":
        render_watchlist()
    elif active == "ANALYTICS":
        render_analytics(holdings, metrics)
    elif active == "NEWS":
        render_news(holdings)
    elif active == "REBALANCE":
        render_rebalance(holdings, metrics)
    elif active == "SETTINGS":
        render_settings()

    # Footer
    st.markdown(
        f'<div class="apex-footer">'
        f'<span>APEX TRADING TERMINAL  v2.0  ·  Data: Yahoo Finance  ·  '
        f'IST timezone</span>'
        f'<span>⚠ FOR INFORMATIONAL USE ONLY — NOT FINANCIAL ADVICE</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
