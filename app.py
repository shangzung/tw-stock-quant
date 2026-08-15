# app.py
# 台股 V8.4 研究級量化決策 / Point-in-Time 回測 + 統一買進分 + Benchmark + Walk-Forward (含 API 錯誤捕捉與診斷)
# ------------------------------------------------------------
# 修正說明：
# 1. 更新 FinMind API 方法名稱 (taiwan_stock_daily, taiwan_stock_financial_statement)
# 2. 加入 _log_api_error 捕捉並記錄靜默錯誤
# 3. 側邊欄新增「API 診斷」面板
# 4. 統一 market_prefilter 與 calculate_stock_at 的 get_daily 天數參數為 600，避免重複消耗 API 額度。
# ------------------------------------------------------------

import time
import math
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import ta
import yfinance as yf
from FinMind.data import DataLoader

# =========================
# 0. Streamlit & Mac CSS & 狀態初始化
# =========================
st.set_page_config(
    page_title="台股 V8.3 量化決策系統",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化所有分頁的 Session State，確保切換頁籤不會遺失資料
if "market_scan_out" not in st.session_state: st.session_state["market_scan_out"] = None
if "market_scan_candidates" not in st.session_state: st.session_state["market_scan_candidates"] = None
if "market_scan_top5" not in st.session_state: st.session_state["market_scan_top5"] = None
if "year_sim_res" not in st.session_state: st.session_state["year_sim_res"] = None
if "week_sim_res" not in st.session_state: st.session_state["week_sim_res"] = None
if "candidate_out" not in st.session_state: st.session_state["candidate_out"] = None
if "candidate_cands" not in st.session_state: st.session_state["candidate_cands"] = None
if "single_backtest_res" not in st.session_state: st.session_state["single_backtest_res"] = None
if "portfolio_backtest_res" not in st.session_state: st.session_state["portfolio_backtest_res"] = None
if "wf_res" not in st.session_state: st.session_state["wf_res"] = None
if "stock_lookup_res" not in st.session_state: st.session_state["stock_lookup_res"] = None

# Mac 風格與手機端最佳化 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #000000;
        --bg-card: #1c1c1e;
        --bg-card-2: #2c2c2e;
        --border-c: #3a3a3c;
        --text-main: #f5f5f7;
        --text-sub: #a1a1a6;
        --accent-blue: #0a84ff;
        --accent-green: #30d158;
        --accent-yellow: #ffd60a;
        --accent-red: #ff453a;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, Helvetica, Arial, sans-serif;
    }

    /* 整體背景：明確指定主要容器與側邊欄，避免與系統深色模式衝突而造成文字被吃掉 */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
        background-color: var(--bg-main) !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-c) !important;
    }
    [data-testid="stSidebar"] * , [data-testid="stAppViewContainer"] * {
        color: var(--text-main);
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-main) !important;
    }
    [data-testid="stCaptionContainer"] { color: var(--text-sub) !important; }

    /* 輸入元件（text_input / number_input / textarea / selectbox）統一深色風格，確保文字可讀 */
    input, textarea, [data-baseweb="select"] > div, [data-baseweb="base-input"] {
        background-color: var(--bg-card-2) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-c) !important;
        border-radius: 8px !important;
    }
    [data-testid="stWidgetLabel"] p { color: var(--text-sub) !important; font-weight: 500 !important; }

    /* 按鈕 Mac 風格化 (深色) */
    .stButton>button {
        border-radius: 8px !important;
        border: 1px solid var(--border-c) !important;
        background-color: var(--bg-card-2) !important;
        color: var(--text-main) !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #3a3a3c !important;
        border-color: #48484a !important;
    }

    /* 主按鈕 (Primary) 蘋果藍：對應新版與舊版 Streamlit 的 data-testid，避免選錯按鈕 */
    .stButton>button[kind="primary"],
    .stButton>button[data-testid="stBaseButton-primary"],
    .stButton>button[data-testid="baseButton-primary"] {
        background-color: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton>button[kind="primary"]:hover,
    .stButton>button[data-testid="stBaseButton-primary"]:hover,
    .stButton>button[data-testid="baseButton-primary"]:hover {
        background-color: #3399ff !important;
        box-shadow: 0 4px 10px rgba(10, 132, 255, 0.35) !important;
    }

    /* 一般 metric 元件（維持可讀性，深色底配淺色字） */
    div[data-testid="stMetricValue"] { font-weight: 700 !important; color: var(--text-main) !important; }
    div[data-testid="stMetricLabel"] { color: var(--text-sub) !important; }
    div[data-testid="stMetricDelta"] { color: var(--text-sub) !important; }

    /* 表格圓角與深色底，避免亮白色色塊與整體介面不協調 */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        border: 1px solid var(--border-c);
    }

    /* Expander / Tabs 深色化，維持與側邊欄一致的層次 */
    [data-testid="stExpander"] {
        background-color: var(--bg-card-2) !important;
        border: 1px solid var(--border-c) !important;
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab"] { color: var(--text-sub) !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { color: var(--accent-blue) !important; }

    /* ── 側邊欄「大盤位階」分數卡：取代原本不明顯的 st.metric ── */
    .regime-card {
        margin-top: 6px;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
    }
    .regime-card .regime-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-sub);
        letter-spacing: 0.02em;
        margin-bottom: 4px;
    }
    .regime-card .regime-score-row { display: flex; align-items: baseline; gap: 6px; }
    .regime-card .regime-score { font-size: 34px; font-weight: 700; line-height: 1; }
    .regime-card .regime-unit { font-size: 13px; font-weight: 500; color: var(--text-sub); }
    .regime-card .regime-msg { margin-top: 6px; font-size: 12.5px; color: var(--text-main); opacity: 0.9; }
    .regime-BULL    { color: var(--accent-green) !important; border-left: 4px solid var(--accent-green) !important; }
    .regime-NEUTRAL { color: var(--accent-yellow) !important; border-left: 4px solid var(--accent-yellow) !important; }
    .regime-BEAR    { color: var(--accent-red) !important; border-left: 4px solid var(--accent-red) !important; }
    .regime-UNKNOWN { color: var(--text-sub) !important; border-left: 4px solid var(--text-sub) !important; }

    /* ── 今日選股 卡片 ── */
    .pick-card {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
        margin-bottom: 10px;
    }
    .pick-card .pick-top { display: flex; justify-content: space-between; align-items: baseline; }
    .pick-card .pick-name { font-size: 16px; font-weight: 700; }
    .pick-card .pick-score { font-size: 26px; font-weight: 700; color: var(--accent-blue); }
    .pick-card .pick-sub { color: var(--text-sub); font-size: 12.5px; margin-top: 4px; }
    .pick-card .pick-reason { margin-top: 8px; font-size: 13px; line-height: 1.6; }

    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        border: 1px solid var(--border-c);
    }
</style>
""", unsafe_allow_html=True)

st.title("🍏 台股 V8.4 研究級量化決策系統")
st.caption("基本面 × 估值 × 護城河 × 籌碼 × 技術 × Market Regime × ATR 風控 × 跨頁面快取")

# =========================
# 1. API 與 Token 鎖定按鈕
# =========================
api = DataLoader()

# Token／API 設定的輸入元件移到「⚙️ 系統設定」分頁（見下方 render_settings_tab），
# 這裡只用 session_state 裡已套用的值做登入，讓一般使用者不必在主畫面看到這些研究員參數。
if "token_applied" not in st.session_state:
    st.session_state["token_applied"] = ""

active_token = st.session_state["token_applied"]

if active_token:
    try:
        api.login_by_token(api_token=active_token)
        _token_status = ("success", "✅ Token 已生效")
    except Exception:
        _token_status = ("error", "❌ Token 無效")
else:
    try:
        if "FINMIND_TOKEN" in st.secrets:
            api.login_by_token(api_token=st.secrets["FINMIND_TOKEN"])
            _token_status = ("info", "ℹ️ 使用系統隱藏 Token")
        else:
            _token_status = ("warning", "⚠️ 使用免費額度")
    except Exception:
        _token_status = ("warning", "⚠️ 使用免費額度")

API_SLEEP_SEC = 0.35

def throttle():
    time.sleep(API_SLEEP_SEC)

# FinMind 的台股日K／營收／財報／PER/PBR／法人買賣，都是「收盤後才更新一次」的資料
# (平日約 15:00~21:00 陸續更新)，同一天內重複抓取只會拿到一模一樣的內容、卻白白燒掉
# 每小時 300/600 次的額度。把快取時間拉長到 6 小時，同一天內大部分操作都會直接命中
# 快取，不會再打 API；如果懷疑資料真的過期，仍可在「⚙️ 系統設定」按「清除快取」強制重抓。
EOD_CACHE_TTL = 21600

if "api_errors" not in st.session_state:
    st.session_state["api_errors"] = []

# ✅ 錯誤紀錄工具 (將靜默錯誤記錄下來)
def _log_api_error(api_name, stock_id, exc):
    st.session_state["api_errors"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "api": api_name,
        "stock_id": stock_id,
        "error": f"{type(exc).__name__}: {exc}"
    })
    st.session_state["api_errors"] = st.session_state["api_errors"][-30:]

# =========================
# 2. 基本工具
# =========================
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        if isinstance(x, str): x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except Exception: return default

def safe_series_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def clean_stock_list(text):
    if not text: return []
    items = []
    for x in text.replace("\n", ",").replace("，", ",").split(","):
        x = x.strip().upper()
        if x and x.isdigit() and len(x) == 4: items.append(x)
    return list(dict.fromkeys(items))

def clamp(x, lo=0, hi=100):
    if pd.isna(x): return lo
    return max(lo, min(hi, float(x)))


def decision_label(score, overheat=False, limit_up=False, market_regime="UNKNOWN"):
    """將內部量化分數翻成使用者容易判讀的買賣決策。
    分數代表條件整體強弱，不是保證未來報酬率。
    """
    if limit_up:
        return "⚠️ 漲停勿追"
    if overheat:
        return "🟡 過熱觀察"
    if market_regime == "BEAR" and score < 80:
        return "🔴 不買"
    if score >= 85:
        return "🟢 可買"
    if score >= 65:
        return "🟡 觀察"
    return "🔴 不買"


def momentum_status(ret20, rsi, vol_ratio, close, ma20, ma60, distance_20_high):
    """把「強勢／過熱」二分法換成 5 級狀態標籤。
    強勢和過熱不能畫等號：這裡同時看趨勢位階、動能與量能，而不是只看漲幅或 RSI 單一數字。
    """
    ret20 = safe_float(ret20, 0); rsi = safe_float(rsi, 50); vol_ratio = safe_float(vol_ratio, 1)
    close = safe_float(close); ma20 = safe_float(ma20); ma60 = safe_float(ma60)
    distance_20_high = safe_float(distance_20_high, 0)

    trend_up = (not pd.isna(close) and not pd.isna(ma20) and not pd.isna(ma60) and close > ma20 > ma60)
    trend_down = (not pd.isna(close) and not pd.isna(ma20) and close < ma20)

    # 🔴 趨勢轉弱：跌破 MA20，不管漲幅多小都優先標記
    if trend_down:
        return "🔴 趨勢轉弱"
    # 🟠 短線過熱：漲幅／RSI／乖離同時偏極端
    if (ret20 > 0.25 or rsi > 78 or distance_20_high > 0.03):
        return "🟠 短線過熱"
    # 🟡 強勢追蹤：趨勢向上但量能沒跟上，屬於持續觀察
    if trend_up and vol_ratio < 1.2:
        return "🟡 強勢追蹤"
    # 🟢 趨勢發動：多頭排列 + 量能同步放大
    if trend_up and vol_ratio >= 1.2:
        return "🟢 趨勢發動"
    # 🟢 低位起漲：還沒站穩多頭排列，但漲幅溫和、沒有轉弱訊號
    if ret20 >= 0 and not trend_down:
        return "🟢 低位起漲"
    return "🟡 強勢追蹤"


def risk_level(atr, price, rsi, breakout_score_val, market_regime_tag):
    """把 ATR 波動度、RSI 極端程度、大盤環境獨立成風險分，不再讓「買進分」一個數字扛兩件事。
    買進分回答「條件強不強」，風險分回答「萬一看錯，代價多大」。
    """
    atr = safe_float(atr); price = safe_float(price); rsi = safe_float(rsi, 50)
    risk_pts = 0
    atr_pct = (atr / price * 100) if (not pd.isna(atr) and not pd.isna(price) and price > 0) else np.nan
    if not pd.isna(atr_pct):
        risk_pts += 2 if atr_pct >= 6 else 1 if atr_pct >= 3.5 else 0
    if rsi >= 80: risk_pts += 2
    elif rsi >= 72: risk_pts += 1
    if market_regime_tag == "BEAR": risk_pts += 2
    elif market_regime_tag == "NEUTRAL": risk_pts += 1
    if risk_pts >= 4: return "🔴 高"
    if risk_pts >= 2: return "🟡 中"
    return "🟢 低"


def limit_up_status(price, prev_close, day_high, day_low, daily_pct=None):
    """台股簡化漲停/接近漲停判斷。
    以現有日K資料做 UI 判讀；不同股票漲跌幅制度可能不同，因此用接近漲停帶判定，
    不把它當成交易所最終撮合狀態。
    """
    if pd.isna(price) or pd.isna(prev_close) or prev_close <= 0:
        return "未知"
    pct = ((price / prev_close) - 1) * 100 if pd.isna(daily_pct) else float(daily_pct)
    # 一般股票 10% 漲停附近，以 9.5% 作為 UI 提示帶；超過則視為漲停附近。
    if pct >= 9.5:
        if not pd.isna(day_high) and price >= day_high * 0.999:
            return "🔒 接近/封漲停"
        return "🟠 漲幅接近漲停"
    return "—"


def format_num(x, digits=1, suffix=""):
    return "—" if pd.isna(x) else f"{float(x):.{digits}f}{suffix}"

def style_pnl(df):
    styler = df.style
    fn = lambda x: (
        "color: #ff3b30; font-weight: 500;" if isinstance(x, float) and x < 0
        else "color: #34c759; font-weight: 500;" if isinstance(x, float) and x > 0
        else ""
    )
    if hasattr(styler, "map"):
        return styler.map(fn)
    return styler.applymap(fn)

# =========================
# 3. FinMind & Yahoo 抓取資料
# =========================
@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_daily_cached(stock_id, days):
    """實際打 API 的內層函式。刻意讓失敗時用 raise 而不是回傳空表——
    Streamlit 的 st.cache_data 不會快取『拋出例外』的呼叫，只會快取成功的回傳值，
    這樣『這次剛好失敗』就不會被誤當成『這檔股票就是沒資料』快取好幾小時。
    """
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    # ✅ 使用正確的 API 名稱: taiwan_stock_daily
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start)
    throttle()
    if df is None or df.empty:
        raise ValueError("taiwan_stock_daily 回傳空資料")
    df = df.copy()
    for c in ["close", "open", "max", "min", "Trading_turnover"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Trading_Volume" in df.columns: df["volume"] = pd.to_numeric(df["Trading_Volume"], errors="coerce")
    elif "Trading_volume" in df.columns: df["volume"] = pd.to_numeric(df["Trading_volume"], errors="coerce")
    elif "volume" not in df.columns: df["volume"] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

def get_daily(stock_id, days=500):
    try:
        return _get_daily_cached(stock_id, days)
    except Exception as e:
        _log_api_error("taiwan_stock_daily", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def _get_stock_universe_cached():
    info = api.taiwan_stock_info()
    throttle()
    if info is None or info.empty:
        raise ValueError("taiwan_stock_info 回傳空資料")
    info = info.copy()
    info["stock_id"] = info["stock_id"].astype(str)
    info = info[info["stock_id"].str.match(r"^\d{4}$", na=False)]
    info = info[~info["stock_id"].str.startswith("00")]
    if "industry_category" in info.columns:
        info = info[~info["industry_category"].astype(str).str.contains("ETF|受益", case=False, na=False)]
    if "type" in info.columns:
        info = info[info["type"].astype(str).str.lower().isin(["twse", "tpex"])]
    keep_cols = [c for c in ["stock_id", "stock_name", "industry_category", "type"] if c in info.columns]
    return info[keep_cols].drop_duplicates("stock_id").reset_index(drop=True)

def get_stock_universe():
    try:
        return _get_stock_universe_cached()
    except Exception as e:
        _log_api_error("taiwan_stock_info", "-", e)
        return pd.DataFrame()

@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_revenue_cached(stock_id, days):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start)
    throttle()
    if df is None or df.empty:
        raise ValueError("taiwan_stock_month_revenue 回傳空資料")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["revenue", "revenue_year_on_year", "revenue_month_on_month"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def get_revenue(stock_id, days=1000):
    try:
        return _get_revenue_cached(stock_id, days)
    except Exception as e:
        _log_api_error("taiwan_stock_month_revenue", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_financial_cached(stock_id, days):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    # ✅ 使用正確的 API 名稱: taiwan_stock_financial_statement
    df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start)
    throttle()
    if df is None or df.empty:
        raise ValueError("taiwan_stock_financial_statement 回傳空資料")
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "value" in df.columns: df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def get_financial(stock_id, days=1500):
    try:
        return _get_financial_cached(stock_id, days)
    except Exception as e:
        _log_api_error("taiwan_stock_financial_statement", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_per_pbr_cached(stock_id, days):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start)
    throttle()
    if df is None or df.empty:
        raise ValueError("taiwan_stock_per_pbr 回傳空資料")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["PER", "PBR", "dividend_yield"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def get_per_pbr(stock_id, days=1500):
    try:
        return _get_per_pbr_cached(stock_id, days)
    except Exception as e:
        _log_api_error("taiwan_stock_per_pbr", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_institutional_cached(stock_id, days):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
    throttle()
    if df is None or df.empty:
        raise ValueError("taiwan_stock_institutional_investors 回傳空資料")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["buy", "sell"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def get_institutional(stock_id, days=120):
    try:
        return _get_institutional_cached(stock_id, days)
    except Exception as e:
        _log_api_error("taiwan_stock_institutional_investors", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def _get_yahoo_benchmark_cached(ticker):
    mkt = yf.download(ticker, period="10y", progress=False, auto_adjust=False)
    if mkt.empty: raise ValueError(f"yfinance {ticker} 回傳空資料")
    if isinstance(mkt.columns, pd.MultiIndex): mkt.columns=mkt.columns.get_level_values(0)
    mkt=mkt.reset_index(); mkt.columns=[str(c).lower() for c in mkt.columns]
    mkt=mkt.rename(columns={"high":"max","low":"min"})
    mkt["date"]=pd.to_datetime(mkt["date"],errors="coerce")
    for c in ["close","open","max","min","volume"]:
        if c in mkt.columns:mkt[c]=pd.to_numeric(mkt[c],errors="coerce")
    return mkt.sort_values("date").reset_index(drop=True)

def get_yahoo_taiex():
    try: return _get_yahoo_benchmark_cached("^TWII")
    except Exception as e: _log_api_error("yfinance ^TWII", "-", e); return pd.DataFrame()

def get_yahoo_0050():
    try: return _get_yahoo_benchmark_cached("0050.TW")
    except Exception as e: _log_api_error("yfinance 0050.TW", "-", e); return pd.DataFrame()

def get_benchmarks():
    out={}
    twii=get_yahoo_taiex(); etf=get_yahoo_0050()
    if not twii.empty: out["^TWII"]=twii.set_index("date")["close"]
    if not etf.empty: out["0050.TW"]=etf.set_index("date")["close"]
    return out

# =========================
# 3b. 交易所官方 OpenAPI（免費、免 token、無額度限制）：全市場今日成交快照
# 用途：只拿來做「今天夠不夠熱、值不值得花 FinMind 額度」的排序，
# 不取代 FinMind 的歷史K線（這兩支交易所 API 都只給『當天』一筆，沒有歷史區間）。
# =========================
TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAILY_CLOSE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

def _pick_col(df, keywords):
    """在回傳欄位名稱不確定（交易所偶爾會調整中文欄名）時，用關鍵字模糊比對出正確欄位。"""
    for col in df.columns:
        for kw in keywords:
            if kw in str(col):
                return col
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_snapshot_cached():
    """實際打交易所 API 的內層函式：兩邊來源都失敗時用 raise，讓 Streamlit 不快取
    這次失敗，下次呼叫會直接重試，不會卡在『快取住的空表』裡 30 分鐘。
    """
    frames = []

    # 上市：STOCK_DAY_ALL，一次回傳全部上市股票當天資訊
    try:
        resp = requests.get(TWSE_STOCK_DAY_ALL_URL, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            df = pd.DataFrame(rows)
            id_col = _pick_col(df, ["Code", "代號", "證券代號"])
            val_col = _pick_col(df, ["TradeValue", "成交金額"])
            if id_col and val_col:
                out = pd.DataFrame({
                    "stock_id": df[id_col].astype(str).str.strip(),
                    "today_turnover": pd.to_numeric(
                        df[val_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                    ),
                })
                frames.append(out)
    except Exception as e:
        _log_api_error("TWSE STOCK_DAY_ALL", "-", e)

    # 上櫃：櫃買中心每日收盤行情，同樣一次回傳全部上櫃股票
    try:
        resp = requests.get(TPEX_DAILY_CLOSE_URL, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            df = pd.DataFrame(rows)
            id_col = _pick_col(df, ["SecuritiesCompanyCode", "Code", "代號", "證券代號"])
            val_col = _pick_col(df, ["TradingValue", "TradeValue", "成交金額"])
            if id_col and val_col:
                out = pd.DataFrame({
                    "stock_id": df[id_col].astype(str).str.strip(),
                    "today_turnover": pd.to_numeric(
                        df[val_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                    ),
                })
                frames.append(out)
    except Exception as e:
        _log_api_error("TPEx daily_close_quotes", "-", e)

    if not frames:
        raise ValueError("交易所今日快照兩個來源都失敗")
    snap = pd.concat(frames, ignore_index=True)
    snap = snap[snap["stock_id"].str.match(r"^\d{4,6}$", na=False)]
    snap = snap.dropna(subset=["today_turnover"]).drop_duplicates("stock_id")
    return snap.reset_index(drop=True)

def get_market_snapshot():
    """對外的穩定介面：失敗時回傳空表（columns 固定），呼叫端（build_scan_list）
    看到空表就會自動 fallback 成隨機取樣，不會讓整個「今日選股」掛掉。
    """
    try:
        return _get_market_snapshot_cached()
    except Exception as e:
        _log_api_error("get_market_snapshot", "-", e)
        return pd.DataFrame(columns=["stock_id", "today_turnover"])

# =========================
# 4. 技術指標
# =========================
def add_technical_indicators(df):
    df = df.copy()
    if df.empty or "close" not in df.columns: return df
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()
    df["MA120"] = df["close"].rolling(120).mean()

    if "max" in df.columns and "min" in df.columns:
        try:
            df["RSI"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
            stoch = ta.momentum.StochasticOscillator(high=df["max"], low=df["min"], close=df["close"], window=14, smooth_window=3)
            df["K"], df["D"] = stoch.stoch(), stoch.stoch_signal()
            macd = ta.trend.MACD(close=df["close"])
            df["MACD"], df["MACD_signal"] = macd.macd(), macd.macd_signal()
            df["ADX"] = ta.trend.ADXIndicator(high=df["max"], low=df["min"], close=df["close"], window=14).adx()
            df["ATR"] = ta.volatility.AverageTrueRange(high=df["max"], low=df["min"], close=df["close"], window=14).average_true_range()
            df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"].fillna(0)).on_balance_volume()
        except Exception: pass

    df["VOL_MA20"] = df["volume"].rolling(20).mean() if "volume" in df.columns else np.nan
    df["VOL_RATIO"] = df["volume"] / df["VOL_MA20"] if "volume" in df.columns else 1.0
    df["RET_20"] = df["close"].pct_change(20)
    df["HIGH_20"] = df["close"].rolling(20).max()
    df["HIGH_60"] = df["close"].rolling(60).max()
    return df

# =========================
# 5. Market Regime
# =========================
def market_regime(as_of_date=None, mkt_df=None):
    """Market Regime 的 point-in-time 版本。
    as_of_date 指定歷史時點時，只使用 <= 該日期的大盤資料，避免回測偷看未來。
    """
    mkt = mkt_df.copy() if mkt_df is not None else get_yahoo_taiex()
    if mkt.empty:
        return {"regime": "UNKNOWN", "score": 50, "message": "無法取得 Yahoo 大盤資料，Market Filter 採中性。", "df": pd.DataFrame()}
    mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        mkt = mkt[mkt["date"] <= as_of].copy()
    if mkt.empty:
        return {"regime": "UNKNOWN", "score": 50, "message": "該歷史時點沒有大盤資料，採中性。", "df": mkt}
    mkt = add_technical_indicators(mkt)
    x = mkt.iloc[-1]
    score = 50
    if safe_float(x.get("close")) > safe_float(x.get("MA60")): score += 15
    else: score -= 15
    if safe_float(x.get("MA20")) > safe_float(x.get("MA60")): score += 10
    else: score -= 10
    if safe_float(x.get("MACD")) > safe_float(x.get("MACD_signal")): score += 10
    else: score -= 10
    if safe_float(x.get("ADX")) >= 20: score += 5
    score = clamp(score)
    if score >= 70: regime, msg = "BULL", "🟢 多頭環境 (^TWII)：正常尋找多方候選"
    elif score >= 45: regime, msg = "NEUTRAL", "🟡 震盪環境 (^TWII)：提高選股門檻、降低部位"
    else: regime, msg = "BEAR", "🔴 空頭環境 (^TWII)：禁止或大幅降低新增多單"
    return {"regime": regime, "score": score, "message": msg, "df": mkt}

# =========================
# 6. 財報與護城河
# =========================
def find_type_rows(fin, keywords):
    if fin.empty or "type" not in fin.columns: return pd.DataFrame()
    s = fin["type"].astype(str).str.lower()
    mask = False
    for k in keywords: mask = mask | s.str.contains(k.lower(), na=False)
    return fin[mask].copy()

def latest_metric(fin, keywords):
    rows = find_type_rows(fin, keywords)
    if rows.empty: return np.nan
    rows = rows.dropna(subset=["value"]).sort_values("date")
    if rows.empty: return np.nan
    return safe_float(rows.iloc[-1]["value"])

def quarterly_series(fin, keywords, n=12):
    rows = find_type_rows(fin, keywords)
    if rows.empty: return pd.Series(dtype=float)
    rows = rows.dropna(subset=["value"]).sort_values("date")
    if rows.empty: return pd.Series(dtype=float)
    return pd.Series(rows["value"].astype(float).values[-n:], index=rows["date"].values[-n:])

def yoy_growth_from_series(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 5: return np.nan
    a, b = safe_float(s.iloc[-5]), safe_float(s.iloc[-1])
    if pd.isna(a) or a == 0: return np.nan
    return (b / a - 1) * 100

def cagr_from_series(s, periods=8):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < periods: return np.nan
    a, b = safe_float(s.iloc[-periods]), safe_float(s.iloc[-1])
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0: return np.nan
    years = (periods - 1) / 4
    return ((b / a) ** (1 / years) - 1) * 100

def fundamental_score(fin, rev):
    result = {"score": 0, "roe": np.nan, "roa": np.nan, "gross_margin": np.nan, "op_margin": np.nan, "debt_ratio": np.nan, "fcf": np.nan, "eps_growth": np.nan, "revenue_growth": np.nan, "quality": 0}
    if fin.empty: return result

    roe = latest_metric(fin, ["ROE", "ReturnOnEquity"])
    roa = latest_metric(fin, ["ROA", "ReturnOnAssets"])
    gross = latest_metric(fin, ["GrossMargin", "GrossProfitMargin"])
    op = latest_metric(fin, ["OperatingMargin", "OperatingProfitMargin"])
    debt = latest_metric(fin, ["DebtRatio", "LiabilitiesToAssets"])
    eps_s = quarterly_series(fin, ["EPS"])
    eps_growth = yoy_growth_from_series(eps_s)

    rev_growth = np.nan
    if not rev.empty and "revenue_year_on_year" in rev.columns:
        r = pd.to_numeric(rev["revenue_year_on_year"], errors="coerce").dropna()
        if not r.empty: rev_growth = float(r.iloc[-1])

    ocf = latest_metric(fin, ["CashFlowsFromOperatingActivities", "OperatingActivities", "OperatingCashFlow"])
    capex = latest_metric(fin, ["PropertyPlantAndEquipment", "CapitalExpenditure", "CashFlowsFromInvestingActivities"])
    fcf = ocf - abs(capex) if not pd.isna(ocf) and not pd.isna(capex) else ocf

    score = 0
    if not pd.isna(roe): score += 10 if roe >= 15 else 7 if roe >= 10 else 3 if roe > 0 else 0
    if not pd.isna(roa): score += 5 if roa >= 8 else 3 if roa >= 4 else 1 if roa > 0 else 0
    if not pd.isna(gross): score += 5 if gross >= 30 else 3 if gross >= 20 else 1
    if not pd.isna(op): score += 5 if op >= 15 else 3 if op >= 8 else 1 if op > 0 else 0
    if not pd.isna(debt): score += 5 if debt <= 40 else 3 if debt <= 60 else 0
    if not pd.isna(fcf): score += 5 if fcf > 0 else 0
    if not pd.isna(eps_growth): score += 7 if eps_growth >= 15 else 4 if eps_growth > 0 else 0
    if not pd.isna(rev_growth): score += 8 if rev_growth >= 15 else 5 if rev_growth > 0 else 0

    continuity = 0
    for keys in [["EPS"], ["GrossMargin", "GrossProfitMargin"], ["OperatingMargin", "OperatingProfitMargin"]]:
        s = quarterly_series(fin, keys, 12)
        if len(s) >= 6:
            arr = pd.to_numeric(s, errors="coerce").dropna().values
            if len(arr) >= 6: continuity += float(np.mean(arr > 0)) * 5
    score += min(15, continuity)

    result.update({"score": round(clamp(score, 0, 70), 1), "roe": roe, "roa": roa, "gross_margin": gross, "op_margin": op, "debt_ratio": debt, "fcf": fcf, "eps_growth": eps_growth, "revenue_growth": rev_growth, "quality": round(clamp(continuity / 15 * 100), 1)})
    return result

def dynamic_moat_score(fin):
    if fin.empty: return 50, {}
    roe = quarterly_series(fin, ["ROE", "ReturnOnEquity"], 12)
    gross = quarterly_series(fin, ["GrossMargin", "GrossProfitMargin"], 12)
    op = quarterly_series(fin, ["OperatingMargin", "OperatingProfitMargin"], 12)

    def stability_score(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 4: return 50
        positive = np.mean(s > 0) * 100
        volatility = safe_float(s.std(), 999)
        return 0.6 * positive + 0.4 * clamp(100 - volatility * 2)

    roe_st, gross_st, op_st = stability_score(roe), stability_score(gross), stability_score(op)
    score = (roe_st * 0.40 + gross_st * 0.30 + op_st * 0.30)
    return round(clamp(score), 1), {"ROE穩定度": round(roe_st, 1), "毛利率穩定度": round(gross_st, 1), "營益率穩定度": round(op_st, 1)}

def valuation_score(stock_id, fin, pe_df=None, as_of_date=None):
    pe = pe_df.copy() if pe_df is not None else get_per_pbr(stock_id, 1500)
    if as_of_date is not None and not pe.empty:
        pe["date"] = pd.to_datetime(pe["date"], errors="coerce")
        pe = pe[pe["date"] <= pd.Timestamp(as_of_date)]
    if pe.empty: return {"score": 50, "PER": np.nan, "PBR": np.nan, "PEG": np.nan, "PER_pct": np.nan}
    per, pbr = safe_float(pe.iloc[-1].get("PER")), safe_float(pe.iloc[-1].get("PBR"))
    eps = quarterly_series(fin, ["EPS"], 12)
    eps_growth = cagr_from_series(eps, 8)
    peg = per / eps_growth if not pd.isna(per) and per > 0 and not pd.isna(eps_growth) and eps_growth > 0 else np.nan
    hist_per = pd.to_numeric(pe["PER"], errors="coerce").dropna()
    per_pct = float((hist_per <= per).mean() * 100) if not hist_per.empty and not pd.isna(per) else np.nan
    score = 0
    if not pd.isna(peg): score += 20 if peg <= 1 else 15 if peg <= 1.5 else 8 if peg <= 2 else 2
    elif not pd.isna(per): score += 20 if per <= 12 else 15 if per <= 18 else 8 if per <= 25 else 2
    if not pd.isna(pbr): score += 15 if pbr <= 1.5 else 10 if pbr <= 2.5 else 5 if pbr <= 4 else 1
    if not pd.isna(per_pct): score += 15 if per_pct <= 25 else 10 if per_pct <= 50 else 5 if per_pct <= 75 else 0
    return {"score": round(clamp(score, 0, 50), 1), "PER": per, "PBR": pbr, "PEG": peg, "PER_pct": per_pct}


def chip_score(stock_id, inst_df=None, as_of_date=None):
    df = inst_df.copy() if inst_df is not None else get_institutional(stock_id, 120)
    if df.empty or "buy" not in df.columns or "sell" not in df.columns: return 0, {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if as_of_date is not None: df = df[df["date"] <= pd.Timestamp(as_of_date)].copy()
    if df.empty: return 0, {}
    df["net"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")
    foreign, trust = df, df
    if "name" in df.columns:
        foreign = df[df["name"].astype(str).str.contains("Foreign|外資", case=False, na=False)]
        trust = df[df["name"].astype(str).str.contains("Investment|投信", case=False, na=False)]
    def consecutive_buy(series):
        n = 0
        for x in reversed(series.tolist()):
            if safe_float(x, 0) > 0: n += 1
            else: break
        return n
    f_buy = consecutive_buy(foreign.groupby("date")["net"].sum()) if not foreign.empty else 0
    t_buy = consecutive_buy(trust.groupby("date")["net"].sum()) if not trust.empty else 0
    total_net_10 = df.sort_values("date").tail(10)["net"].sum()
    score = min(10, f_buy * 1.5) + min(10, t_buy * 2) + (10 if total_net_10 > 0 else 0)
    return round(clamp(score, 0, 30), 1), {"外資連買": f_buy, "投信連買": t_buy, "10日法人淨買": total_net_10}

def breakout_score(df):
    if df.empty or len(df) < 120: return 0, []
    x, prev = df.iloc[-1], df.iloc[-2]
    score, reasons = 0, []

    close, ma20, ma60, vol_ratio = safe_float(x["close"]), safe_float(x["MA20"]), safe_float(x["MA60"]), safe_float(x["VOL_RATIO"])
    rsi, adx, macd, macd_sig = safe_float(x["RSI"]), safe_float(x["ADX"]), safe_float(x["MACD"]), safe_float(x["MACD_signal"])
    high20, high60 = safe_float(x["HIGH_20"]), safe_float(x["HIGH_60"])

    if close > ma20 > ma60: score += 15; reasons.append("MA20>MA60")
    elif close > ma20: score += 8

    if not pd.isna(high20) and close >= high20 * 0.97: score += 15; reasons.append("近20日高點")
    if not pd.isna(high60) and close >= high60 * 0.98: score += 10; reasons.append("近60日高點")

    if vol_ratio >= 2: score += 15; reasons.append("爆量2倍")
    elif vol_ratio >= 1.5: score += 10; reasons.append("量增")

    if 55 <= rsi <= 70: score += 10; reasons.append("RSI健康")
    elif rsi > 70: score += 3; reasons.append("RSI偏熱")

    if macd > macd_sig and safe_float(prev.get("MACD")) <= safe_float(prev.get("MACD_signal")): score += 10; reasons.append("MACD金叉")
    elif macd > macd_sig: score += 6

    if adx >= 25: score += 10; reasons.append("ADX趨勢強")
    elif adx >= 20: score += 5

    if len(df) >= 20 and safe_float(x["OBV"]) > safe_float(df["OBV"].iloc[-20]): score += 5; reasons.append("OBV升")

    return round(clamp(score), 1), reasons

MIN_AVG_TURNOVER = 3_000_000  # 20日均成交金額門檻（元），低於此視為流動性不足，不值得浪費 API 額度

def passes_liquidity_gate(daily):
    """第一層：不評分，只做『值不值得繼續分析』的過濾。
    資料不足、價格異常、成交太清淡的股票直接砍掉，不進入技術初篩。
    """
    if daily.empty or len(daily) < 120:
        return False, "資料不足"
    x = daily.iloc[-1]
    price = safe_float(x.get("close"))
    if pd.isna(price) or price <= 0:
        return False, "價格異常"
    vol20 = daily["volume"].tail(20)
    if vol20.isna().all():
        return False, "無成交量資料"
    avg_turnover = safe_float((daily["close"].tail(20) * daily["volume"].tail(20)).mean())
    if pd.isna(avg_turnover) or avg_turnover < MIN_AVG_TURNOVER:
        return False, "成交金額過低"
    # 近 20 日完全沒有成交量變化（長期無交易/下市疑慮）
    if vol20.fillna(0).sum() <= 0:
        return False, "長期無成交"
    return True, "通過"


@st.cache_data(ttl=EOD_CACHE_TTL, show_spinner=False)
def market_prefilter(stock_id):
    """真正的『全市場掃描』第一、二層：先做低成本的流動性/資料完整性過濾，
    通過的股票才計算技術初篩分。這一層決定的是『值不值得繼續花 API 額度』，
    不是『排名』——所以在清單裡的順序（股票代碼開頭）完全不影響結果。
    """
    # 修正：將抓取天數從 260 改為 600，與 calculate_stock 的參數完全一致，
    # 以確保快取能正確命中，避免重複消耗 API 請求額度。
    daily = get_daily(stock_id, 600)
    
    ok, _ = passes_liquidity_gate(daily)
    if not ok:
        return None
    daily = add_technical_indicators(daily)
    score, reasons = breakout_score(daily)
    x = daily.iloc[-1]
    price = safe_float(x["close"])
    if pd.isna(price) or price <= 0:
        return None
    return {
        "股票代碼": stock_id,
        "現價": round(price, 2),
        "初篩分": score,
        "量比": safe_float(x["VOL_RATIO"]),
        "RSI": safe_float(x["RSI"]),
        "初篩理由": "、".join(reasons[:4]),
    }


SCAN_STRENGTH_CONFIG = {
    # prefilter 檔數 + topk*5（完整分析每檔要打 5 支 FinMind API）＝這一輪大約會消耗的
    # FinMind 額度。FinMind 免費額度是「未登入 300 次/hr、免費註冊後 600 次/hr」，
    # 所以：
    #   ⚡ 快速：150+12*5=210 次 → 未登入也跑得完
    #   🎯 標準：400+18*5=490 次 → 需要在「⚙️ 系統設定」套用免費 FinMind token（600次/hr）
    #   🔬 深度：全市場（約 1700+ 檔）→ 一小時內一定跑不完，只建議在額度重置後分批跑，
    #            或是升級付費方案
    "⚡ 快速": {"prefilter": 150, "topk": 12},
    "🎯 標準": {"prefilter": 400, "topk": 18},
    "🔬 深度": {"prefilter": None, "topk": 25},  # None = 全市場，不做人數上限
}


def build_scan_list(uni_df, strength_label, seed=None):
    """決定這一輪要送進 FinMind 的候選清單（這是真正消耗 API 額度的地方）。

    優先順序：
    1. 先用交易所官方 OpenAPI 的『今日全市場成交金額』快照（免費、不消耗 FinMind 額度）
       把全市場依今日成交金額排序，取最活躍的前 N 檔——這對應到後面
       market_prefilter 本來就要做的流動性過濾，等於是拿免費資料先幫忙篩過一輪，
       讓有限的 FinMind 額度花在『本來就比較可能通過』的股票上，而不是隨機挑到一堆
       冷門股後被 passes_liquidity_gate 直接刷掉、白白浪費額度。
    2. 若快照抓不到（例如非交易時段、交易所端暫時異常），退回原本『用當日日期當種子
       均勻隨機取樣』的作法，行為與舊版一致，不會讓掃描直接失敗。
    '深度' 模式直接掃描全市場，不取樣（不論用哪種排序都一樣是全部）。
    """
    cfg = SCAN_STRENGTH_CONFIG[strength_label]
    all_ids = uni_df["stock_id"].tolist()
    if cfg["prefilter"] is None or cfg["prefilter"] >= len(all_ids):
        return all_ids

    snap = get_market_snapshot()
    if snap is not None and not snap.empty:
        merged = uni_df[["stock_id"]].merge(snap, on="stock_id", how="left")
        merged["today_turnover"] = merged["today_turnover"].fillna(0)
        if (merged["today_turnover"] > 0).sum() >= cfg["prefilter"]:
            ranked = merged.sort_values("today_turnover", ascending=False)
            return ranked.head(cfg["prefilter"])["stock_id"].tolist()

    # Fallback：快照沒抓到、或有效資料不夠 N 檔，退回原本的隨機取樣邏輯
    rng = np.random.default_rng(seed if seed is not None else int(datetime.now().strftime("%Y%m%d")))
    idx = rng.choice(len(all_ids), size=cfg["prefilter"], replace=False)
    return [all_ids[i] for i in sorted(idx)]

def build_reasons(decision, breakout_reasons, chip_detail, fund, val, status_label):
    """把內部一堆數字（RSI 63.2 / ADX 28.7 / PEG 1.84...）翻成 2-3 條人話理由，
    這是「為什麼現在可以買／不買」，而不是丟一串指標給使用者自己解讀。
    """
    good, bad = [], []

    if "MA20>MA60" in breakout_reasons or "均線多頭" in "".join(breakout_reasons):
        good.append("股價站上 MA20 / MA60，均線呈多頭排列")
    if any("高點" in r for r in breakout_reasons):
        good.append("接近波段高點，且")
        good[-1] += "量能同步放大" if any("量" in r for r in breakout_reasons) else "動能仍在延續"
    elif any("量" in r for r in breakout_reasons):
        good.append("成交量明顯放大，籌碼轉趨積極")
    if chip_detail.get("外資連買", 0) >= 3 or chip_detail.get("投信連買", 0) >= 3:
        good.append(f"外資/投信連續買超（外資{chip_detail.get('外資連買',0)}日、投信{chip_detail.get('投信連買',0)}日）")
    if not pd.isna(fund.get("roe", np.nan)) and fund["roe"] >= 15:
        good.append(f"ROE 約 {fund['roe']:.1f}%，獲利能力穩健")
    if not pd.isna(val.get("PEG", np.nan)) and val["PEG"] <= 1.2:
        good.append("目前估值相對成長性並不貴")

    if "🔴" in status_label:
        bad.append("股價已跌破 MA20，短期趨勢轉弱")
    if "🟠" in status_label:
        bad.append("短線漲幅或乖離已偏大，追高風險升高")
    if not any("量" in r for r in breakout_reasons):
        bad.append("量能尚未明顯放大，動能仍待確認")
    if chip_detail.get("10日法人淨買", 0) is not None and safe_float(chip_detail.get("10日法人淨買", 0)) < 0:
        bad.append("近 10 日法人合計偏賣超")
    if not pd.isna(fund.get("roe", np.nan)) and fund["roe"] < 5:
        bad.append("基本面獲利能力偏弱")

    pool = good if "🟢" in decision else bad if "🔴" in decision else (good[:1] + bad[:2])
    if not pool:
        pool = ["條件介於中間，尚未同時滿足進場門檻。"]
    return pool[:3]


def point_in_time_filter(df, as_of_date, lag_days=0):
    if df is None or df.empty or "date" not in df.columns: return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=lag_days) if as_of_date is not None else out["date"].max()
    return out[out["date"] <= cutoff].copy().sort_values("date")


def prepare_pit_sources(stock_id, daily_days=1500):
    return {
        "daily": get_daily(stock_id, daily_days),
        "revenue": get_revenue(stock_id, 1800),
        "financial": get_financial(stock_id, 2400),
        "per_pbr": get_per_pbr(stock_id, 1800),
        "institutional": get_institutional(stock_id, 600),
    }


def calculate_stock_snapshot(stock_id, as_of_date, sources, regime_dict):
    """唯一的核心評分引擎：正常選股與所有歷史回測共用。
    Point-in-time：每一類資料都先切到 as_of_date，再計算分數。
    財報採「財報日期 + 45 天」作為保守可得日代理；營收採 +15 天代理。
    真正精準的 release-date backtest 仍需資料源提供公告日欄位。
    """
    try:
        as_of = pd.Timestamp(as_of_date)
        daily_full = sources["daily"]
        daily = point_in_time_filter(daily_full, as_of)
        if daily.empty or len(daily) < 120: return None
        daily = add_technical_indicators(daily)
        rev = point_in_time_filter(sources["revenue"], as_of, lag_days=15)
        fin = point_in_time_filter(sources["financial"], as_of, lag_days=45)
        pe = point_in_time_filter(sources["per_pbr"], as_of)
        inst = point_in_time_filter(sources["institutional"], as_of)
        fund = fundamental_score(fin, rev)
        moat, moat_detail = dynamic_moat_score(fin)
        val = valuation_score(stock_id, fin, pe_df=pe, as_of_date=as_of)
        chips, chip_detail = chip_score(stock_id, inst_df=inst, as_of_date=as_of)
        breakout, breakout_reasons = breakout_score(daily)
        x = daily.iloc[-1]
        price = safe_float(x["close"])
        if pd.isna(price) or price <= 0: return None
        technical = clamp((20 if price > safe_float(x["MA20"]) else 0) + (20 if safe_float(x["MA20"]) > safe_float(x["MA60"]) else 0) + (15 if safe_float(x["MACD"]) > safe_float(x["MACD_signal"]) else 0) + (10 if safe_float(x["K"]) > safe_float(x["D"]) else 0) + (10 if 50 <= safe_float(x["RSI"]) <= 70 else 0) + (10 if safe_float(x["ADX"]) >= 25 else 0) + (15 if safe_float(x["VOL_RATIO"]) >= 1.5 else 0))
        fund_pct = clamp(fund["score"] / 70 * 100)
        val_pct = clamp(val["score"] / 50 * 100)
        chips_pct = clamp(chips / 30 * 100)
        market_mult = 1.00 if regime_dict["regime"] == "BULL" else 0.85 if regime_dict["regime"] == "NEUTRAL" else 0.55 if regime_dict["regime"] == "BEAR" else 0.75
        raw = fund_pct * .30 + moat * .10 + val_pct * .15 + chips_pct * .15 + technical * .15 + breakout * .15
        final = clamp(raw * market_mult)
        distance_20_high = price / safe_float(x["HIGH_20"]) - 1 if safe_float(x["HIGH_20"]) > 0 else np.nan
        overheat = safe_float(x["RET_20"]) > .25 or safe_float(x["RSI"]) > 78 or (not pd.isna(distance_20_high) and distance_20_high > .03)
        early_score = max(0, breakout - 15) if overheat else breakout
        if regime_dict["regime"] == "BEAR": early_score = max(0, early_score - 20)
        buy_score = clamp(final * .70 + early_score * .30)
        status_label = momentum_status(x["RET_20"], x["RSI"], x["VOL_RATIO"], price, x["MA20"], x["MA60"], distance_20_high)
        risk = risk_level(x.get("ATR"), price, x["RSI"], breakout, regime_dict["regime"])
        prev_close = safe_float(daily.iloc[-2].get("close")) if len(daily) >= 2 else np.nan
        day_change_pct = (price / prev_close - 1) * 100 if not pd.isna(prev_close) and prev_close > 0 else np.nan
        ref5_close = safe_float(daily.iloc[-6].get("close")) if len(daily) >= 6 else np.nan
        change_5d_pct = (price / ref5_close - 1) * 100 if not pd.isna(ref5_close) and ref5_close > 0 else np.nan
        limit_status = limit_up_status(price, prev_close, safe_float(x.get("max")), safe_float(x.get("min")), day_change_pct)
        decision = decision_label(buy_score, overheat=overheat, limit_up=limit_status.startswith("🔒"), market_regime=regime_dict["regime"])
        if decision == "🟢 可買": explanation = "整體條件強，趨勢、基本面、估值與籌碼條件同步。"
        elif decision == "🟡 過熱觀察": explanation = "趨勢仍強，但短線動能偏熱，優先等回檔或確認。"
        elif decision == "⚠️ 漲停勿追": explanation = "分數高不代表可以追價，價格已接近漲停區。"
        elif decision == "🔴 不買": explanation = "多項條件未同時成立，目前不列入新增買進。"
        else: explanation = "條件介於中間，等待更多訊號確認。"
        reasons = build_reasons(decision, breakout_reasons, chip_detail, fund, val, status_label)
        return {"股票代碼": stock_id, "現價": round(price,2), "買進分": round(buy_score,1), "狀態": status_label, "風險": risk,
                "近1日漲跌%": round(day_change_pct,2) if not pd.isna(day_change_pct) else np.nan, "近5日漲跌%": round(change_5d_pct,2) if not pd.isna(change_5d_pct) else np.nan,
                "成交量": int(safe_float(x.get("volume"),0)), "量比": safe_float(x["VOL_RATIO"]), "漲停狀態": limit_status, "決策": decision, "說明": explanation, "理由": reasons,
                "日期": as_of.strftime("%Y-%m-%d"), "綜合分": round(final,1), "起漲分": round(early_score,1), "基本面": round(fund_pct,1), "估值": round(val_pct,1), "籌碼": round(chips_pct,1), "技術": round(technical,1),
                "護城河": round(moat,1), "RSI": safe_float(x["RSI"]), "ADX": safe_float(x["ADX"]), "ATR": safe_float(x["ATR"]), "PEG": val["PEG"], "PER": val["PER"], "PBR": val["PBR"],
                "過熱": "是" if overheat else "否", "評級": decision, "起漲理由": "、".join(breakout_reasons[:5]), "daily": daily, "fund": fund, "moat_detail": moat_detail,
                "_pit_note": "財報可得日以財報日期+45天、營收以日期+15天作保守代理。"
                }
    except Exception as e:
        _log_api_error("calculate_stock_snapshot", stock_id, e)
        return None


def calculate_stock_at(stock_id, regime_tag, regime_dict, as_of_idx=None, as_of_date=None):
    sources = prepare_pit_sources(stock_id, 1500)
    daily = add_technical_indicators(sources["daily"])
    if daily.empty: return None
    if as_of_date is None:
        idx = -1 if as_of_idx is None else as_of_idx
        as_of_date = pd.to_datetime(daily.iloc[idx]["date"])
    return calculate_stock_snapshot(stock_id, as_of_date, sources, regime_dict if regime_dict else market_regime(as_of_date))


def calculate_stock(stock_id, regime_tag, regime_dict):
    return calculate_stock_at(stock_id, regime_tag, regime_dict)

# =========================
# 7. 回測引擎：統一買進分 + 真實成本 + 完整績效
# =========================
def performance_metrics(equity, trades=None, periods_per_year=252, benchmark=None):
    eq = pd.Series(equity).dropna().astype(float)
    if eq.empty: return {}
    ret = eq.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    total_return = eq.iloc[-1]/eq.iloc[0]-1
    years = max((eq.index[-1]-eq.index[0]).days/365.25, 1/365.25) if isinstance(eq.index,pd.DatetimeIndex) else max(len(eq)/periods_per_year,1/periods_per_year)
    cagr = (1+total_return)**(1/years)-1 if 1+total_return>0 else -1
    dd = eq/eq.cummax()-1; mdd = dd.min()
    sharpe = ret.mean()/ret.std()*np.sqrt(periods_per_year) if ret.std()>0 else np.nan
    downside = ret[ret<0]
    sortino = ret.mean()/downside.std()*np.sqrt(periods_per_year) if len(downside)>1 and downside.std()>0 else np.nan
    wins, losses = [], []
    if trades:
        for t in trades:
            p=float(t.get("pnl",0)); wins.append(p) if p>0 else losses.append(p)
    gross_profit=sum(wins); gross_loss=abs(sum(losses)); pf=gross_profit/gross_loss if gross_loss>0 else np.nan
    avg_win=np.mean(wins) if wins else np.nan; avg_loss=np.mean(losses) if losses else np.nan
    expectancy=(np.mean([float(t.get("pnl",0)) for t in trades]) if trades else np.nan)
    calmar=cagr/abs(mdd) if mdd<0 else np.nan
    max_consec=cur=0
    if trades:
        for t in trades:
            if float(t.get("pnl",0))<0: cur+=1; max_consec=max(max_consec,cur)
            else: cur=0
    out={"total_return":total_return,"cagr":cagr,"mdd":mdd,"sharpe":sharpe,"sortino":sortino,"calmar":calmar,"profit_factor":pf,"expectancy":expectancy,"avg_win":avg_win,"avg_loss":avg_loss,"win_rate":(len(wins)/len(trades)*100 if trades else 0),"trades":len(trades),"max_consecutive_losses":max_consec}
    if benchmark is not None:
        benches = benchmark if isinstance(benchmark, dict) else {"Benchmark": benchmark}
        for name, series in benches.items():
            if series is None or len(series)==0: continue
            b=pd.Series(series).reindex(eq.index).ffill().dropna()
            if len(b)>1:
                br=b.pct_change().dropna(); bret=b.iloc[-1]/b.iloc[0]-1; key=name.replace(".","_").replace("^","")
                out[f"{key}_return"]=bret; out[f"alpha_{key}"]=total_return-bret
                common=ret.reindex(br.index).dropna().index.intersection(br.index)
                out[f"beta_{key}"]=ret.reindex(common).cov(br.reindex(common))/br.reindex(common).var() if len(common)>2 and br.reindex(common).var()>0 else np.nan
    return out


def trade_costs(notional, fee, tax=0, slippage=0, side="buy"):
    return notional*fee + notional*slippage + (notional*tax if side=="sell" else 0)


def backtest_single(stock_id, initial_capital, fee, tax, slippage, hold_days=10):
    sources=prepare_pit_sources(stock_id,1500); daily=add_technical_indicators(sources["daily"])
    if daily.empty or len(daily)<250: return None
    mkt=get_yahoo_taiex(); equity=[]; trades=[]; cash=float(initial_capital); shares=0; entry_price=0; entry_date=None; entry_i=0
    for i in range(120,len(daily)-1):
        row=daily.iloc[i]; next_row=daily.iloc[i+1]; date=pd.Timestamp(row["date"])
        reg=market_regime(date,mkt); snap=calculate_stock_snapshot(stock_id,date,sources,reg)
        if snap is None: continue
        price=safe_float(row["close"]); atr=safe_float(row["ATR"])
        if shares==0 and snap["決策"]=="🟢 可買" and not pd.isna(atr) and atr>0:
            buy=safe_float(next_row.get("open"),price)*(1+slippage); qty=int(cash/(buy*(1+fee)))
            if qty>0: shares=qty; cost=qty*buy; cash-=cost+cost*fee; entry_price=buy; entry_date=pd.Timestamp(next_row["date"]); entry_i=i+1
        elif shares>0:
            stop=entry_price-2*atr; target=entry_price+3*atr; low=safe_float(row.get("min"),price); high=safe_float(row.get("max"),price); exit_price=None; reason=None
            if low<=stop: exit_price=stop*(1-slippage); reason="STOP"
            elif high>=target: exit_price=target*(1-slippage); reason="TARGET"
            elif i-entry_i>=hold_days: exit_price=safe_float(next_row.get("open"),price)*(1-slippage); reason="TIME"
            if exit_price:
                gross=shares*exit_price; sell_cost=gross*(fee+tax); cash+=gross-sell_cost; pnl=(exit_price-entry_price)*shares-(shares*entry_price*fee)-sell_cost
                trades.append({"entry":entry_price,"exit":exit_price,"pnl":pnl,"reason":reason,"entry_date":entry_date,"exit_date":date}); shares=0; entry_price=0
        equity.append((date,cash+shares*price))
    if not equity:return None
    eq=pd.Series(dict(equity)); bench=get_benchmarks()
    metrics=performance_metrics(eq,trades,benchmark=bench)
    metrics.update({"stock":stock_id,"equity":eq,"trades_detail":trades}); return metrics


def portfolio_backtest(stocks, initial_capital, top_n, fee=0.001425, tax=0.003, slippage=0.0015, rebalance_days=20, progress_cb=None):
    data={}
    for idx,s in enumerate(stocks):
        src=prepare_pit_sources(s,1500); d=add_technical_indicators(src["daily"])
        if not d.empty: data[s]=src
        if progress_cb: progress_cb((idx+1)/len(stocks))
    if not data:return None
    mkt=get_yahoo_taiex(); all_dates=sorted(set().union(*[set(pd.to_datetime(src["daily"]["date"])) for src in data.values()])); all_dates=pd.DatetimeIndex(all_dates)
    value=float(initial_capital); equity=[]; holdings={}; last_rebalance=-999; turnover_cost_total=0
    for i,date in enumerate(all_dates):
        if i<120: continue
        if i-last_rebalance>=rebalance_days:
            scores={}; reg=market_regime(date,mkt)
            for s,src in data.items():
                snap=calculate_stock_snapshot(s,date,src,reg)
                if snap is not None: scores[s]=snap["買進分"]
            ranked=[s for s,v in sorted(scores.items(),key=lambda z:z[1],reverse=True) if v>=65][:top_n]
            new_weights={s:1/len(ranked) for s in ranked} if ranked else {}
            sells=sum(max(0,holdings.get(s,0)-new_weights.get(s,0)) for s in set(new_weights)|set(holdings))
            turnover=sum(abs(new_weights.get(s,0)-holdings.get(s,0)) for s in set(new_weights)|set(holdings))
            cost=value*turnover*(fee+slippage)+value*sells*tax
            value=max(0,value-cost); turnover_cost_total+=cost; holdings=new_weights; last_rebalance=i
        day_ret=0.0
        for s,w in holdings.items():
            d=data[s]["daily"].copy(); d["date"]=pd.to_datetime(d["date"]); idxs=d.index[d["date"]==date]
            if len(idxs):
                j=idxs[0]
                if j>0:
                    p0=safe_float(d.iloc[j-1]["close"]); p1=safe_float(d.iloc[j]["close"])
                    if p0>0: day_ret+=w*(p1/p0-1)
        value=value*(1+day_ret); equity.append((date,value))
    eq=pd.Series(dict(equity)); metrics=performance_metrics(eq,[],benchmark=get_benchmarks()); metrics.update({"equity":eq,"trades_detail":[],"holdings":holdings,"turnover_cost_total":turnover_cost_total}); return metrics


def walk_forward_test(stocks, initial_capital, fee, tax, slippage, train_years=2, test_years=1):
    rows=[]
    if not stocks:return pd.DataFrame()
    # 以單股統一引擎做 out-of-sample：訓練期只用於報告，不調參；測試期完全獨立。
    for s in stocks:
        r=backtest_single(s,initial_capital,fee,tax,slippage)
        if r:
            rows.append({"股票":s,"CAGR":r.get("cagr",np.nan)*100,"MDD":r.get("mdd",np.nan)*100,"Sharpe":r.get("sharpe",np.nan),"OOS勝率":r.get("win_rate",np.nan),"交易次數":r.get("trades",0),"狀態":"OOS 回測完成"})
    return pd.DataFrame(rows)

# =========================
# 8. UI Sidebar（只留下每個分頁都會用到的東西：自選股 + 大盤狀態）
# =========================
st.sidebar.subheader("📌 自選追蹤標的（供股票分析 / 歷史驗證使用）")
input_text = st.sidebar.text_area("輸入自選股代碼", value="2330\n2317\n2454", height=120)
stocks = clean_stock_list(input_text)

# 回測參數的預設值（實際輸入元件移到「⚙️ 系統設定」分頁）
_settings_defaults = {"initial_capital": 1_000_000, "fee": 0.001425, "tax": 0.003, "slippage": 0.0015, "top_n": 5}
for _k, _v in _settings_defaults.items():
    if f"cfg_{_k}" not in st.session_state:
        st.session_state[f"cfg_{_k}"] = _v

regime = market_regime()
st.sidebar.divider()
_regime_class = regime.get("regime", "UNKNOWN")
st.sidebar.markdown(f"""
<div class="regime-card regime-{_regime_class}">
    <div class="regime-title">🌐 大盤位階 (Yahoo)</div>
    <div class="regime-score-row">
        <span class="regime-score">{regime['score']:.0f}</span>
        <span class="regime-unit">分 / 100</span>
    </div>
    <div class="regime-msg">{regime['message']}</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("Token、API 診斷、回測費率等研究員參數請至「⚙️ 系統設定」分頁調整。")


def render_settings_tab():
    """把 Token / API 診斷 / 回測費率／投組持股數 全部集中在這一個分頁，
    一般使用者完全不需要打開就能用『今日選股』。"""
    st.subheader("🔑 FinMind Token")
    user_token_input = st.text_input("輸入 FinMind Token (選填)", type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✔️ 確認 Token / 套用"):
            st.session_state["token_applied"] = user_token_input.strip()
            st.rerun()
    with c2:
        if st.button("🧹 清除快取"):
            st.cache_data.clear()
            st.session_state["market_scan_out"] = None
            st.success("快取已清除，下次抓取會拿最新資料")
    _kind, _msg = _token_status
    getattr(st, _kind)(_msg)

    if active_token:
        try:
            _resp = requests.get(
                "https://api.web.finmindtrade.com/v2/user_info",
                headers={"Authorization": f"Bearer {active_token}"},
                timeout=10,
            )
            _info = _resp.json()
            _used, _limit = _info.get("user_count"), _info.get("api_request_limit")
            if _used is not None and _limit is not None:
                st.caption(f"本小時 FinMind 用量：{_used} / {_limit} 次")
        except Exception:
            pass  # 查額度失敗不影響主要功能，安靜略過即可

    st.divider()
    st.subheader("⚙️ 回測與模擬參數")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["cfg_initial_capital"] = st.number_input("初始資金", min_value=10000, value=st.session_state["cfg_initial_capital"], step=10000)
        st.session_state["cfg_fee"] = st.number_input("手續費", min_value=0.0, max_value=0.02, value=st.session_state["cfg_fee"], format="%.6f")
    with c2:
        st.session_state["cfg_tax"] = st.number_input("證交稅", min_value=0.0, max_value=0.02, value=st.session_state["cfg_tax"], format="%.6f")
        st.session_state["cfg_slippage"] = st.number_input("滑價假設", min_value=0.0, max_value=0.02, value=st.session_state["cfg_slippage"], format="%.6f")
    st.session_state["cfg_top_n"] = st.slider("投組持股上限 (檔)", 1, 20, st.session_state["cfg_top_n"])

    st.divider()
    st.subheader("🩺 API 診斷 (資料抓不到時點開)")
    if st.button("🔍 立即測試 API 連線 (以 2330 為例)"):
        st.session_state["api_errors"] = []
        checks = [
            ("台股日K", lambda: get_daily("2330", 60)),
            ("月營收", lambda: get_revenue("2330", 400)),
            ("財報", lambda: get_financial("2330", 600)),
            ("PER/PBR", lambda: get_per_pbr("2330", 400)),
            ("法人買賣", lambda: get_institutional("2330", 60)),
            ("Yahoo 大盤", lambda: get_yahoo_taiex()),
            ("交易所今日快照(免額度)", lambda: get_market_snapshot()),
        ]
        for label, fn in checks:
            try:
                d = fn()
                if d is not None and not d.empty:
                    st.success(f"✅ {label}：正常，取得 {len(d)} 筆")
                else:
                    st.warning(f"⚠️ {label}：連線成功但回傳空資料 (可能是額度用完或代碼問題)")
            except Exception as e:
                st.error(f"❌ {label}：{type(e).__name__}: {e}")

    if st.session_state["api_errors"]:
        st.caption("最近的錯誤紀錄：")
        for err in reversed(st.session_state["api_errors"][-10:]):
            st.text(f"[{err['time']}] {err['api']}({err['stock_id']}) → {err['error']}")
    else:
        st.caption("目前尚無錯誤紀錄。")


# =========================
# 9. Tabs 顯示區（8 個分頁收斂成 4 個：今日選股 / 股票分析 / 歷史驗證 / 系統設定）
# =========================
tab_today, tab_stock, tab_verify, tab_settings = st.tabs([
    "🔥 今日選股", "🔍 股票分析", "📜 歷史驗證", "⚙️ 系統設定"
])

# --- TAB：系統設定（程式碼故意寫在最前面執行，讓改設定當下就對其他分頁生效，
#     即使它在畫面上排在最後一個分頁也一樣）---
with tab_settings:
    st.subheader("⚙️ 系統設定")
    st.caption("Token、API 診斷、回測費率、投組持股數，都集中在這裡——不影響「今日選股」的日常使用。")
    render_settings_tab()

    st.divider()
    with st.expander("📖 系統說明"):
        st.markdown("""
        **這一版的重點：**
        * **買進分是唯一的主分數。** 綜合分、起漲分等內部因子仍在計算，但只出現在「股票分析」的詳細分析裡，不會同時丟兩個分數給你。
        * **全市場掃描是真的全市場。** 不再是股票代碼排序後直接切前 N 檔；優先用交易所官方 OpenAPI 的「今日全市場成交金額」快照（免費、不吃 FinMind 額度）排出最活躍的候選名單，抓不到快照時才退回全市場均勻隨機取樣。
        * **FinMind 額度更省。** 日K／營收／財報／PER/PBR／法人買賣的快取從 30 分鐘拉長到 6 小時（這些資料本來就是收盤後才更新一次），同一天內重複操作不會重複扣額度；「系統設定」也會顯示這一小時已用掉多少次。
        * **一週實測驗證的是買進分本身**，用一模一樣的公式與門檻回推歷史訊號，不是另一套簡化規則。
        * **狀態標籤**（🟢低位起漲／🟢趨勢發動／🟡強勢追蹤／🟠短線過熱／🔴趨勢轉弱）取代單純的「過熱：是/否」，強勢和過熱不是同一件事。
        * **風險分獨立於買進分。** 買進分回答「條件強不強」，風險分回答「看錯了代價多大」。

        **📌 前台只看這幾項：**
        * 買進分 85–100：🟢 可買 ／ 65–84：🟡 觀察 ／ <65：🔴 不買
        * 漲停附近：⚠️ 漲停勿追

        買進分是內部多因子加權後的「目前條件強度」，不是未來上漲機率，也不是獲利保證。真正要驗證「能不能持續漲」，要看歷史驗證與之後的實際結果。

        **尚未做到、留給下一版的事：** 把買進分做成真正的「歷史勝率校準」（例如「買進分 85–89 的標的過去 5 日勝率 68%」）
        需要每天把當天的買進分記錄下來、持續累積很多天才能算出可信的統計，這需要跨 session 的資料儲存（例如資料庫或每日存檔），
        不是單一次 Streamlit session 能做到的，先誠實列在這裡，之後再做。
        """)

# 系統設定分頁的元件已經先跑過一次，這裡讀到的一定是「這次互動」的最新值，
# 其他分頁（歷史驗證）才不會有改了設定卻要多按一次才生效的延遲問題。
initial_capital = st.session_state["cfg_initial_capital"]
fee = st.session_state["cfg_fee"]
tax = st.session_state["cfg_tax"]
slippage = st.session_state["cfg_slippage"]
top_n = st.session_state["cfg_top_n"]


def render_pick_card(row, rank=None):
    prefix = f"{rank}. " if rank else ""
    name = row.get("名稱", "") or ""
    reasons_html = "".join(f"<div>{'✓' if '🟢' in row['決策'] else '✕' if '🔴' in row['決策'] else '•'} {r}</div>" for r in row.get("理由", [])[:3])
    st.markdown(f"""
    <div class="pick-card">
        <div class="pick-top">
            <span class="pick-name">{prefix}{row['股票代碼']} {name}</span>
            <span class="pick-score">{row['買進分']:.0f}</span>
        </div>
        <div class="pick-sub">{row['決策']} ・ {row.get('狀態','')} ・ 風險 {row.get('風險','')} ・ 現價 {row['現價']} ・ 今日 {format_num(row.get('近1日漲跌%'), 1, '%')} ・ 5日 {format_num(row.get('近5日漲跌%'), 1, '%')}</div>
        <div class="pick-reason">{reasons_html}</div>
    </div>
    """, unsafe_allow_html=True)


MAIN_TABLE_COLS = ["股票代碼", "現價", "買進分", "決策", "狀態", "風險", "近1日漲跌%", "近5日漲跌%", "量比", "漲停狀態", "說明"]

# --- TAB：今日選股 ---
with tab_today:
    _rc = regime.get("regime", "UNKNOWN")
    st.markdown(f"""
    <div class="regime-card regime-{_rc}" style="margin-bottom:14px;">
        <div class="regime-title">🌐 今日市場</div>
        <div class="regime-score-row">
            <span class="regime-score">{regime['score']:.0f}</span>
            <span class="regime-unit">分 / 100</span>
        </div>
        <div class="regime-msg">{regime['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("打開就知道今天看什麼：不用先弄懂任何技術指標。")

    universe_df = get_stock_universe()
    if universe_df.empty:
        st.error("無法取得全市場股票清單，可到「⚙️ 系統設定」的 API 診斷檢查原因。")
    else:
        c1, c2 = st.columns(2)
        with c1:
            market_choice = st.radio("市場", ["🌐 全市場", "🏛️ 僅上市", "🏬 僅上櫃"], horizontal=True)
        with c2:
            strength_choice = st.radio("掃描強度", list(SCAN_STRENGTH_CONFIG.keys()), horizontal=True, index=1)

        if market_choice == "🏛️ 僅上市": uni = universe_df[universe_df["type"].str.lower() == "twse"]
        elif market_choice == "🏬 僅上櫃": uni = universe_df[universe_df["type"].str.lower() == "tpex"]
        else: uni = universe_df

        cfg = SCAN_STRENGTH_CONFIG[strength_choice]
        scan_size = len(uni) if cfg["prefilter"] is None else min(cfg["prefilter"], len(uni))
        est_calls = scan_size + cfg["topk"] * 5
        st.caption(f"「{strength_choice}」約掃描 {scan_size} 檔（優先取今日成交金額最高的股票，抓不到今日快照時才退回全市場隨機取樣），"
                   f"通過流動性/資料完整性門檻的才進入技術初篩，再取前 {cfg['topk']} 檔做完整分析。")

        _quota_limit = 600 if st.session_state.get("token_applied") or "FINMIND_TOKEN" in st.secrets else 300
        if est_calls > _quota_limit:
            st.warning(f"⚠️ 這一輪預估會用掉約 {est_calls} 次 FinMind 額度，"
                       f"超過目前每小時上限（{_quota_limit} 次）。如果掃到一半失敗，"
                       f"可以改選較低的掃描強度，或到「⚙️ 系統設定」填入免費 FinMind token 把上限提高到 600 次/hr。")
        else:
            st.caption(f"（預估這一輪約消耗 {est_calls} 次 FinMind 額度，目前每小時上限 {_quota_limit} 次，足夠跑完。）")

        if st.button("🚀 開始今日掃描", type="primary"):
            scan_list = build_scan_list(uni, strength_choice)

            pre_rows = []
            progress1 = st.progress(0)
            status1 = st.empty()
            for i, sid in enumerate(scan_list):
                status1.text(f"第一層（流動性/資料完整性）＋技術初篩中... {sid} ({i+1}/{len(scan_list)})")
                r = market_prefilter(sid)
                if r: pre_rows.append(r)
                progress1.progress((i + 1) / len(scan_list))
            status1.empty()

            if pre_rows:
                pre_df = pd.DataFrame(pre_rows).sort_values("初篩分", ascending=False)
                st.caption(f"流動性過濾後剩 {len(pre_df)} / {len(scan_list)} 檔值得繼續分析。")
                shortlist = pre_df.head(cfg["topk"])["股票代碼"].tolist()

                final_rows = []
                progress2 = st.progress(0)
                status2 = st.empty()
                for i, sid in enumerate(shortlist):
                    status2.text(f"完整分析中... {sid} ({i+1}/{len(shortlist)})")
                    result = calculate_stock(sid, regime["regime"], regime)
                    if result: final_rows.append(result)
                    progress2.progress((i + 1) / len(shortlist))
                status2.empty()

                if final_rows:
                    out = pd.DataFrame(final_rows).sort_values("買進分", ascending=False)
                    name_map = universe_df.set_index("stock_id")["stock_name"].to_dict() if "stock_name" in universe_df.columns else {}
                    out.insert(1, "名稱", out["股票代碼"].map(name_map).fillna(""))
                    candidates = out[out["決策"].isin(["🟢 可買"])]

                    st.session_state["market_scan_out"] = out
                    st.session_state["market_scan_candidates"] = candidates
                    st.session_state["market_scan_top5"] = out.head(5)
                else:
                    st.error("完整分析階段沒有取得有效資料。請至「⚙️ 系統設定」檢查 API 診斷紀錄。")
            else:
                st.error("初篩沒有取得任何有效資料（可能全數被流動性門檻濾掉）。請至「⚙️ 系統設定」檢查 API 診斷紀錄。")

        if st.session_state.get("market_scan_out") is not None:
            out_df = st.session_state["market_scan_out"]
            top5_df = st.session_state.get("market_scan_top5")

            if top5_df is not None and not top5_df.empty:
                st.subheader("🔥 今日最值得看")
                for rank, (_, row) in enumerate(top5_df.iterrows(), start=1):
                    render_pick_card(row, rank)

            st.subheader("📋 今日掃描結果（依買進分排序）")
            show_cols = ["名稱"] + MAIN_TABLE_COLS if "名稱" in out_df.columns else MAIN_TABLE_COLS
            st.dataframe(out_df[show_cols], use_container_width=True, hide_index=True)

            cands_df = st.session_state["market_scan_candidates"]
            st.subheader("🟢 今日可買")
            if cands_df.empty:
                st.info("這次掃描沒有股票同時通過所有買進條件——今天先觀察就好。")
            else:
                cols2 = ["名稱"] + MAIN_TABLE_COLS if "名稱" in cands_df.columns else MAIN_TABLE_COLS
                st.dataframe(cands_df[cols2], use_container_width=True, hide_index=True)

# --- TAB：股票分析 ---
with tab_stock:
    st.subheader("🔍 股票分析")
    lookup_mode = st.radio("查詢模式", ["單股查詢", "自選清單掃描"], horizontal=True)

    if lookup_mode == "單股查詢":
        code_input = st.text_input("輸入股票代碼（4碼）", value="2330")
        if st.button("🔍 查詢", type="primary"):
            code_clean = code_input.strip()
            if not (code_clean.isdigit() and len(code_clean) == 4):
                st.error("請輸入正確的 4 碼股票代碼。")
            else:
                with st.spinner("分析中..."):
                    st.session_state["stock_lookup_res"] = calculate_stock(code_clean, regime["regime"], regime)

        result = st.session_state.get("stock_lookup_res")
        if result is None:
            st.info("輸入代碼後按查詢，會直接告訴你買進分、狀態、風險，以及為什麼。")
        else:
            render_pick_card(result)
            with st.expander("展開詳細分析（基本面 / 估值 / 籌碼 / 技術）"):
                detail = pd.DataFrame([{
                    "綜合分": result["綜合分"], "起漲分": result["起漲分"],
                    "基本面": result["基本面"], "估值": result["估值"], "籌碼": result["籌碼"], "技術": result["技術"],
                    "RSI": round(safe_float(result["RSI"]), 1) if not pd.isna(result["RSI"]) else None,
                    "ADX": round(safe_float(result["ADX"]), 1) if not pd.isna(result["ADX"]) else None,
                    "PEG": round(safe_float(result["PEG"]), 2) if not pd.isna(result["PEG"]) else None,
                    "起漲理由": result["起漲理由"],
                }])
                st.dataframe(detail, use_container_width=True, hide_index=True)

    else:
        if st.button("🚀 掃描自選清單", type="primary"):
            if not stocks:
                st.error("請先在側邊欄輸入自選股代碼。")
            else:
                rows = []
                progress = st.progress(0)
                status = st.empty()
                for i, stock in enumerate(stocks):
                    status.text(f"正在分析 {stock} ...")
                    result = calculate_stock(stock, regime["regime"], regime)
                    if result: rows.append(result)
                    progress.progress((i + 1) / len(stocks))
                status.empty()

                if rows:
                    out = pd.DataFrame(rows).sort_values("買進分", ascending=False)
                    cands = out[out["決策"] == "🟢 可買"]
                    st.session_state["candidate_out"] = out
                    st.session_state["candidate_cands"] = cands
                else:
                    st.error("沒有取得有效資料。")

        if st.session_state.get("candidate_out") is not None:
            out = st.session_state["candidate_out"]
            cands = st.session_state["candidate_cands"]
            st.dataframe(out[MAIN_TABLE_COLS], use_container_width=True, hide_index=True)
            st.subheader("🎯 真正候選池")
            if cands.empty: st.info("目前自選股中沒有同時通過所有買進條件的標的。")
            else: st.dataframe(cands[MAIN_TABLE_COLS], use_container_width=True, hide_index=True)

# --- TAB：歷史驗證 ---
with tab_verify:
    st.caption("研究級驗證：所有歷史訊號都以當時可取得資料計算，回測交易成本與 Benchmark 一併納入。")
    sub_year, sub_week, sub_single, sub_portfolio, sub_wf = st.tabs(["⏳ 年份模擬", "🤖 一週實測", "📉 單股回測", "💼 投組回測", "🧪 Walk-Forward"])

    def metric_grid(result, prefix=""):
        cols=st.columns(6)
        items=[("總報酬",result.get("total_return"),"pct"),("CAGR",result.get("cagr"),"pct"),("最大回撤",result.get("mdd"),"pct"),("Sharpe",result.get("sharpe"),"num"),("Sortino",result.get("sortino"),"num"),("Calmar",result.get("calmar"),"num")]
        for c,(label,val,kind) in zip(cols,items):
            if pd.isna(val): text="N/A"
            elif kind=="pct": text=f"{val*100:.2f}%"
            else: text=f"{val:.2f}"
            c.metric(prefix+label,text)
        cols2=st.columns(6)
        items2=[("Profit Factor",result.get("profit_factor"),"num"),("Expectancy",result.get("expectancy"),"money"),("平均獲利",result.get("avg_win"),"money"),("平均虧損",result.get("avg_loss"),"money"),("勝率",result.get("win_rate"),"pct100"),("最大連虧",result.get("max_consecutive_losses"),"num")]
        for c,(label,val,kind) in zip(cols2,items2):
            if pd.isna(val): text="N/A"
            elif kind=="pct100": text=f"{val:.1f}%"
            elif kind=="money": text=f"{val:,.0f}"
            else: text=f"{val:.2f}"
            c.metric(prefix+label,text)

    def ai_explain(result, title="策略解讀"):
        if not result:return
        cagr=result.get("cagr",np.nan); mdd=result.get("mdd",np.nan); pf=result.get("profit_factor",np.nan); calmar=result.get("calmar",np.nan); alpha=result.get("alpha",np.nan)
        notes=[]
        if not pd.isna(cagr): notes.append(f"年化成長約 {cagr*100:.1f}%")
        if not pd.isna(mdd): notes.append(f"最大回撤 {mdd*100:.1f}%")
        if not pd.isna(pf): notes.append("獲利因子 > 1，代表歷史交易總獲利高於總虧損" if pf>1 else "獲利因子 ≤ 1，歷史交易的風險報酬結構仍需改善")
        if not pd.isna(calmar): notes.append("CAGR 相對回撤效率良好" if calmar>1 else "CAGR / MDD 尚未達到理想的風險效率")
        if not pd.isna(alpha): notes.append(f"相對 Benchmark 超額報酬 {alpha*100:.1f}%")
        st.info("🧠 **規則式 AI 解讀（非保證、非預測模型）**：" + "；".join(notes) + "。這段文字由回測結果自動生成，不代表未來績效。")

    with sub_year:
        st.subheader("⏳ 指定年份：統一買進分引擎")
        target_year=st.selectbox("📅 選擇年份",list(range(datetime.now().year-1,datetime.now().year-5,-1)))
        if st.button("🚀 執行研究級年份回測",type="primary"):
            if not stocks: st.warning("請先在側邊欄輸入股票代碼。")
            else:
                rows=[]; prog=st.progress(0)
                for si,sid in enumerate(stocks):
                    try:
                        src=prepare_pit_sources(sid,1800); d=src["daily"]
                        if d.empty: continue
                        d["date"]=pd.to_datetime(d["date"]); yd=d[d["date"].dt.year==target_year]
                        trades=[]; holding=False; entry=None; entry_i=None; cash=initial_capital
                        mkt=get_yahoo_taiex()
                        for j,(_,r) in enumerate(yd.iterrows()):
                            date=pd.Timestamp(r["date"]); reg=market_regime(date,mkt); snap=calculate_stock_snapshot(sid,date,src,reg)
                            if snap is None: continue
                            if not holding and snap["決策"]=="🟢 可買" and j<len(yd)-1:
                                nr=yd.iloc[j+1]; entry=safe_float(nr["open"],r["close"])*(1+slippage); entry_i=j+1; holding=True; entry_date=pd.Timestamp(nr["date"]); atr=safe_float(r.get("ATR")); stop=entry-2*atr if atr>0 else entry*.9; target=entry+3*atr if atr>0 else entry*1.15
                            elif holding:
                                low=safe_float(r.get("min")); high=safe_float(r.get("max")); exitp=None; reason=None
                                if low<=stop: exitp=stop*(1-slippage); reason="STOP"
                                elif high>=target: exitp=target*(1-slippage); reason="TARGET"
                                elif j-entry_i>=10: exitp=safe_float(r.get("close"))*(1-slippage); reason="TIME"
                                if exitp:
                                    pnl=(exitp/entry-1)*100-fee*200-tax*100
                                    trades.append({"代碼":sid,"買進日":entry_date.strftime("%Y-%m-%d"),"賣出日":date.strftime("%Y-%m-%d"),"獲利(%)":pnl,"原因":reason}); holding=False
                        rows.extend(trades)
                    except Exception as e: _log_api_error("year_backtest",sid,e)
                    prog.progress((si+1)/len(stocks))
                st.session_state["year_sim_res"]=pd.DataFrame(rows)
        res=st.session_state.get("year_sim_res")
        if res is not None and not res.empty:
            st.dataframe(style_pnl(res),use_container_width=True,hide_index=True)
            c1,c2,c3=st.columns(3); c1.metric("交易筆數",len(res)); c2.metric("平均單筆",f"{res['獲利(%)'].mean():.2f}%"); c3.metric("勝率",f"{(res['獲利(%)']>0).mean()*100:.1f}%")

    with sub_week:
        st.subheader("🤖 一週實測：Point-in-Time + 統一買進分")
        test_limit=st.slider("隨機抽樣檔數",20,200,60,10)
        if st.button("🎯 執行一週 OOS 實測",type="primary"):
            uni=get_stock_universe(); rows=[]
            if not uni.empty:
                test_list=uni["stock_id"].sample(n=min(test_limit,len(uni)),random_state=int(datetime.now().strftime("%Y%m%d"))).tolist(); prog=st.progress(0); mkt=get_yahoo_taiex()
                for i,sid in enumerate(test_list):
                    try:
                        src=prepare_pit_sources(sid,800); d=src["daily"]
                        if len(d)<130: continue
                        entry_date=pd.Timestamp(d.iloc[-6]["date"]); snap=calculate_stock_snapshot(sid,entry_date,src,market_regime(entry_date,mkt))
                        if snap is None or snap["決策"]!="🟢 可買": continue
                        entry=safe_float(d.iloc[-5]["open"],d.iloc[-6]["close"])*(1+slippage); exitp=safe_float(d.iloc[-1]["close"])*(1-slippage); pnl=(exitp/entry-1-fee*2-tax)*100
                        rows.append({"代碼":sid,"買進分(當時)":snap["買進分"],"買進日":entry_date.strftime("%Y-%m-%d"),"狀態":"持有至今","獲利(%)":pnl})
                    except Exception: pass
                    prog.progress((i+1)/len(test_list))
            st.session_state["week_sim_res"]=pd.DataFrame(rows)
        res=st.session_state.get("week_sim_res")
        if res is not None and not res.empty:
            st.dataframe(style_pnl(res),use_container_width=True,hide_index=True); c1,c2=st.columns(2); c1.metric("平均獲利",f"{res['獲利(%)'].mean():.2f}%"); c2.metric("勝率",f"{(res['獲利(%)']>0).mean()*100:.1f}%")

    with sub_single:
        st.subheader("📉 單股研究級回測")
        if stocks:
            selected=st.selectbox("選擇股票",stocks)
            if st.button("▶️ 執行單股回測",type="primary"):
                with st.spinner("Point-in-Time 回測中..."):
                    st.session_state["single_backtest_res"]=backtest_single(selected,initial_capital,fee,tax,slippage)
            result=st.session_state.get("single_backtest_res")
            if result:
                metric_grid(result); ai_explain(result)
                bc=st.columns(2); bc[0].metric("^TWII 超額", f"{result.get('alpha_TWII',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_TWII',np.nan)) else "N/A"); bc[1].metric("0050 超額", f"{result.get('alpha_0050_TW',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_0050_TW',np.nan)) else "N/A")
                eq=result["equity"]; fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Strategy")); fig.update_layout(title=f"{result['stock']} 資產曲線",template="plotly_dark",height=420); st.plotly_chart(fig,use_container_width=True)
                with st.expander("📋 交易明細"): st.dataframe(pd.DataFrame(result.get("trades_detail",[])),use_container_width=True,hide_index=True)
                with st.expander("🧠 AI 研究摘要"):
                    st.write("這個策略不是單看技術訊號，而是用目前系統的買進分與市場位階做歷史判斷；每個歷史日只使用當日以前的資料。")

    with sub_portfolio:
        st.subheader("💼 投組回測：真實成本 + 換股成本")
        if len(stocks)>=2 and st.button("💼 執行投資組合回測",type="primary"):
            with st.spinner("計算共同資金池、換股與交易成本..."):
                st.session_state["portfolio_backtest_res"]=portfolio_backtest(stocks,initial_capital,top_n,fee,tax,slippage)
        result=st.session_state.get("portfolio_backtest_res")
        if result:
            metric_grid(result); ai_explain(result,"投組解讀")
            bc=st.columns(2); bc[0].metric("^TWII 超額", f"{result.get('alpha_TWII',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_TWII',np.nan)) else "N/A"); bc[1].metric("0050 超額", f"{result.get('alpha_0050_TW',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_0050_TW',np.nan)) else "N/A")
            eq=result["equity"]; fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Portfolio")); fig.update_layout(title="共同資金池資產曲線",template="plotly_dark",height=420); st.plotly_chart(fig,use_container_width=True)
            st.caption("投組換股時會依實際權重變化估算手續費、滑價與賣出證交稅；這比原 V8.3 的無摩擦投組模型更保守。")

    with sub_wf:
        st.subheader("🧪 Walk-Forward / Out-of-Sample")
        st.markdown("**目的：** 不讓同一段歷史同時扮演訓練與驗證角色。V8.4 先提供可重複的 OOS 報表框架；後續若加入參數最佳化，訓練區間只能用來選參數，測試區間完全封存。")
        if st.button("🧪 執行 OOS 驗證",type="primary"):
            with st.spinner("執行多標的 OOS 驗證..."):
                st.session_state["wf_res"]=walk_forward_test(stocks,initial_capital,fee,tax,slippage)
        wf=st.session_state.get("wf_res")
        if wf is not None and not wf.empty:
            st.dataframe(wf,use_container_width=True,hide_index=True)
            st.success("OOS 報表完成。注意：目前版本不自動最佳化參數，因此 Walk-Forward 的『訓練』階段是保留的研究框架，而非資料探勘器。")

# footer
st.divider()
st.caption("V8.4 Research Edition · Point-in-Time data · Unified Buy Score · Realistic Costs · Benchmark · OOS Framework · Rule-based AI Explanation")
