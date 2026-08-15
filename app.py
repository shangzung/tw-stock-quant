# app.py
# 台股 V8.3 研究級量化決策 / 狀態保留與系統自走回測升級版 (含 API 錯誤捕捉與診斷)
# ------------------------------------------------------------
# 修正說明：
# 1. 更新 FinMind API 方法名稱 (taiwan_stock_daily, taiwan_stock_financial_statement)
# 2. 加入 _log_api_error 捕捉並記錄靜默錯誤
# 3. 側邊欄新增「API 診斷」面板
# ------------------------------------------------------------

import time
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
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
if "year_sim_res" not in st.session_state: st.session_state["year_sim_res"] = None
if "week_sim_res" not in st.session_state: st.session_state["week_sim_res"] = None
if "candidate_out" not in st.session_state: st.session_state["candidate_out"] = None
if "candidate_cands" not in st.session_state: st.session_state["candidate_cands"] = None
if "factor_rank" not in st.session_state: st.session_state["factor_rank"] = None
if "single_backtest_res" not in st.session_state: st.session_state["single_backtest_res"] = None
if "portfolio_backtest_res" not in st.session_state: st.session_state["portfolio_backtest_res"] = None

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
</style>
""", unsafe_allow_html=True)

st.title("🍏 台股 V8.3 研究級量化決策系統")
st.caption("基本面 × 估值 × 護城河 × 籌碼 × 技術 × Market Regime × ATR 風控 × 跨頁面快取")

# =========================
# 1. API 與 Token 鎖定按鈕
# =========================
api = DataLoader()

st.sidebar.header("🎛️ 系統與授權設定")

if "token_applied" not in st.session_state:
    st.session_state["token_applied"] = ""

user_token_input = st.sidebar.text_input("輸入 FinMind Token (選填)", type="password")

col_tok1, col_tok2 = st.sidebar.columns(2)
with col_tok1:
    if st.button("✔️ 確認 Token / 套用"):
        st.session_state["token_applied"] = user_token_input.strip()
        st.rerun()
with col_tok2:
    if st.button("🧹 清除快取"):
        st.cache_data.clear()
        # 同時清除畫面暫存
        st.session_state["market_scan_out"] = None
        st.sidebar.success("快取已清除，下次抓取會拿最新資料")

active_token = st.session_state["token_applied"]

if active_token:
    try:
        api.login_by_token(api_token=active_token)
        st.sidebar.success("✅ Token 已生效")
    except Exception:
        st.sidebar.error("Token 無效")
else:
    try:
        if "FINMIND_TOKEN" in st.secrets:
            api.login_by_token(api_token=st.secrets["FINMIND_TOKEN"])
            st.sidebar.info("使用系統隱藏 Token")
        else:
            st.sidebar.warning("⚠️ 使用免費額度")
    except Exception:
        st.sidebar.warning("⚠️ 使用免費額度")

API_SLEEP_SEC = 0.35

def throttle():
    time.sleep(API_SLEEP_SEC)

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
@st.cache_data(ttl=1800, show_spinner=False)
def get_daily(stock_id, days=500):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        # ✅ 使用正確的 API 名稱: taiwan_stock_daily
        df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start)
        throttle()
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy()
        for c in ["close", "open", "max", "min", "Trading_turnover"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Trading_Volume" in df.columns: df["volume"] = pd.to_numeric(df["Trading_Volume"], errors="coerce")
        elif "Trading_volume" in df.columns: df["volume"] = pd.to_numeric(df["Trading_volume"], errors="coerce")
        elif "volume" not in df.columns: df["volume"] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("taiwan_stock_daily", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_universe():
    try:
        info = api.taiwan_stock_info()
        throttle()
        if info is None or info.empty: return pd.DataFrame()
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
    except Exception as e:
        _log_api_error("taiwan_stock_info", "-", e)
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_revenue(stock_id, days=1000):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start)
        throttle()
        if df is None or df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["revenue", "revenue_year_on_year", "revenue_month_on_month"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("taiwan_stock_month_revenue", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_financial(stock_id, days=1500):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        # ✅ 使用正確的 API 名稱: taiwan_stock_financial_statement
        df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start)
        throttle()
        if df is None or df.empty: return pd.DataFrame()
        if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "value" in df.columns: df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("taiwan_stock_financial_statement", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_per_pbr(stock_id, days=1500):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = api.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start)
        throttle()
        if df is None or df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["PER", "PBR", "dividend_yield"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("taiwan_stock_per_pbr", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_institutional(stock_id, days=120):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        throttle()
        if df is None or df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["buy", "sell"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("taiwan_stock_institutional_investors", stock_id, e)
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_yahoo_taiex():
    try:
        mkt = yf.download("^TWII", period="2y", progress=False)
        if mkt.empty: return pd.DataFrame()
        if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
        mkt = mkt.reset_index()
        mkt.columns = [str(c).lower() for c in mkt.columns]
        mkt = mkt.rename(columns={"date": "date", "open": "open", "high": "max", "low": "min", "close": "close", "volume": "volume"})
        mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
        for c in ["close", "open", "max", "min", "volume"]:
            if c in mkt.columns: mkt[c] = pd.to_numeric(mkt[c], errors="coerce")
        return mkt.sort_values("date").reset_index(drop=True)
    except Exception as e:
        _log_api_error("yfinance ^TWII", "-", e)
        return pd.DataFrame()

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
def market_regime():
    mkt = get_yahoo_taiex()
    if mkt.empty:
        return {"regime": "UNKNOWN", "score": 50, "message": "無法取得 Yahoo 大盤資料，Market Filter 採中性。", "df": pd.DataFrame()}

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

def valuation_score(stock_id, fin):
    pe = get_per_pbr(stock_id, 1500)
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

def chip_score(stock_id):
    df = get_institutional(stock_id, 120)
    if df.empty or "buy" not in df.columns or "sell" not in df.columns: return 0, {}
    df["net"] = df["buy"] - df["sell"]

    foreign, trust = df, df
    if "name" in df.columns:
        foreign = df[df["name"].astype(str).str.contains("Foreign|外資", case=False, na=False)]
        trust = df[df["name"].astype(str).str.contains("Investment|投信", case=False, na=False)]

    def consecutive_buy(series):
        n = 0
        for x in reversed(series.tolist()):
            if x > 0: n += 1
            else: break
        return n

    f_buy = consecutive_buy(foreign.groupby("date")["net"].sum()) if not foreign.empty else 0
    t_buy = consecutive_buy(trust.groupby("date")["net"].sum()) if not trust.empty else 0
    total_net_10 = df.tail(10)["net"].sum()

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

@st.cache_data(ttl=1800, show_spinner=False)
def quick_prefilter_score(stock_id):
    daily = get_daily(stock_id, 260)
    if daily.empty or len(daily) < 120:
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

@st.cache_data(ttl=1800, show_spinner=False)
def calculate_stock(stock_id, regime_tag, regime_dict):
    try:
        daily = get_daily(stock_id, 600)
        if daily.empty or len(daily) < 120: return None
        daily = add_technical_indicators(daily)
        rev = get_revenue(stock_id, 1200)
        fin = get_financial(stock_id, 1800)

        fund = fundamental_score(fin, rev)
        moat, moat_detail = dynamic_moat_score(fin)
        val = valuation_score(stock_id, fin)
        chips, chip_detail = chip_score(stock_id)
        breakout, breakout_reasons = breakout_score(daily)

        x = daily.iloc[-1]
        price = safe_float(x["close"])

        technical = clamp((20 if price > safe_float(x["MA20"]) else 0) + (20 if safe_float(x["MA20"]) > safe_float(x["MA60"]) else 0) + (15 if safe_float(x["MACD"]) > safe_float(x["MACD_signal"]) else 0) + (10 if safe_float(x["K"]) > safe_float(x["D"]) else 0) + (10 if 50 <= safe_float(x["RSI"]) <= 70 else 0) + (10 if safe_float(x["ADX"]) >= 25 else 0) + (15 if safe_float(x["VOL_RATIO"]) >= 1.5 else 0))
        fund_pct = clamp(fund["score"] / 70 * 100)
        val_pct = clamp(val["score"] / 50 * 100)
        chips_pct = clamp(chips / 30 * 100)

        market_mult = 1.00 if regime_dict["regime"] == "BULL" else 0.85 if regime_dict["regime"] == "NEUTRAL" else 0.55 if regime_dict["regime"] == "BEAR" else 0.75
        raw = (fund_pct * 0.30 + moat * 0.10 + val_pct * 0.15 + chips_pct * 0.15 + technical * 0.15 + breakout * 0.15)
        final = clamp(raw * market_mult)

        distance_20_high = (price / safe_float(x["HIGH_20"]) - 1 if safe_float(x["HIGH_20"]) > 0 else np.nan)
        overheat = (safe_float(x["RET_20"]) > 0.25 or safe_float(x["RSI"]) > 78 or (not pd.isna(distance_20_high) and distance_20_high > 0.03))
        early_score = max(0, breakout - 15) if overheat else breakout
        if regime_dict["regime"] == "BEAR": early_score = max(0, early_score - 20)

        grade = "🔥 強勢起漲候選" if final >= 80 and early_score >= 75 and regime_dict["regime"] != "BEAR" else "🟢 起漲觀察" if final >= 70 and early_score >= 65 and regime_dict["regime"] != "BEAR" else "🟡 中性觀察" if final >= 60 else "🔴 排除"

        return {
            "股票代碼": stock_id, "現價": round(price, 2), "綜合分": round(final, 1), "起漲分": round(early_score, 1),
            "基本面": round(fund_pct, 1), "估值": round(val_pct, 1), "籌碼": round(chips_pct, 1), "技術": round(technical, 1),
            "量比": safe_float(x["VOL_RATIO"]), "RSI": safe_float(x["RSI"]), "ADX": safe_float(x["ADX"]), "PEG": val["PEG"],
            "過熱": "是" if overheat else "否", "評級": grade, "起漲理由": "、".join(breakout_reasons[:5]),
            "daily": daily, "fund": fund, "moat_detail": moat_detail
        }
    except Exception as e:
        _log_api_error("calculate_stock", stock_id, e)
        return None

# =========================
# 7. 回測引擎
# =========================
def backtest_single(stock_id, initial_capital, fee, tax, slippage, hold_days=10):
    df = get_daily(stock_id, 1500)
    if df.empty or len(df) < 250: return None
    df = add_technical_indicators(df).copy()

    cash, shares, entry_price, trades, equity_curve = initial_capital, 0, 0, [], []
    entry_i = 0
    for i in range(120, len(df) - 1):
        row, next_row = df.iloc[i], df.iloc[i + 1]
        price, atr = safe_float(row["close"]), safe_float(row["ATR"])
        if pd.isna(price) or pd.isna(atr) or atr <= 0:
            equity_curve.append(cash + shares * price)
            continue

        trend = (price > safe_float(row["MA20"]) and safe_float(row["MA20"]) > safe_float(row["MA60"]))
        momentum = (safe_float(row["MACD"]) > safe_float(row["MACD_signal"]) and 50 <= safe_float(row["RSI"]) <= 75)
        volume = safe_float(row["VOL_RATIO"]) >= 1.2

        if shares == 0 and trend and momentum and volume:
            buy_price = safe_float(next_row["open"], price) * (1 + slippage)
            shares = int(cash / (buy_price * (1 + fee)))
            if shares > 0:
                cost = shares * buy_price
                cash -= cost + (cost * fee)
                entry_price, entry_i = buy_price, i + 1

        elif shares > 0:
            stop_price, target_price = entry_price - 2 * atr, entry_price + 3 * atr
            low, high = safe_float(row["min"], price), safe_float(row["max"], price)
            exit_price, exit_reason = None, None

            if low <= stop_price: exit_price, exit_reason = stop_price * (1 - slippage), "STOP"
            elif high >= target_price: exit_price, exit_reason = target_price * (1 - slippage), "TARGET"
            elif i - entry_i >= hold_days: exit_price, exit_reason = safe_float(next_row["open"], price) * (1 - slippage), "TIME"

            if exit_price:
                gross = shares * exit_price
                cash += gross - (gross * fee) - (gross * tax)
                pnl = cash - initial_capital - sum(t["pnl"] for t in trades)
                trades.append({"entry": entry_price, "exit": exit_price, "pnl": pnl, "reason": exit_reason})
                shares, entry_price = 0, 0

        equity_curve.append(cash + shares * price)

    if not equity_curve: return None
    eq = pd.Series(equity_curve)
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = eq.iloc[-1] / initial_capital - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else np.nan
    wins = sum(1 for t in trades if t["pnl"] > 0)

    return {"stock": stock_id, "total_return": total_return, "mdd": mdd, "sharpe": sharpe, "sortino": sortino, "win_rate": (wins / len(trades) * 100 if trades else 0), "trades": len(trades), "equity": eq}

@st.cache_data(ttl=1800, show_spinner=False)
def _load_indexed_daily(stock_id, days=1500):
    df = add_technical_indicators(get_daily(stock_id, days))
    if df.empty: return df
    return df.set_index("date")

def portfolio_backtest(stocks, initial_capital, top_n, progress_cb=None):
    data = {}
    for idx, stock in enumerate(stocks):
        d = _load_indexed_daily(stock, 1500)
        if not d.empty: data[stock] = d
        if progress_cb: progress_cb((idx + 1) / len(stocks))

    if not data: return None

    prices = pd.concat({s: d["close"] for s, d in data.items()}, axis=1).sort_index().ffill().dropna(how="all")
    equity = pd.Series(index=prices.index, dtype=float)
    holdings = {}

    for i in range(120, len(prices)):
        date = prices.index[i]

        if i == 120 or i % 20 == 0:
            scores = {}
            for stock, d in data.items():
                sub = d[d.index <= date]
                if len(sub) < 120: continue
                x = sub.iloc[-1]
                score = (30 if safe_float(x["close"]) > safe_float(x["MA20"]) > safe_float(x["MA60"]) else 0) + (20 if safe_float(x["MACD"]) > safe_float(x["MACD_signal"]) else 0) + (15 if safe_float(x["RSI"]) >= 50 else 0) + (20 if safe_float(x["VOL_RATIO"]) >= 1.2 else 0) + (15 if safe_float(x["ADX"]) >= 20 else 0)
                scores[stock] = score
            ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
            holdings = {stock: 1 / len(ranked) for stock in ranked} if ranked else {}

        if i > 0:
            prev_date = prices.index[i - 1]
            portfolio_ret = 0.0
            for s, weight in holdings.items():
                if s not in prices.columns: continue
                prev_p = safe_float(prices.loc[prev_date, s])
                cur_p = safe_float(prices.loc[date, s])
                if prev_p and prev_p > 0 and not pd.isna(cur_p):
                    portfolio_ret += weight * (cur_p / prev_p - 1)
        else:
            portfolio_ret = 0.0

        equity.iloc[i] = initial_capital if i == 120 else equity.iloc[i - 1] * (1 + portfolio_ret)

    equity = equity.dropna()
    ret = equity.pct_change().dropna()
    return {"equity": equity, "return": equity.iloc[-1] / initial_capital - 1, "mdd": (equity / equity.cummax() - 1).min(), "sharpe": (ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan)}

# =========================
# 8. UI Sidebar
# =========================
st.sidebar.subheader("📌 自訂追蹤標的設定")
input_text = st.sidebar.text_area("輸入自選股代碼", value="2330\n2317\n2454", height=140)
stocks = clean_stock_list(input_text)

st.sidebar.subheader("⚙️ 回測與模擬參數")
initial_capital = st.sidebar.number_input("初始資金", min_value=10000, value=1_000_000, step=10000)
fee = st.sidebar.number_input("手續費", min_value=0.0, max_value=0.02, value=0.001425, format="%.6f")
tax = st.sidebar.number_input("證交稅", min_value=0.0, max_value=0.02, value=0.003, format="%.6f")
slippage = st.sidebar.number_input("滑價假設", min_value=0.0, max_value=0.02, value=0.0015, format="%.6f")
top_n = st.sidebar.slider("投組持股上限 (檔)", 1, 20, 5)

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

# ✅ API 診斷面板
with st.sidebar.expander("🩺 API 診斷 (資料抓不到時點開)"):
    if st.button("🔍 立即測試 API 連線 (以 2330 為例)"):
        st.session_state["api_errors"] = []
        checks = [
            ("台股日K", lambda: get_daily("2330", 60)),
            ("月營收", lambda: get_revenue("2330", 400)),
            ("財報", lambda: get_financial("2330", 600)),
            ("PER/PBR", lambda: get_per_pbr("2330", 400)),
            ("法人買賣", lambda: get_institutional("2330", 60)),
            ("Yahoo 大盤", lambda: get_yahoo_taiex()),
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
# 9. Tabs 顯示區
# =========================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 全市場飆股掃描", "⏳ 年份模擬", "🤖 一週實測(系統抓股)", "🔎 自選候選掃描", "📊 自選因子排名", "📉 單股回測", "💼 投組回測", "📖 說明"
])

# --- TAB 0: 全市場飆股掃描 ---
with tab0:
    st.subheader("🎯 全市場飆股掃描")
    st.markdown("用技術面粗篩找出動能最強的一批，再對這批候選跑完整分析。**（切換頁籤資料不會消失）**")

    universe_df = get_stock_universe()
    if universe_df.empty:
        st.error("無法取得全市場股票清單，可到側邊欄「API 診斷」檢查原因。")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: market_choice = st.selectbox("市場範圍", ["全部 (上市+上櫃)", "僅上市 (TWSE)", "僅上櫃 (TPEX)"])
        with c2: prefilter_limit = st.slider("初篩掃描檔數上限", 50, 1000, 300, step=50)
        with c3: top_k = st.slider("進入完整分析的候選數", 5, 60, 25, step=5)

        if market_choice == "僅上市 (TWSE)": uni = universe_df[universe_df["type"].str.lower() == "twse"]
        elif market_choice == "僅上櫃 (TPEX)": uni = universe_df[universe_df["type"].str.lower() == "tpex"]
        else: uni = universe_df

        st.caption(f"預估消耗約 {min(prefilter_limit, len(uni)) + top_k * 5} 次 API 額度。")

        if st.button("🚀 開始全市場掃描", type="primary"):
            scan_list = uni["stock_id"].tolist()[:prefilter_limit]
            pre_rows = []
            progress1 = st.progress(0)
            status1 = st.empty()
            for i, sid in enumerate(scan_list):
                status1.text(f"初篩中... {sid} ({i+1}/{len(scan_list)})")
                r = quick_prefilter_score(sid)
                if r: pre_rows.append(r)
                progress1.progress((i + 1) / len(scan_list))
            status1.empty()

            if pre_rows:
                pre_df = pd.DataFrame(pre_rows).sort_values("初篩分", ascending=False)
                shortlist = pre_df.head(top_k)["股票代碼"].tolist()

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
                    out = pd.DataFrame(final_rows).sort_values(["起漲分", "綜合分"], ascending=False)
                    name_map = universe_df.set_index("stock_id")["stock_name"].to_dict() if "stock_name" in universe_df.columns else {}
                    out.insert(1, "名稱", out["股票代碼"].map(name_map).fillna(""))
                    candidates = out[(out["起漲分"] >= 65) & (out["綜合分"] >= 60) & (out["過熱"] == "否")]

                    # 儲存到 session_state
                    st.session_state["market_scan_out"] = out
                    st.session_state["market_scan_candidates"] = candidates
                else:
                    st.error("完整分析階段沒有取得有效資料。請檢查 API 診斷紀錄。")
            else:
                st.error("初篩沒有取得任何有效資料。請檢查 API 診斷紀錄。")

        # 顯示快取內的資料
        if st.session_state.get("market_scan_out") is not None:
            out_df = st.session_state["market_scan_out"]
            cands_df = st.session_state["market_scan_candidates"]
            show_cols = ["股票代碼", "現價", "綜合分", "起漲分", "基本面", "估值", "籌碼", "技術", "量比", "RSI", "ADX", "PEG", "評級", "過熱", "起漲理由"]

            st.subheader("📋 全市場掃描結果 (依起漲分排序)")
            st.dataframe(out_df[["名稱"] + show_cols] if "名稱" in out_df.columns else out_df[show_cols], use_container_width=True, hide_index=True)

            st.subheader("🔥 真正飆股候選池")
            if cands_df.empty:
                st.info("這次掃描沒有股票同時通過起漲、綜合分與過熱過濾。")
            else:
                cols2 = ["名稱", "股票代碼", "現價", "綜合分", "起漲分", "評級", "起漲理由"] if "名稱" in cands_df.columns else ["股票代碼", "現價", "綜合分", "起漲分", "評級", "起漲理由"]
                st.dataframe(cands_df[cols2], use_container_width=True, hide_index=True)

# --- TAB 1: 年份歷史模擬 ---
with tab1:
    st.subheader("⏳ 指定年份歷史模擬")
    target_year = st.selectbox("📅 選擇模擬年份", [2025, 2024, 2023, 2022])

    if st.button("🚀 開始年份模擬", type="primary"):
        if not stocks: st.warning("請先在側邊欄輸入股票代碼。")
        else:
            all_trades = []
            progress = st.progress(0)
            status = st.empty()
            for si, s in enumerate(stocks):
                status.text(f"正在模擬 {s} 於 {target_year} 年的表現...")
                try:
                    df = add_technical_indicators(get_daily(s, 1500))
                    df_year = df[df['date'].dt.year == target_year].copy()
                    if df_year.empty:
                        progress.progress((si + 1) / len(stocks))
                        continue

                    holding, entry_price, entry_date, stop_loss, take_profit = False, 0.0, None, 0.0, 0.0

                    for i in range(len(df_year)):
                        row = df_year.iloc[i]
                        if holding:
                            exit_price, reason = 0, ""
                            if row['min'] <= stop_loss: exit_price, reason = stop_loss * (1 - slippage), "🔴 停損"
                            elif row['max'] >= take_profit: exit_price, reason = take_profit * (1 - slippage), "🟢 停利"
                            elif i == len(df_year) - 1: exit_price, reason = row['close'] * (1 - slippage), "💼 年底結算"

                            if exit_price > 0 and entry_price > 0:
                                pnl_pct = (exit_price / entry_price * (1 - fee - tax) * (1 - fee) - 1) * 100
                                all_trades.append({
                                    "代碼": s,
                                    "買進日": entry_date.strftime("%m-%d") if entry_date is not None else "-",
                                    "賣出日": row['date'].strftime("%m-%d"),
                                    "獲利(%)": round(pnl_pct, 2),
                                    "原因": reason
                                })
                                holding = False

                        if not holding and i < len(df_year) - 1:
                            if safe_float(row["close"]) > safe_float(row["MA20"]) > safe_float(row["MA60"]) and safe_float(row["VOL_RATIO"]) >= 1.5 and safe_float(row["MACD"]) > safe_float(row["MACD_signal"]):
                                holding = True
                                entry_price = row['close'] * (1 + slippage) * (1 + fee)
                                entry_date = row['date']
                                atr = safe_float(row['ATR'])
                                stop_loss = entry_price - (2 * atr) if not pd.isna(atr) else entry_price * 0.90
                                take_profit = entry_price + (3 * atr) if not pd.isna(atr) else entry_price * 1.15
                except Exception:
                    pass
                progress.progress((si + 1) / len(stocks))

            status.empty()
            st.session_state["year_sim_res"] = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    if st.session_state.get("year_sim_res") is not None:
        res_df = st.session_state["year_sim_res"]
        if not res_df.empty:
            st.dataframe(style_pnl(res_df), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("該年平均單次獲利", f'{res_df["獲利(%)"].mean():.2f} %')
            c2.metric("系統進場勝率", f'{(res_df["獲利(%)"] > 0).mean() * 100:.1f} %')
        else:
            st.info("查無進場訊號。")

# --- TAB 2: 一週實測 (系統自動抓股) ---
with tab2:
    st.subheader("🤖 系統自選一週實測")
    st.markdown(
        "不再只針對手動輸入的標的。此功能將從**全市場隨機抽樣**，"
        "自動找出在「約一週前（5~6個交易日）」真正觸發技術面突破訊號的股票，"
        "模擬當時買進並抱到今天的真實獲利與勝率，藉此驗證系統的選股準確度。"
    )
    test_limit = st.slider("隨機抽樣檔數 (避免耗盡 API)", 50, 800, 300, step=50)

    if st.button("🎯 執行全市場一週實測", type="primary"):
        uni = get_stock_universe()
        if uni.empty:
            st.error("無法取得市場清單。")
        else:
            test_list = uni["stock_id"].sample(n=min(test_limit, len(uni)), random_state=42).tolist()
            sim_results = []
            progress = st.progress(0)
            status = st.empty()

            for si, s in enumerate(test_list):
                status.text(f"回測抽樣標的中... {s} ({si+1}/{len(test_list)})")
                try:
                    df = add_technical_indicators(get_daily(s, 100))
                    if not df.empty and len(df) >= 20:
                        # 抓取大約 5~6 個交易日前的狀態
                        target_idx = max(0, len(df) - 6)
                        entry_row = df.iloc[target_idx]

                        # 判斷 5 天前是否觸發進場訊號 (與系統邏輯一致)
                        signal = (safe_float(entry_row["close"]) > safe_float(entry_row["MA20"]) > safe_float(entry_row["MA60"]) and
                                  safe_float(entry_row["VOL_RATIO"]) >= 1.2 and
                                  safe_float(entry_row["MACD"]) > safe_float(entry_row["MACD_signal"]))

                        if signal:
                            entry_price = safe_float(entry_row["close"])
                            atr = safe_float(entry_row["ATR"])
                            buy_cost = entry_price * (1 + slippage) * (1 + fee)
                            stop_loss = entry_price - (2 * atr) if not pd.isna(atr) else np.nan
                            take_profit = entry_price + (3 * atr) if not pd.isna(atr) else np.nan

                            status_label = "⏳ 持有中"
                            sell_revenue = safe_float(df.iloc[-1]["close"]) * (1 - slippage) * (1 - fee - tax)

                            # 模擬這 5 天的過程，是否提前打到停損/停利
                            period_df = df.iloc[target_idx+1:]
                            for _, r in period_df.iterrows():
                                if not pd.isna(stop_loss) and safe_float(r["min"]) <= stop_loss:
                                    status_label = "🔴 停損出場"
                                    sell_revenue = stop_loss * (1 - slippage) * (1 - fee - tax)
                                    break
                                elif not pd.isna(take_profit) and safe_float(r["max"]) >= take_profit:
                                    status_label = "🟢 停利出場"
                                    sell_revenue = take_profit * (1 - slippage) * (1 - fee - tax)
                                    break

                            pnl_pct = (sell_revenue / buy_cost - 1) * 100
                            name_map = uni.set_index("stock_id")["stock_name"].to_dict() if "stock_name" in uni.columns else {}
                            sim_results.append({
                                "代碼": s,
                                "名稱": name_map.get(s, ""),
                                "買進日 (約一週前)": entry_row["date"].strftime("%m-%d"),
                                "狀態": status_label,
                                "獲利(%)": round(pnl_pct, 2)
                            })
                except Exception:
                    pass
                progress.progress((si + 1) / len(test_list))

            status.empty()
            st.session_state["week_sim_res"] = pd.DataFrame(sim_results) if sim_results else pd.DataFrame()

    if st.session_state.get("week_sim_res") is not None:
        res_df = st.session_state["week_sim_res"]
        if not res_df.empty:
            st.dataframe(style_pnl(res_df), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.metric("一周實測平均獲利", f'{res_df["獲利(%)"].mean():.2f} %')
            c2.metric("一周實測勝率", f'{(res_df["獲利(%)"] > 0).mean() * 100:.1f} %')
        else:
            st.info("這批抽樣的股票在 5 天前皆無觸發進場訊號。")

# --- TAB 3: 候選掃描 ---
with tab3:
    st.subheader("🔥 側邊欄自選：起漲前候選掃描器")
    if st.button("🚀 掃描自選候選", type="primary"):
        if not stocks: st.error("請先在側邊欄輸入股票代碼。")
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
                out = pd.DataFrame(rows).sort_values(["起漲分", "綜合分"], ascending=False)
                cands = out[(out["起漲分"] >= 65) & (out["綜合分"] >= 60) & (out["過熱"] == "否")]
                st.session_state["candidate_out"] = out
                st.session_state["candidate_cands"] = cands
            else:
                st.error("沒有取得有效資料。")

    if st.session_state.get("candidate_out") is not None:
        out = st.session_state["candidate_out"]
        cands = st.session_state["candidate_cands"]
        show_cols = ["股票代碼", "現價", "綜合分", "起漲分", "基本面", "估值", "籌碼", "技術", "量比", "RSI", "ADX", "PEG", "評級", "過熱", "起漲理由"]
        st.dataframe(out[show_cols], use_container_width=True, hide_index=True)
        st.subheader("🎯 真正候選池")
        if cands.empty: st.info("目前自選股中沒有同時通過起漲、綜合分與過熱過濾的標的。")
        else: st.dataframe(cands[["股票代碼", "現價", "綜合分", "起漲分", "評級", "起漲理由"]], use_container_width=True, hide_index=True)

# --- TAB 4: 因子排名 ---
with tab4:
    st.subheader("📊 側邊欄自選：多因子排名")
    if st.button("📊 執行完整排名"):
        if not stocks: st.error("請先輸入股票代碼。")
        else:
            rows = []
            progress = st.progress(0)
            for i, stock in enumerate(stocks):
                result = calculate_stock(stock, regime["regime"], regime)
                if result: rows.append(result)
                progress.progress((i + 1) / len(stocks))

            if rows:
                st.session_state["factor_rank"] = pd.DataFrame(rows).sort_values(["綜合分", "基本面", "技術"], ascending=False)
            else:
                st.error("沒有取得有效資料。")

    if st.session_state.get("factor_rank") is not None:
        st.dataframe(st.session_state["factor_rank"].drop(columns=["daily", "fund", "moat_detail"], errors="ignore"), use_container_width=True, hide_index=True)

# --- TAB 5: 單股回測 ---
with tab5:
    st.subheader("📉 單股歷史回測")
    if stocks:
        selected = st.selectbox("選擇股票", stocks)
        if st.button("▶️ 執行單股回測"):
            with st.spinner("回測中..."):
                result = backtest_single(selected, initial_capital, fee, tax, slippage)
                st.session_state["single_backtest_res"] = result

        if st.session_state.get("single_backtest_res") is not None:
            result = st.session_state["single_backtest_res"]
            if result is None:
                st.error("資料不足，無法完成回測。")
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("策略報酬", f"{result['total_return']*100:.2f}%")
                c2.metric("最大回撤", f"{result['mdd']*100:.2f}%")
                c3.metric("Sharpe", f"{result['sharpe']:.2f}" if not pd.isna(result["sharpe"]) else "N/A")
                c4.metric("Sortino", f"{result['sortino']:.2f}" if not pd.isna(result["sortino"]) else "N/A")
                c5.metric("勝率", f"{result['win_rate']:.1f}%")

                eq = result["equity"]
                fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy"))
                fig.update_layout(title=f"{result['stock']} 策略資產曲線", xaxis_title="日期", yaxis_title="資產", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else: st.info("請先輸入股票代碼。")

# --- TAB 6: 投組回測 ---
with tab6:
    st.subheader("💼 自選多股票共同資金池回測")
    if st.button("💼 執行投資組合回測"):
        if len(stocks) < 2: st.error("至少輸入 2 檔股票。")
        else:
            progress = st.progress(0)
            status = st.empty()
            status.text("正在下載並計算各股票技術指標...")
            result = portfolio_backtest(stocks, initial_capital, top_n, progress_cb=progress.progress)
            status.empty()
            st.session_state["portfolio_backtest_res"] = result

    if st.session_state.get("portfolio_backtest_res") is not None:
        result = st.session_state["portfolio_backtest_res"]
        if result is None:
            st.error("無法建立投資組合，請檢查資料。")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("投組報酬", f"{result['return']*100:.2f}%")
            c2.metric("最大回撤", f"{result['mdd']*100:.2f}%")
            c3.metric("Sharpe", f"{result['sharpe']:.2f}" if not pd.isna(result["sharpe"]) else "N/A")

            fig = go.Figure(go.Scatter(x=result["equity"].index, y=result["equity"].values, mode="lines", name="Portfolio"))
            fig.update_layout(title="共同資金池投資組合資產曲線", xaxis_title="日期", yaxis_title="資產", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 7: 說明 ---
with tab7:
    st.subheader("🛡️ 系統說明")
    st.markdown("""
    **V8.3 新增功能重點：**
    * **💾 跨頁籤狀態保留 (Session State)**：現在你執行完「全市場掃描」或「回測」，點擊別的頁籤再回來，**資料再也不會消失**，不再需要浪費時間跟額度重新跑。
    * **🤖 真實一週實測升級**：改掉原本「只能測自己手動輸入股票」的邏輯。現在系統會**自動從全市場抽樣**，倒退回一週前檢查「當時有沒有觸發進場訊號」，如果有，就幫你結算抱到今天的獲利。這才能真正體現量化系統本身選股的勝率。
    * **🩺 內建 API 診斷與紀錄**：側邊欄內建連線測試面板，出錯時不必瞎猜，點開馬上抓漏！
    """)