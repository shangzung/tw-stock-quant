# app.py
# 台股 V10.0 Smart Real-Time Scanner：盤中即時 + 盤後深度 + PIT 回測 + 統一買進分 / Point-in-Time 回測 + 統一買進分 + Benchmark + Walk-Forward (含 API 錯誤捕捉與診斷)
# ------------------------------------------------------------
# 修正說明：
# 1. 更新 FinMind API 方法名稱 (taiwan_stock_daily, taiwan_stock_financial_statement)
# 2. 加入 _log_api_error 捕捉並記錄靜默錯誤
# 3. 側邊欄新增「API 診斷」面板
# 4. 統一 market_prefilter 與 calculate_stock_at 的 get_daily 天數參數為 600，避免重複消耗 API 額度。
# ------------------------------------------------------------

import time
import math
import json
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# -1. 本機持久化：Token / 掃描結果 / 庫存清單
#     全部存在 app.py 同一層的 .quant_compass_cache 資料夾，純本機檔案，
#     不會上傳到任何地方；換電腦或砍掉這個資料夾就等於全部重來。
# =========================
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()
CACHE_DIR = APP_DIR / ".quant_compass_cache"
CACHE_DIR.mkdir(exist_ok=True)
TOKEN_FILE = CACHE_DIR / "finmind_token.json"
SCAN_CACHE_FILE = CACHE_DIR / "last_scan.pkl"
HOLDINGS_FILE = CACHE_DIR / "holdings.json"




# =========================
# V10.0 Strategy Validation Engine
# =========================
RESEARCH_LOG_FILE = CACHE_DIR / "research_signal_log.jsonl"
CALIBRATION_CACHE_FILE = CACHE_DIR / "calibration_result.pkl"


def append_research_snapshot(df, saved_at=None, market_regime=None, market_score=None):
    """跨 session 保存每日訊號快照；只保存校準所需的輕量欄位。"""
    if df is None or df.empty:
        return False
    try:
        ts = saved_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        keep = ["股票代碼", "名稱", "買進分", "優先級", "決策", "狀態", "風險", "資料品質", "現價", "日期"]
        rows = []
        for _, r in df.iterrows():
            rec = {"snapshot_at": ts, "market_regime": market_regime, "market_score": market_score}
            for k in keep:
                v = r.get(k)
                if isinstance(v, (np.integer,)): v = int(v)
                elif isinstance(v, (np.floating,)): v = float(v)
                elif pd.isna(v) if not isinstance(v, (list, dict)) else False: v = None
                rec[k] = v
            rows.append(json.dumps(rec, ensure_ascii=False, default=str))
        with RESEARCH_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        return True
    except Exception as e:
        _log_api_error("append_research_snapshot", "", e)
        return False


def load_research_log():
    if not RESEARCH_LOG_FILE.exists():
        return pd.DataFrame()
    rows = []
    try:
        with RESEARCH_LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _score_bucket(x):
    try: x = float(x)
    except Exception: return "N/A"
    if x >= 90: return "90+"
    if x >= 85: return "85–89"
    if x >= 80: return "80–84"
    if x >= 75: return "75–79"
    if x >= 70: return "70–74"
    if x >= 65: return "65–69"
    if x >= 60: return "60–64"
    return "<60"


def build_forward_calibration(log_df, max_samples=500, forward_days=(5, 10, 20)):
    """將歷史訊號與訊號後第一個交易日及其後 N 日收盤連接，建立前瞻報酬校準。"""
    if log_df is None or log_df.empty or "股票代碼" not in log_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = log_df.copy()
    d["signal_date"] = pd.to_datetime(d.get("日期"), errors="coerce")
    if d["signal_date"].isna().all() and "snapshot_at" in d.columns:
        d["signal_date"] = pd.to_datetime(d["snapshot_at"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["signal_date"])
    d = d.sort_values("snapshot_at" if "snapshot_at" in d.columns else "signal_date")
    d = d.drop_duplicates(["股票代碼", "signal_date"], keep="last")
    if len(d) > max_samples:
        d = d.tail(max_samples)
    cache = {}
    rows = []
    for _, r in d.iterrows():
        sid = str(r.get("股票代碼", "")).strip()
        if not sid: continue
        if sid not in cache:
            cache[sid] = get_daily(sid, 1500)
        px = cache[sid]
        if px is None or px.empty or "date" not in px.columns or "close" not in px.columns:
            continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px["close"] = pd.to_numeric(px["close"], errors="coerce")
        px = px.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        sig = pd.Timestamp(r["signal_date"]).normalize()
        future = px[px["date"] > sig]
        if future.empty: continue
        entry = float(future.iloc[0]["close"])
        rec = {"股票代碼": sid, "訊號日": sig.strftime("%Y-%m-%d"),
               "買進分": safe_float(r.get("買進分")), "分數區間": _score_bucket(r.get("買進分")),
               "市場環境": r.get("market_regime"), "風險": r.get("風險"), "entry_close": entry}
        for n in forward_days:
            rec[f"{n}D報酬"] = (float(future.iloc[n-1]["close"]) / entry - 1) if len(future) >= n else np.nan
        rows.append(rec)
    detail = pd.DataFrame(rows)
    if detail.empty: return detail, pd.DataFrame()
    summary = []
    for bucket in ["<60", "60–64", "65–69", "70–74", "75–79", "80–84", "85–89", "90+"]:
        g = detail[detail["分數區間"] == bucket]
        if g.empty: continue
        rec = {"分數區間": bucket, "樣本數": len(g)}
        for n in forward_days:
            v = pd.to_numeric(g[f"{n}D報酬"], errors="coerce").dropna()
            rec[f"{n}D勝率"] = float((v > 0).mean() * 100) if len(v) else np.nan
            rec[f"{n}D平均報酬"] = float(v.mean() * 100) if len(v) else np.nan
            rec[f"{n}D中位數"] = float(v.median() * 100) if len(v) else np.nan
        summary.append(rec)
    return detail, pd.DataFrame(summary)


def strategy_drift_report(detail, horizon="10D報酬", recent_n=20, baseline_n=60):
    if detail is None or detail.empty or horizon not in detail.columns:
        return {"status": "INSUFFICIENT", "message": "尚無可用前瞻報酬資料。"}
    d = detail.sort_values("訊號日")
    v = pd.to_numeric(d[horizon], errors="coerce").dropna()
    if len(v) < recent_n:
        return {"status": "INSUFFICIENT", "message": f"樣本不足：至少需要 {recent_n} 筆有效 {horizon}。"}
    recent = v.tail(recent_n)
    base = v.tail(max(baseline_n, len(v)))
    rw, bw = float((recent > 0).mean()), float((base > 0).mean())
    rr, br = float(recent.mean()), float(base.mean())
    wd, rd = rw - bw, rr - br
    if wd <= -0.15 or rd <= -0.05: status = "DRIFT"
    elif wd <= -0.08 or rd <= -0.02: status = "WATCH"
    else: status = "STABLE"
    msg = {"DRIFT": "🔴 最近策略表現明顯低於歷史基準，建議降低風險曝險並檢查因子失效。",
           "WATCH": "🟡 最近策略表現弱於歷史基準，進入觀察區。",
           "STABLE": "🟢 最近策略表現仍在歷史合理範圍。"}[status]
    return {"status": status, "recent_win": rw, "baseline_win": bw, "win_delta": wd,
            "recent_return": rr, "baseline_return": br, "return_delta": rd,
            "recent_n": len(recent), "baseline_n": len(base), "message": msg}


def calibration_reliability(n):
    n = int(n or 0)
    if n >= 200: return "高"
    if n >= 80: return "中"
    if n >= 30: return "低"
    return "極低"

def load_saved_token():
    try:
        if TOKEN_FILE.exists():
            return json.loads(TOKEN_FILE.read_text(encoding="utf-8")).get("token", "")
    except Exception:
        pass
    return ""


def save_token_to_disk(token):
    try:
        TOKEN_FILE.write_text(json.dumps({"token": token}, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


def clear_saved_token():
    try:
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
        return True
    except Exception:
        return False


def load_saved_scan():
    """開啟 App 時，把上次留下來的掃描結果讀回來，直到下一次執行盤後深度掃描才會被覆蓋掉。"""
    try:
        if SCAN_CACHE_FILE.exists():
            return pd.read_pickle(SCAN_CACHE_FILE)
    except Exception:
        pass
    return None


def save_scan_to_disk(payload: dict):
    try:
        pd.to_pickle(payload, SCAN_CACHE_FILE)
        return True
    except Exception:
        return False


def clear_saved_scan():
    try:
        if SCAN_CACHE_FILE.exists():
            SCAN_CACHE_FILE.unlink()
        return True
    except Exception:
        return False


def _coerce_holdings_dtypes(df):
    """確保庫存表格的欄位型別跟 column_config 一致（代碼=文字、股數/成本=數字），
    不然 st.data_editor 在型別對不上時會直接丟出 StreamlitAPIException 讓整頁掛掉。"""
    if df is None or df.empty:
        return pd.DataFrame({
            "股票代碼": pd.Series(dtype="object"),
            "持有股數": pd.Series(dtype="float64"),
            "持有成本": pd.Series(dtype="float64"),
        })
    out = df.copy()
    if "股票代碼" not in out.columns: out["股票代碼"] = pd.Series(dtype="object")
    if "持有股數" not in out.columns: out["持有股數"] = pd.Series(dtype="float64")
    if "持有成本" not in out.columns: out["持有成本"] = pd.Series(dtype="float64")
    out["股票代碼"] = out["股票代碼"].astype("object")
    out["持有股數"] = pd.to_numeric(out["持有股數"], errors="coerce")
    out["持有成本"] = pd.to_numeric(out["持有成本"], errors="coerce")
    return out[["股票代碼", "持有股數", "持有成本"]]


def load_saved_holdings():
    try:
        if HOLDINGS_FILE.exists():
            data = json.loads(HOLDINGS_FILE.read_text(encoding="utf-8"))
            if data:
                return _coerce_holdings_dtypes(pd.DataFrame(data))
    except Exception:
        pass
    return _coerce_holdings_dtypes(None)


def save_holdings_to_disk(df):
    try:
        HOLDINGS_FILE.write_text(df.to_json(orient="records", force_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


def holdings_rows_from_df(df):
    """把持久化用的 DataFrame 轉成自訂列編輯器用的 list-of-dict（每列一個穩定 id）。"""
    rows = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            code = r.get("股票代碼")
            code = "" if pd.isna(code) else str(code).strip()
            if not code:
                continue
            shares = r.get("持有股數")
            cost = r.get("持有成本")
            rows.append({
                "id": str(uuid.uuid4()),
                "code": code,
                "shares": 0.0 if pd.isna(shares) else float(shares),
                "cost": 0.0 if pd.isna(cost) else float(cost),
            })
    if not rows:
        rows.append({"id": str(uuid.uuid4()), "code": "", "shares": 0.0, "cost": 0.0})
    return rows


def holdings_df_from_rows(rows):
    """把自訂列編輯器的資料轉回原本程式碼共用的 DataFrame 格式。"""
    data = [{"股票代碼": (r.get("code") or "").strip(), "持有股數": r.get("shares", 0) or 0,
             "持有成本": r.get("cost", 0.0) or 0.0} for r in rows]
    return _coerce_holdings_dtypes(pd.DataFrame(data)) if data else _coerce_holdings_dtypes(None)

# =========================
# 0. Streamlit & Mac CSS & 狀態初始化
# =========================
st.set_page_config(
    page_title="台股量化羅盤 · Quant Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化所有分頁的 Session State，確保切換頁籤不會遺失資料
if "market_scan_out" not in st.session_state: st.session_state["market_scan_out"] = None
if "market_scan_candidates" not in st.session_state: st.session_state["market_scan_candidates"] = None
if "market_scan_top5" not in st.session_state: st.session_state["market_scan_top5"] = None
if "market_scan_saved_at" not in st.session_state: st.session_state["market_scan_saved_at"] = None
if st.session_state["market_scan_out"] is None:
    # App 重開後，先把上次留下的掃描結果讀回來，直到下一次執行盤後深度掃描才會被洗掉。
    _saved_scan = load_saved_scan()
    if _saved_scan:
        st.session_state["market_scan_out"] = _saved_scan.get("out")
        st.session_state["market_scan_candidates"] = _saved_scan.get("candidates")
        st.session_state["market_scan_top5"] = _saved_scan.get("top5")
        st.session_state["market_scan_saved_at"] = _saved_scan.get("saved_at")
if "year_sim_res" not in st.session_state: st.session_state["year_sim_res"] = None
if "week_sim_res" not in st.session_state: st.session_state["week_sim_res"] = None
if "candidate_out" not in st.session_state: st.session_state["candidate_out"] = None
if "candidate_cands" not in st.session_state: st.session_state["candidate_cands"] = None
if "single_backtest_res" not in st.session_state: st.session_state["single_backtest_res"] = None
if "portfolio_backtest_res" not in st.session_state: st.session_state["portfolio_backtest_res"] = None
if "wf_res" not in st.session_state: st.session_state["wf_res"] = None
if "stock_lookup_res" not in st.session_state: st.session_state["stock_lookup_res"] = None
if "watchlist_codes" not in st.session_state:
    st.session_state["watchlist_codes"] = ["2330", "5351", "3481", "2317", "2454"]
if "token_applied" not in st.session_state:
    # App 重開後，先把上次儲存的 Token 讀回來，不用每次都重新輸入。
    st.session_state["token_applied"] = load_saved_token()
if "holdings_rows" not in st.session_state:
    st.session_state["holdings_rows"] = holdings_rows_from_df(load_saved_holdings())
if "holdings_health_res" not in st.session_state:
    st.session_state["holdings_health_res"] = None

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
        /* 台股標準：紅漲、綠跌；與決策/風險顏色分離 */
        --tw-up: #ff453a;
        --tw-down: #30d158;
        --tw-flat: #a1a1a6;
        --decision-buy: #30d158;
        --decision-watch: #ffd60a;
        --decision-stop: #ff453a;
        /* data_editor / dataframe 是 canvas 畫的 glide-data-grid，顏色主要靠 .streamlit/config.toml
           的 [theme] 設定；這裡的變數是給有支援讀 CSS 變數版本的備援，不是主要修法 */
        --gdg-bg-cell: #1c1c1e;
        --gdg-bg-cell-medium: #2c2c2e;
        --gdg-bg-header: #2c2c2e;
        --gdg-bg-header-has-focus: #3a3a3c;
        --gdg-border-color: #3a3a3c;
        --gdg-text-dark: #f5f5f7;
        --gdg-text-medium: #a1a1a6;
        --gdg-text-light: #a1a1a6;
        --gdg-accent-color: #0a84ff;
        --gdg-accent-fg: #ffffff;
        --gdg-bg-bubble: #2c2c2e;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .block-container { padding-top: 1.6rem !important; max-width: 1240px !important; }

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

    /* ── Hero 主標題區：品牌識別 + 一句話說明這是做什麼的 ── */
    .hero-banner {
        position: relative;
        margin: 4px 0 22px 0;
        padding: 30px 34px;
        border-radius: 20px;
        overflow: hidden;
        background:
            radial-gradient(circle at 12% -10%, rgba(10,132,255,0.35), transparent 55%),
            radial-gradient(circle at 100% 0%, rgba(48,209,88,0.18), transparent 45%),
            linear-gradient(180deg, #1c1c1f 0%, #131315 100%);
        border: 1px solid var(--border-c);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .hero-topline { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }
    .hero-live { font-size:11px; font-weight:700; color:var(--accent-green); letter-spacing:.04em; }
    .terminal-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:0 0 18px 0; }
    .terminal-card { padding:14px 16px; border:1px solid var(--border-c); border-radius:12px; background:linear-gradient(180deg,#202024 0%,#171719 100%); }
    .terminal-card .tc-label { font-size:10px; letter-spacing:.09em; color:var(--text-sub); font-weight:700; }
    .terminal-card .tc-value { margin-top:6px; font-size:24px; line-height:1.1; font-weight:800; font-variant-numeric:tabular-nums; }
    .terminal-card .tc-unit { font-size:12px; color:var(--text-sub); margin-left:3px; }
    .terminal-card .tc-sub { margin-top:6px; font-size:11.5px; color:var(--text-sub); line-height:1.4; }
    @media (max-width: 900px) { .terminal-grid { grid-template-columns:1fr 1fr; } }
    @media (max-width: 560px) { .terminal-grid { grid-template-columns:1fr; } }

    .hero-banner .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: var(--accent-blue);
        text-transform: uppercase;
        background: rgba(10,132,255,0.12);
        border: 1px solid rgba(10,132,255,0.35);
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 14px;
    }
    .hero-banner h1 {
        margin: 0 0 8px 0 !important;
        font-size: 34px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.15 !important;
        background: linear-gradient(90deg, #ffffff 0%, #c8c8cf 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-banner .hero-sub {
        font-size: 15px;
        color: var(--text-sub);
        line-height: 1.6;
        max-width: 720px;
    }
    .hero-banner .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    .hero-banner .hero-tag {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-main);
        background: var(--bg-card-2);
        border: 1px solid var(--border-c);
        padding: 5px 11px;
        border-radius: 999px;
    }

    /* ── 分頁 Tabs：底線動畫、字重層次，更接近原生 macOS 分段控制項 ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--border-c);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px !important;
        padding: 10px 16px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text-main) !important; background-color: var(--bg-card-2) !important; }
    .stTabs [aria-selected="true"] { font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent-blue) !important; height: 2.5px !important; }

    /* 卡片式元件 hover 微浮起，呼應 macOS 介面互動細節 */
    .pick-card, .regime-card, [data-testid="stExpander"] {
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .pick-card:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,0,0,0.28); }

    hr, [data-testid="stDivider"] { border-color: var(--border-c) !important; opacity: 0.6; }

    /* ── 庫存健康檢查：風險偏好卡片、信心分數條、統計卡、持股結果卡 ── */
    .risk-profile-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 6px 0 4px 0;
    }
    .risk-profile-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
    }
    .risk-profile-card.active {
        border-color: var(--accent-blue);
        background: linear-gradient(180deg, rgba(10,132,255,0.14) 0%, var(--bg-card-2) 100%);
        box-shadow: 0 0 0 1px var(--accent-blue) inset;
    }
    .risk-profile-card .rp-title { font-size: 14.5px; font-weight: 700; margin-bottom: 4px; display:flex; align-items:center; gap:6px;}
    .risk-profile-card .rp-desc { font-size: 12px; color: var(--text-sub); line-height: 1.5; }
    .risk-profile-card .rp-num { font-size: 11.5px; color: var(--accent-blue); font-weight: 600; margin-top: 6px; }

    .stat-chip-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 4px 0 14px 0; }
    .stat-chip {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
        text-align: left;
    }
    .stat-chip .sc-label { font-size: 12px; color: var(--text-sub); font-weight: 600; margin-bottom: 6px; }
    .stat-chip .sc-value { font-size: 24px; font-weight: 700; line-height: 1; }

    .holding-card {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
        margin-bottom: 12px;
    }
    .holding-card .hc-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
    .holding-card .hc-name { font-size: 17px; font-weight: 700; }
    .holding-card .hc-badge {
        display:inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 700;
        margin-left: 8px; border: 1px solid transparent;
    }
    .holding-card .hc-meta { font-size: 12.5px; color: var(--text-sub); display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }
    .holding-card .hc-meta b { color: var(--text-main); font-weight: 600; }

    .confidence-row { display:flex; align-items:center; gap:10px; margin-top: 12px; }
    .confidence-label { font-size: 12px; color: var(--text-sub); font-weight:600; white-space:nowrap; }
    .confidence-track { flex:1; height: 8px; border-radius: 999px; background: var(--bg-main); border: 1px solid var(--border-c); overflow:hidden; }
    .confidence-fill { height: 100%; border-radius: 999px; }
    .confidence-num { font-size: 13px; font-weight: 700; width: 34px; text-align:right; }

    .price-target-row { display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
    .price-target-box { padding: 8px 10px; border-radius: 10px; background: var(--bg-main); border: 1px solid var(--border-c); }
    .price-target-box .pt-label { font-size: 11px; color: var(--text-sub); font-weight:600; }
    .price-target-box .pt-value { font-size: 15px; font-weight: 700; margin-top: 2px; }

    .holding-card ul.hc-reasons { margin: 12px 0 0 0; padding: 0 0 0 18px; font-size: 13.5px; color: var(--text-sub); line-height: 1.7; }

    /* =========================
       V10 Taiwan Trading Terminal UI
       ========================= */
    .terminal-header {
        display:flex; justify-content:space-between; align-items:center; gap:18px;
        padding:10px 2px 18px 2px; margin-bottom:4px;
    }
    .brand-block { display:flex; align-items:center; gap:12px; }
    .brand-mark {
        width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        color:#8ec5ff; border:1px solid #2b6aa3; background:radial-gradient(circle,#102b49 0%,#08111d 72%);
        font-size:25px; box-shadow:0 0 22px rgba(10,132,255,.18);
    }
    .brand-title { font-size:22px; font-weight:800; letter-spacing:.02em; line-height:1.05; }
    .brand-title span { color:#58a6ff; font-size:.82em; }
    .brand-sub { color:var(--text-sub); font-size:12px; margin-top:4px; }
    .header-status { color:var(--text-sub); font-size:11.5px; display:flex; align-items:center; gap:7px; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--tw-up); box-shadow:0 0 8px rgba(255,69,58,.55); }

    .market-overview { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:4px 0 14px; }
    .overview-card {
        padding:13px 14px; border:1px solid var(--border-c); border-radius:12px;
        background:linear-gradient(180deg,#151922 0%,#0d1016 100%); min-height:86px;
    }
    .overview-label { color:var(--text-sub); font-size:11px; font-weight:600; margin-bottom:8px; }
    .overview-value { font-size:22px; font-weight:800; line-height:1.05; font-variant-numeric:tabular-nums; }
    .overview-sub { margin-top:5px; font-size:11px; font-variant-numeric:tabular-nums; }
    .tw-up { color:var(--tw-up) !important; }
    .tw-down { color:var(--tw-down) !important; }
    .tw-flat { color:var(--tw-flat) !important; }

    .scanner-launch-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:8px 0 16px; }
    .scanner-launch {
        border:1px solid var(--border-c); border-radius:12px; padding:15px 17px;
        background:linear-gradient(180deg,#11151d 0%,#0b0e13 100%);
    }
    .scanner-launch.live { border-color:rgba(255,69,58,.48); background:linear-gradient(180deg,rgba(90,12,12,.22),#0b0e13); }
    .scanner-launch.eod { border-color:rgba(10,132,255,.5); background:linear-gradient(180deg,rgba(10,50,100,.22),#0b0e13); }
    .scanner-launch-title { font-size:14px; font-weight:800; margin-bottom:4px; }
    .scanner-launch.live .scanner-launch-title { color:var(--tw-up); }
    .scanner-launch.eod .scanner-launch-title { color:#58a6ff; }
    .scanner-launch-sub { color:var(--text-sub); font-size:11.5px; line-height:1.5; }
    .scanner-launch .launch-badge { display:inline-block; margin-top:8px; padding:3px 8px; border-radius:999px; font-size:10.5px; border:1px solid var(--border-c); color:var(--text-sub); }

    .section-bar {
        display:flex; justify-content:space-between; align-items:center; gap:10px;
        margin:12px 0 8px; padding:10px 12px; border:1px solid var(--border-c); border-radius:10px;
        background:#0d1118;
    }
    .section-bar-title { font-size:13px; font-weight:800; }
    .section-bar-sub { color:var(--text-sub); font-size:10.5px; }

    .legend-card {
        margin-top:12px; padding:12px 13px; border:1px solid var(--border-c); border-radius:10px;
        background:linear-gradient(180deg,#14171d,#0e1014);
    }
    .legend-title { font-size:12px; font-weight:800; margin-bottom:8px; }
    .legend-row { display:flex; align-items:center; gap:7px; font-size:11.5px; margin:5px 0; color:var(--text-sub); }
    .legend-swatch { width:7px; height:7px; border-radius:50%; flex:0 0 auto; }

    .score-badge { display:inline-block; min-width:38px; padding:3px 7px; border-radius:6px; text-align:center; font-weight:800; color:#fff; background:#222b38; }
    .decision-buy { color:#fff !important; background:rgba(48,209,88,.78); border-radius:5px; padding:3px 7px; font-weight:800; }
    .decision-watch { color:#111 !important; background:rgba(255,214,10,.9); border-radius:5px; padding:3px 7px; font-weight:800; }
    .decision-stop { color:#fff !important; background:rgba(255,69,58,.82); border-radius:5px; padding:3px 7px; font-weight:800; }

    @media (max-width: 1100px) {
        .market-overview { grid-template-columns:repeat(3,1fr); }
    }
    @media (max-width: 900px) {
        .risk-profile-grid, .stat-chip-row, .price-target-row { grid-template-columns: 1fr 1fr; }
        .market-overview, .scanner-launch-grid { grid-template-columns:1fr 1fr; }
        .terminal-header { align-items:flex-start; flex-direction:column; }
    }

    /* ── 側邊欄自選觀察名單：chip 清單，取代原本的 data_editor 白底表格 ── */
    .chip-item {
        display: flex; align-items: center; gap: 8px;
        padding: 9px 12px; margin-bottom: 6px;
        border-radius: 9px; border: 1px solid var(--border-c);
        background-color: var(--bg-card-2);
        font-size: 13.5px; font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .chip-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-blue); flex-shrink: 0; }
    [data-testid="stSidebar"] .stButton>button {
        padding: 0.25rem 0.5rem !important; min-height: 34px !important;
    }
    [data-testid="stSidebar"] [data-testid="column"]:has(button[kind="secondary"]) { display:flex; align-items:center; }

    /* ── 持有部位 / 自訂列編輯器：欄位標題列 ── */
    .row-editor-head {
        font-size: 11.5px; font-weight: 700; color: var(--text-sub);
        letter-spacing: 0.03em; padding: 0 2px 6px 2px; text-transform: uppercase;
    }

    /* st.container(border=True) 用在「持有部位」每一列，統一卡片化風格，取代 data_editor 的白底格線 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border-color: var(--border-c) !important;
        background-color: var(--bg-card-2) !important;
        transition: border-color 0.15s ease-in-out;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #55555a !important; }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

    /* number_input 的加減按鈕深色化，字型改等寬數字，更有「金融終端機」的專業感 */
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        font-variant-numeric: tabular-nums;
        font-weight: 600 !important;
    }
    [data-testid="stNumberInput"] button {
        background-color: var(--bg-card) !important;
        border-color: var(--border-c) !important;
        color: var(--text-sub) !important;
    }
    [data-testid="stNumberInput"] button:hover { background-color: var(--bg-card-2) !important; color: var(--text-main) !important; }

    /* 表單（新增觀察股 / 未來擴充）容器去除多餘留白，融入側邊欄風格 */
    [data-testid="stSidebar"] [data-testid="stForm"] {
        border: none !important; padding: 0 !important; background: transparent !important;
    }

    /* ── 隱藏 Streamlit 自帶的工具列 / Deploy 按鈕 / Fork·GitHub 按鈕 / 頁尾浮水印 ──
       這些不是我們畫面的一部分，全部藏起來，讓畫面乾淨、只有這個 App 本身的內容。 */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
    footer { visibility: hidden !important; height: 0 !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    div[class^="viewerBadge"], div[class*=" viewerBadge"] { display: none !important; }
    iframe[title*="viewer_badge"], iframe[title*="Streamlit"] { display: none !important; }
    a[href*="streamlit.io"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="terminal-header">
  <div class="brand-block">
    <div class="brand-mark">✦</div>
    <div>
      <div class="brand-title">QUANT COMPASS <span>V10.0</span></div>
      <div class="brand-sub">Smart Real-Time Scanner · 台股量化決策終端</div>
    </div>
  </div>
  <div class="header-status">
    <span class="status-dot"></span> 台股市場研究模式 · Point-in-Time / Unified Buy Score
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 1. API 與 Token 鎖定按鈕
# =========================
api = DataLoader()
_FINMIND_THREAD_LOCAL = threading.local()

# Token／API 設定的輸入元件移到「⚙️ 系統設定」分頁（見下方 render_settings_tab），
# 這裡只用 session_state 裡已套用的值做登入，讓一般使用者不必在主畫面看到這些研究員參數。
active_token = st.session_state["token_applied"]
_ACTIVE_FINMIND_TOKEN = active_token

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

def get_finmind_api():
    """每個 worker thread 使用自己的 DataLoader，避免多執行緒共用同一個 client。
    這是 V10 盤後掃描的重要穩定性修正：原本 18 檔同時分析時，共用 DataLoader
    可能造成請求互相干擾，最後全部回傳空資料。"""
    client = getattr(_FINMIND_THREAD_LOCAL, "client", None)
    if client is None:
        client = DataLoader()
        token = _ACTIVE_FINMIND_TOKEN
        if not token:
            try:
                token = st.secrets.get("FINMIND_TOKEN", "")
            except Exception:
                token = ""
        if token:
            try:
                client.login_by_token(api_token=token)
            except Exception:
                pass
        _FINMIND_THREAD_LOCAL.client = client
    return client

def has_finmind_secret():
    """安全檢查是否設定了 FINMIND_TOKEN。
    本機沒有 secrets.toml 檔案時，直接讀取 st.secrets 會丟出
    StreamlitSecretNotFoundError；這裡統一攔截，沒有金鑰就當作沒有，不讓整個流程中斷。
    """
    try:
        return "FINMIND_TOKEN" in st.secrets
    except Exception:
        return False


API_SLEEP_SEC = 0.35

def throttle():
    time.sleep(API_SLEEP_SEC)


def parallel_map(items, fn, max_workers=3):
    """Run independent stock tasks concurrently while keeping result order stable."""
    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers or 1), len(items)))
    if workers == 1:
        return [fn(x) for x in items]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="twq") as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    _flush_api_errors()
    return results

# FinMind 的台股日K／營收／財報／PER/PBR／法人買賣，都是「收盤後才更新一次」的資料
# (平日約 15:00~21:00 陸續更新)，同一天內重複抓取只會拿到一模一樣的內容、卻白白燒掉
# 每小時 300/600 次的額度。把快取時間拉長到 6 小時，同一天內大部分操作都會直接命中
# 快取，不會再打 API；如果懷疑資料真的過期，仍可在「⚙️ 系統設定」按「清除快取」強制重抓。
EOD_CACHE_TTL = 21600

# =========================
# V10.0 Smart Scanner：資料分層政策
# 盤中掃描只使用交易所快照 + 已快取的盤後研究結果，原則上 0 FinMind。
# 盤後深度掃描使用 V10 核心研究引擎；盤中只做即時行情調整，不複製另一套基本面評分。
# =========================
INTRADAY_CACHE_TTL = 15
INTRADAY_TOP_N = 30
INTRADAY_BASE_WEIGHT = 0.70
INTRADAY_MOMENTUM_WEIGHT = 0.30
INTRADAY_SCAN_MIN_TURNOVER = 50_000_000
INTRADAY_TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?response=json&type=ALLBUT0999"
INTRADAY_TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
INTRADAY_SCAN_CACHE_FILE = CACHE_DIR / "intraday_scan.pkl"
EOD_SCAN_CACHE_FILE = CACHE_DIR / "eod_scan.pkl"

if "api_errors" not in st.session_state:
    st.session_state["api_errors"] = []

_API_ERROR_LOCK = threading.Lock()
_API_ERROR_BUFFER = []

def _log_api_error(api_name, stock_id, exc):
    """Thread-safe error collector. Worker threads never mutate Streamlit session state directly."""
    item = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "api": api_name,
        "stock_id": stock_id,
        "error": f"{type(exc).__name__}: {exc}"
    }
    with _API_ERROR_LOCK:
        _API_ERROR_BUFFER.append(item)
        del _API_ERROR_BUFFER[:-50]

def _flush_api_errors():
    with _API_ERROR_LOCK:
        pending = list(_API_ERROR_BUFFER)
        _API_ERROR_BUFFER.clear()
    if pending:
        current = st.session_state.get("api_errors", [])
        st.session_state["api_errors"] = (current + pending)[-30:]

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



def decision_priority(score, risk, market_regime, status):
    """把買進分、風險與市場環境整合成 UI 用的優先級；不改變核心買進分。"""
    score = safe_float(score, 0)
    risk_penalty = {"🟢 低": 0, "🟡 中": 8, "🔴 高": 18}.get(risk, 10)
    regime_penalty = {"BULL": 0, "NEUTRAL": 6, "BEAR": 15}.get(market_regime, 10)
    heat_penalty = 8 if "過熱" in str(status) else 0
    return round(clamp(score - risk_penalty - regime_penalty - heat_penalty), 1)


def data_quality_label(result):
    """UI 顯示資料完整度，避免使用者把缺資料誤認成低分。"""
    if not result:
        return "⚪ 無資料"
    fields = ["基本面", "估值", "籌碼", "技術", "RSI", "ADX", "ATR"]
    valid = sum(not pd.isna(safe_float(result.get(k))) for k in fields)
    if valid >= 7:
        return "🟢 完整"
    if valid >= 5:
        return "🟡 部分缺資料"
    return "🔴 資料不足"


def risk_reward_ratio(price, stop, target):
    price, stop, target = safe_float(price), safe_float(stop), safe_float(target)
    if any(pd.isna(v) for v in [price, stop, target]) or price <= stop:
        return np.nan
    risk = price - stop
    reward = target - price
    return reward / risk if risk > 0 else np.nan


def render_factor_bars(result):
    """股票分析頁的因子拆解：讓買進分不是黑箱。"""
    factors = [
        ("基本面", result.get("基本面", np.nan)),
        ("估值", result.get("估值", np.nan)),
        ("籌碼", result.get("籌碼", np.nan)),
        ("技術", result.get("技術", np.nan)),
        ("護城河", result.get("護城河", np.nan)),
        ("起漲", result.get("起漲分", np.nan)),
    ]
    rows = [(n, float(v)) for n, v in factors if not pd.isna(safe_float(v))]
    if not rows:
        return
    fig = go.Figure(go.Bar(
        x=[v for _, v in rows],
        y=[n for n, _ in rows],
        orientation="h",
        text=[f"{v:.0f}" for _, v in rows],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 100], title="因子強度")
    fig.update_layout(
        title="📊 買進分因子拆解",
        template="plotly_dark",
        height=330,
        margin=dict(l=20, r=45, t=55, b=35),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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

def _tw_color(v):
    """台股價格/報酬配色：正數紅、負數綠、0 灰。"""
    try:
        x = float(v)
        if pd.isna(x) or x == 0:
            return "color:#a1a1a6;font-weight:600;"
        return "color:#ff453a;font-weight:700;" if x > 0 else "color:#30d158;font-weight:700;"
    except Exception:
        return ""

def style_market_returns(df, columns=None):
    if df is None or df.empty:
        return df
    styler = df.style
    cols = columns or [c for c in df.columns if any(k in str(c) for k in ["漲跌", "報酬", "獲利", "損益"]) and "MDD" not in str(c) and "回撤" not in str(c)]
    for col in cols:
        if col in df.columns:
            try:
                if hasattr(styler, "map"):
                    styler = styler.map(_tw_color, subset=[col])
                else:
                    styler = styler.applymap(_tw_color, subset=[col])
            except Exception:
                pass
    return styler

def style_pnl(df):
    return style_market_returns(df, [c for c in ["獲利(%)", "損益%", "未實現損益"] if c in df.columns])

def style_scan_table(df):
    styler = style_market_returns(df)
    # 即時/盤後表格的價格與買進分維持清楚的金融終端顏色。
    for col in ["買進分", "即時調整分", "基準買進分", "盤中動能分"]:
        if col in df.columns:
            def score_color(v):
                try:
                    x=float(v)
                    if x>=85: return "color:#ffd60a;font-weight:800;"
                    if x>=65: return "color:#f5f5f7;font-weight:700;"
                    return "color:#a1a1a6;font-weight:600;"
                except Exception:
                    return ""
            try:
                styler = styler.map(score_color, subset=[col]) if hasattr(styler, "map") else styler.applymap(score_color, subset=[col])
            except Exception:
                pass
    return styler

def style_intraday_table(df):
    return style_market_returns(df, [c for c in ["漲跌", "即時漲跌%"] if c in df.columns])


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
    df = get_finmind_api().taiwan_stock_daily(stock_id=stock_id, start_date=start)
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
    info = get_finmind_api().taiwan_stock_info()
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
    df = get_finmind_api().taiwan_stock_month_revenue(stock_id=stock_id, start_date=start)
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
    df = get_finmind_api().taiwan_stock_financial_statement(stock_id=stock_id, start_date=start)
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
    df = get_finmind_api().taiwan_stock_per_pbr(stock_id=stock_id, start_date=start)
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
    df = get_finmind_api().taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
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
# V10.0 盤中即時資料層：不碰 FinMind
# =========================
def _num(v):
    return pd.to_numeric(pd.Series([v]).astype(str).str.replace(",", "", regex=False).str.replace("--", "", regex=False), errors="coerce").iloc[0]

def _normalize_intraday_frame(df, source):
    if df is None or df.empty:
        return pd.DataFrame()
    id_col = _pick_col(df, ["證券代號", "SecuritiesCompanyCode", "Code", "代號"])
    name_col = _pick_col(df, ["證券名稱", "SecuritiesCompanyName", "名稱"])
    price_col = _pick_col(df, ["成交價", "成交價格", "Close", "close", "最後成交價", "收盤價"])
    change_col = _pick_col(df, ["漲跌價差", "漲跌", "Change", "change"])
    pct_col = _pick_col(df, ["漲跌幅", "漲跌幅%", "ChangePercent", "change_percent"])
    vol_col = _pick_col(df, ["成交股數", "成交量", "TradingVolume", "Volume", "volume"])
    turnover_col = _pick_col(df, ["成交金額", "成交額", "TradingValue", "TradeValue", "today_turnover"])
    if not id_col:
        return pd.DataFrame()
    out = pd.DataFrame({"股票代碼": df[id_col].astype(str).str.strip().str.replace(".0", "", regex=False)})
    out = out[out["股票代碼"].str.match(r"^\d{4}$", na=False)].copy()
    if name_col: out["名稱"] = df.loc[out.index, name_col].astype(str).values
    if price_col: out["即時價"] = df.loc[out.index, price_col].map(_num).values
    if change_col: out["漲跌"] = df.loc[out.index, change_col].map(_num).values
    if pct_col: out["即時漲跌%"] = df.loc[out.index, pct_col].map(_num).values
    if vol_col: out["成交量"] = df.loc[out.index, vol_col].map(_num).values
    if turnover_col: out["成交金額"] = df.loc[out.index, turnover_col].map(_num).values
    out["來源"] = source
    return out.reset_index(drop=True)

@st.cache_data(ttl=INTRADAY_CACHE_TTL, show_spinner=False)
def get_intraday_market_snapshot():
    frames = []
    try:
        r = requests.get(INTRADAY_TWSE_URL, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        payload = r.json()
        tables = payload.get("tables", []) if isinstance(payload, dict) else []
        for table in tables:
            fields, data = table.get("fields"), table.get("data")
            if fields and data:
                fdf = pd.DataFrame(data, columns=fields)
                n = _normalize_intraday_frame(fdf, "TWSE")
                if not n.empty: frames.append(n)
    except Exception as e:
        _log_api_error("TWSE intraday snapshot", "-", e)
    try:
        r = requests.get(INTRADAY_TPEX_URL, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        payload = r.json()
        tdf = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
        n = _normalize_intraday_frame(tdf, "TPEx")
        if not n.empty: frames.append(n)
    except Exception as e:
        _log_api_error("TPEx intraday snapshot", "-", e)
    if not frames:
        raise ValueError("交易所盤中快照無有效資料")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("股票代碼", keep="first")
    for c in ["即時價", "漲跌", "即時漲跌%", "成交量", "成交金額"]:
        if c not in out: out[c] = np.nan
    return out

def _intraday_score(row):
    pct = safe_float(row.get("即時漲跌%"), 0)
    turnover = safe_float(row.get("成交金額"), 0)
    score = 50.0
    score += clamp(pct * 6, -30, 30)
    if turnover >= 1_000_000_000: score += 12
    elif turnover >= 500_000_000: score += 8
    elif turnover >= 100_000_000: score += 4
    return clamp(score)

def run_intraday_scan(universe_df, top_n=INTRADAY_TOP_N):
    snap = get_intraday_market_snapshot()
    if snap.empty: return pd.DataFrame()
    uni = universe_df[[c for c in ["stock_id", "stock_name", "type"] if c in universe_df.columns]].copy()
    uni = uni.rename(columns={"stock_id":"股票代碼", "stock_name":"名稱"})
    out = uni.merge(snap, on="股票代碼", how="inner", suffixes=("", "_snap"))
    out["成交金額"] = pd.to_numeric(out["成交金額"], errors="coerce").fillna(0)
    out = out[out["成交金額"] >= INTRADAY_SCAN_MIN_TURNOVER].copy()
    if out.empty: return out
    out["盤中動能分"] = out.apply(_intraday_score, axis=1)
    saved = load_saved_scan()
    if isinstance(saved, dict) and isinstance(saved.get("out"), pd.DataFrame) and not saved["out"].empty:
        base = saved["out"][[c for c in ["股票代碼", "買進分", "風險", "狀態", "決策"] if c in saved["out"].columns]].drop_duplicates("股票代碼")
        out = out.merge(base, on="股票代碼", how="left")
    out["基準買進分"] = pd.to_numeric(out.get("買進分"), errors="coerce")
    out["基準買進分"] = out["基準買進分"].fillna(50)
    out["即時調整分"] = (out["基準買進分"] * INTRADAY_BASE_WEIGHT + out["盤中動能分"] * INTRADAY_MOMENTUM_WEIGHT).round(1)
    out["盤中訊號"] = np.select([out["即時漲跌%"] >= 5, out["即時漲跌%"] >= 2, out["即時漲跌%"] <= -3], ["🔥 強勢放量", "🟢 盤中轉強", "🔴 盤中轉弱"], default="🟡 中性")
    out = out.sort_values(["即時調整分", "成交金額"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    out.insert(0, "排名", np.arange(1, len(out)+1))
    return out

def save_intraday_scan(df):
    try:
        pd.to_pickle({"out": df, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, INTRADAY_SCAN_CACHE_FILE)
    except Exception: pass

def load_intraday_scan():
    try:
        return pd.read_pickle(INTRADAY_SCAN_CACHE_FILE)
    except Exception:
        return None

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
    # 盤後 shortlist 的完整研究現在採「批次 API」：
    # 初篩仍是每檔日 K；完整研究約 5 個 batch request，而不是 topk×5。
    "⚡ 快速": {"prefilter": 150, "topk": 12},
    "🎯 標準": {"prefilter": 400, "topk": 18},
    "🔬 深度": {"prefilter": None, "topk": 25},
}

def effective_scan_config(strength_label):
    """依是否有 Token 自動避免免費額度被 400 檔初篩吃光。"""
    cfg = dict(SCAN_STRENGTH_CONFIG[strength_label])
    if not active_token and not has_finmind_secret():
        if strength_label == "🎯 標準":
            cfg.update({"prefilter": 240, "topk": 12})
        elif strength_label == "🔬 深度":
            cfg.update({"prefilter": 240, "topk": 12})
    return cfg


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
    cfg = effective_scan_config(strength_label)
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
        ref20_close = safe_float(daily.iloc[-21].get("close")) if len(daily) >= 21 else np.nan
        change_20d_pct = (price / ref20_close - 1) * 100 if not pd.isna(ref20_close) and ref20_close > 0 else np.nan
        limit_status = limit_up_status(price, prev_close, safe_float(x.get("max")), safe_float(x.get("min")), day_change_pct)
        decision = decision_label(buy_score, overheat=overheat, limit_up=limit_status.startswith("🔒"), market_regime=regime_dict["regime"])
        priority = decision_priority(buy_score, risk, regime_dict["regime"], status_label)
        quality_inputs = {
            "基本面": any(not pd.isna(safe_float(fund.get(k))) for k in ["roe","roa","gross_margin","op_margin","eps_growth","revenue_growth"]),
            "估值": any(not pd.isna(safe_float(val.get(k))) for k in ["PER","PBR"]),
            "籌碼": bool(chip_detail),
            "技術": all(not pd.isna(safe_float(x.get(k))) for k in ["RSI","ADX","ATR"]),
        }
        quality = "🟢 完整" if sum(quality_inputs.values()) == 4 else ("🟡 部分缺資料" if sum(quality_inputs.values()) >= 2 else "🔴 資料不足")
        if decision == "🟢 可買": explanation = "整體條件強，趨勢、基本面、估值與籌碼條件同步。"
        elif decision == "🟡 過熱觀察": explanation = "趨勢仍強，但短線動能偏熱，優先等回檔或確認。"
        elif decision == "⚠️ 漲停勿追": explanation = "分數高不代表可以追價，價格已接近漲停區。"
        elif decision == "🔴 不買": explanation = "多項條件未同時成立，目前不列入新增買進。"
        else: explanation = "條件介於中間，等待更多訊號確認。"
        reasons = build_reasons(decision, breakout_reasons, chip_detail, fund, val, status_label)
        return {"股票代碼": stock_id, "現價": round(price,2), "買進分": round(buy_score,1), "優先級": priority,
                "狀態": status_label, "風險": risk, "資料品質": quality,
                "近1日漲跌%": round(day_change_pct,2) if not pd.isna(day_change_pct) else np.nan, "近5日漲跌%": round(change_5d_pct,2) if not pd.isna(change_5d_pct) else np.nan, "近20日漲跌%": round(change_20d_pct,2) if not pd.isna(change_20d_pct) else np.nan,
                "成交量": int(safe_float(x.get("volume"),0)), "量比": safe_float(x["VOL_RATIO"]), "漲停狀態": limit_status, "決策": decision, "說明": explanation, "理由": reasons,
                "日期": as_of.strftime("%Y-%m-%d"), "綜合分": round(final,1), "起漲分": round(early_score,1), "基本面": round(fund_pct,1), "估值": round(val_pct,1), "籌碼": round(chips_pct,1), "技術": round(technical,1),
                "護城河": round(moat,1), "RSI": safe_float(x["RSI"]), "ADX": safe_float(x["ADX"]), "ATR": safe_float(x["ATR"]), "PEG": val["PEG"], "PER": val["PER"], "PBR": val["PBR"],
                "過熱": "是" if overheat else "否", "評級": decision, "起漲理由": "、".join(breakout_reasons[:5]),
                "趨勢": [round(float(v), 2) for v in daily["close"].tail(20).pct_change().fillna(0).cumsum().add(1).tolist()],
                "daily": daily, "fund": fund, "moat_detail": moat_detail,
                "_pit_note": "財報可得日以財報日期+45天、營收以日期+15天作保守代理。"
                }
    except Exception as e:
        _log_api_error("calculate_stock_snapshot", stock_id, e)
        return None


def _normalize_batch_df(df, numeric_cols=()):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "stock_id" in out.columns:
        out["stock_id"] = out["stock_id"].astype(str)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def prepare_pit_sources_batch(stock_ids, daily_days=1500):
    """盤後完整分析的批次資料抓取。

    針對 shortlist 一次抓 5 個 dataset，再依 stock_id 切回個股。
    FinMind DataLoader 支援 stock_id_list，因此 18 檔完整分析不再變成 18×5 次 API；
    這同時降低 Token 消耗，也避免多執行緒同時轟 API。
    若某一個批次 dataset 失敗，只讓該 dataset 為空，不影響其他資料繼續分析。
    """
    ids = [str(x) for x in dict.fromkeys(stock_ids) if str(x)]
    if not ids:
        return {}
    client = get_finmind_api()
    start_daily = (datetime.now() - timedelta(days=daily_days)).strftime("%Y-%m-%d")
    starts = {
        "revenue": (datetime.now() - timedelta(days=1800)).strftime("%Y-%m-%d"),
        "financial": (datetime.now() - timedelta(days=2400)).strftime("%Y-%m-%d"),
        "per_pbr": (datetime.now() - timedelta(days=1800)).strftime("%Y-%m-%d"),
        "institutional": (datetime.now() - timedelta(days=600)).strftime("%Y-%m-%d"),
    }
    calls = [
        ("daily", lambda: client.taiwan_stock_daily(stock_id_list=ids, start_date=start_daily), ["close","open","max","min","Trading_turnover","Trading_Volume","Trading_money"]),
        ("revenue", lambda: client.taiwan_stock_month_revenue(stock_id_list=ids, start_date=starts["revenue"]), ["revenue","revenue_year_on_year","revenue_month_on_month"]),
        ("financial", lambda: client.taiwan_stock_financial_statement(stock_id_list=ids, start_date=starts["financial"]), ["value"]),
        ("per_pbr", lambda: client.taiwan_stock_per_pbr(stock_id_list=ids, start_date=starts["per_pbr"]), ["PER","PBR","dividend_yield"]),
        ("institutional", lambda: client.taiwan_stock_institutional_investors(stock_id_list=ids, start_date=starts["institutional"]), ["buy","sell"]),
    ]
    raw = {}
    for name, fn, numeric in calls:
        try:
            df = fn()
            throttle()
            if df is None or df.empty:
                raw[name] = pd.DataFrame()
                _log_api_error(f"batch_{name}", ",".join(ids[:8]), ValueError("批次 API 回傳空資料"))
            else:
                raw[name] = _normalize_batch_df(df, numeric)
                if name == "daily":
                    if "Trading_Volume" in raw[name].columns:
                        raw[name]["volume"] = pd.to_numeric(raw[name]["Trading_Volume"], errors="coerce")
                    elif "Trading_volume" in raw[name].columns:
                        raw[name]["volume"] = pd.to_numeric(raw[name]["Trading_volume"], errors="coerce")
        except Exception as e:
            raw[name] = pd.DataFrame()
            _log_api_error(f"batch_{name}", ",".join(ids[:8]), e)

    result = {sid: {} for sid in ids}
    for sid in ids:
        for name, df in raw.items():
            if df.empty or "stock_id" not in df.columns:
                result[sid][name] = pd.DataFrame()
            else:
                result[sid][name] = df[df["stock_id"].astype(str) == sid].copy().sort_values("date").reset_index(drop=True)
    return result


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
# 6.5 庫存健康檢查：把「現在該怎麼辦」變成一個可以直接照做的建議
#     沿用跟「今日選股」完全相同的評分引擎（calculate_stock），
#     再結合你自己輸入的持有成本，換算成停損／加碼／攤平／出清的操作建議。
# =========================
RISK_PROFILE_PRESETS = {
    "保守": {"base_stop_loss_pct": 6.0, "desc": "資金保護優先，跌破防線就先出場，寧可少賺也不多賠。"},
    "平衡": {"base_stop_loss_pct": 10.0, "desc": "在風險與空間之間取中間值，適合大多數人。"},
    "積極": {"base_stop_loss_pct": 15.0, "desc": "願意承受較大波動、換取讓獲利部位『多跑一段』的空間。"},
}


def evaluate_holding_action(stock_row, cost, shares, stop_loss_pct=8.0, take_profit_pct=None):
    """回傳這一檔庫存的健康檢查結果：操作建議、理由、AI 動態停損／停利價、續抱信心分。

    核心邏輯（取代過去死板的固定 % 停損停利）：
    1. 停損防線 = 「近期高點 - ATR × 趨勢係數」與「成本價 × (1-你的下限%)」兩者取較高（較不吃虧）的一個。
       趨勢係數依 ADX（趨勢強度）自動調整：趨勢越強，給的空間越大，避免「一回檔就被洗出場、錯過主升段」；
       趨勢走弱／盤整時則自動收緊，避免虧損持續擴大。
    2. 停利改用「風險倍數（R-multiple）」動態目標價，而不是固定 %，獲利目標會隨波動度自動放大縮小，
       並在達標時建議「先了結一部分、剩餘部位改設移動停損」，取代一次全出，兼顧「不賣飛」與「顧獲利」。
    3. 攤平只在「基本面／技術面仍偏多、且尚未跌破你的停損防線」時才建議，並且明確提醒「最多攤平一次、
       攤平後要重設停損」，避免陷入越攤越薄的無底洞。
    4. 額外提供 0-100 的「續抱信心分」，把買進分、均線排列、是否過熱、風險燈號綜合成一個分數，
       讓你在「續抱觀察」這種中性情況下，也能看出目前偏樂觀還是偏保守。

    這是規則式＋量化訊號組成的參考建議，不是投資建議，最終判斷仍要自己做。
    """
    price = safe_float(stock_row.get("現價"))
    atr = safe_float(stock_row.get("ATR"))
    buy_score = safe_float(stock_row.get("買進分"), 0)
    decision = stock_row.get("決策", "") or ""
    status = stock_row.get("狀態", "") or ""
    risk = stock_row.get("風險", "") or ""
    rsi = safe_float(stock_row.get("RSI"), 50)
    adx = safe_float(stock_row.get("ADX"), 20)
    if pd.isna(adx):
        adx = 20

    daily = stock_row.get("daily")
    ma20 = ma60 = high60 = np.nan
    if isinstance(daily, pd.DataFrame) and not daily.empty:
        last = daily.iloc[-1]
        ma20 = safe_float(last.get("MA20"))
        ma60 = safe_float(last.get("MA60"))
        high60 = safe_float(last.get("HIGH_60"))

    cost = safe_float(cost)
    pnl_pct = (price / cost - 1) * 100 if cost > 0 and not pd.isna(price) else np.nan
    market_value = price * shares if not pd.isna(price) else np.nan
    unrealized_pnl = (price - cost) * shares if (not pd.isna(price) and cost > 0) else np.nan

    # 1) 趨勢係數：ADX 越高（趨勢越明確），給的 ATR 緩衝倍數越大
    if adx >= 30:
        atr_mult, trend_tag = 3.0, "強趨勢"
    elif adx >= 20:
        atr_mult, trend_tag = 2.2, "中度趨勢"
    else:
        atr_mult, trend_tag = 1.4, "盤整偏弱"

    trail_anchor = np.nanmax([v for v in [price, high60] if not pd.isna(v)]) if (
        not pd.isna(price) or not pd.isna(high60)) else np.nan
    atr_stop = trail_anchor - atr_mult * atr if (not pd.isna(trail_anchor) and not pd.isna(atr) and atr > 0) else np.nan
    hard_stop = cost * (1 - stop_loss_pct / 100) if cost > 0 else np.nan
    stop_candidates = [v for v in [atr_stop, hard_stop] if not pd.isna(v)]
    suggested_stop = max(stop_candidates) if stop_candidates else np.nan
    if not pd.isna(suggested_stop) and not pd.isna(price) and suggested_stop >= price:
        suggested_stop = price * 0.97  # 極端情況的保護，避免停損價高於現價

    # 2) 動態停利目標：以「風險倍數」取代固定 %，讓目標隨波動度自動縮放
    risk_per_share = (price - suggested_stop) if (not pd.isna(suggested_stop) and not pd.isna(price)) else np.nan
    target1 = price + risk_per_share * 1.5 if (not pd.isna(risk_per_share) and risk_per_share > 0) else np.nan
    target2 = price + risk_per_share * 3.0 if (not pd.isna(risk_per_share) and risk_per_share > 0) else np.nan

    # 3) 續抱信心分：買進分為底，疊加均線排列／是否過熱／風險燈號
    confidence = buy_score
    if not pd.isna(ma20) and not pd.isna(ma60) and not pd.isna(price):
        if price > ma20 > ma60:
            confidence += 8
        elif price < ma20 < ma60:
            confidence -= 10
    if rsi >= 80:
        confidence -= 12
    elif rsi <= 25:
        confidence -= 6
    if risk == "🔴 高":
        confidence -= 10
    confidence = clamp(confidence)

    trend_broken = "🔴" in status
    tech_weak = "🔴" in decision
    overheated = "🟠" in status or rsi >= 80

    reasons = []
    if pd.isna(price):
        action = "⚠️ 資料不足"
        reasons.append("目前抓不到有效股價，稍後再檢查一次，或到「⚙️ 系統設定」看 API 診斷。")
    elif not pd.isna(suggested_stop) and price <= suggested_stop:
        action = "🔻 建議停損"
        reasons.append(f"現價已跌破 AI 動態停損防線 {suggested_stop:.2f}（依{trend_tag}、ATR×{atr_mult:.1f} 與你的成本下限估算）。")
        reasons.append("紀律優先：先出場保住本金，之後條件轉強再重新評估進場，比留在場上『賭它彈回來』更划算。")
    elif trend_broken and tech_weak:
        action = "🚪 建議出清"
        reasons.append("股價已跌破 MA20，且綜合決策已轉為「不買」等級，趨勢轉弱訊號明確。")
        if not pd.isna(pnl_pct):
            reasons.append(f"目前未實現損益 {pnl_pct:+.1f}%，技術面已不支持續抱。")
    elif not pd.isna(target2) and not pd.isna(price) and price >= target2:
        action = "🎯 建議獲利了結一部分"
        reasons.append(f"現價已達 AI 第二目標 {target2:.2f}（風險倍數 3R），建議先了結約 1/3～1/2 部位落袋。")
        reasons.append(f"剩餘部位改用移動停損 {suggested_stop:.2f} 顧住獲利即可，不用整筆賣掉，避免賣飛後面的漲幅。")
    elif not pd.isna(target1) and not pd.isna(price) and price >= target1 and overheated:
        action = "🎯 可考慮部分獲利了結"
        reasons.append(f"現價已達第一目標 {target1:.2f}，且短線偏過熱（{status}），可先落袋一小部分，其餘續抱看能不能挑戰 {target2:.2f}。")
    elif not pd.isna(pnl_pct) and pnl_pct < 0 and buy_score >= 60 and risk != "🔴 高" and not trend_broken and not pd.isna(suggested_stop) and price > suggested_stop:
        action = "➕ 可考慮攤平"
        reasons.append(f"目前未實現損益 {pnl_pct:.1f}%，但基本面／技術面條件仍偏正向（買進分 {buy_score:.0f}），且尚未跌破 AI 停損防線 {suggested_stop:.2f}。")
        reasons.append("建議最多攤平一次：攤平後務必用新成本重新計算停損價，一旦再度跌破就出場，避免越攤越薄、掉進無底洞。")
    elif not pd.isna(pnl_pct) and pnl_pct >= 0 and "🟢" in decision and risk != "🔴 高" and not overheated:
        action = "📈 可考慮加碼"
        reasons.append(f"目前獲利 {pnl_pct:.1f}%，趨勢與買進條件持續偏多，且沒有短線過熱訊號。")
    else:
        action = "🤝 續抱觀察"
        reasons.append(f"目前條件中性（續抱信心分 {confidence:.0f}），沒有出現明確的加碼或減碼訊號，維持原部位並持續留意停損防線 {suggested_stop:.2f}。" if not pd.isna(suggested_stop) else "目前條件中性，沒有出現明確的加碼或減碼訊號，維持原部位即可。")

    return {
        "現價": price, "損益%": round(pnl_pct, 2) if not pd.isna(pnl_pct) else np.nan,
        "市值": round(market_value, 0) if not pd.isna(market_value) else np.nan,
        "未實現損益": round(unrealized_pnl, 0) if not pd.isna(unrealized_pnl) else np.nan,
        "建議停損價": round(suggested_stop, 2) if not pd.isna(suggested_stop) else np.nan,
        "目標價1": round(target1, 2) if not pd.isna(target1) else np.nan,
        "目標價2": round(target2, 2) if not pd.isna(target2) else np.nan,
        "續抱信心分": round(confidence, 0) if not pd.isna(confidence) else np.nan,
        "趨勢係數": trend_tag,
        "買進分": buy_score, "決策": decision, "狀態": status, "風險": risk,
        "操作建議": action, "理由": reasons,
    }


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


def backtest_single(stock_id, initial_capital, fee, tax, slippage, hold_days=10, start_date=None, end_date=None):
    sources=prepare_pit_sources(stock_id,1500); daily=add_technical_indicators(sources["daily"])
    if daily.empty or len(daily)<250: return None
    daily["date"]=pd.to_datetime(daily["date"], errors="coerce")
    mkt=get_yahoo_taiex(); equity=[]; trades=[]; cash=float(initial_capital); shares=0; entry_price=0; entry_date=None; entry_i=0
    start_ts=pd.Timestamp(start_date) if start_date is not None else None
    end_ts=pd.Timestamp(end_date) if end_date is not None else None
    for i in range(120,len(daily)-1):
        if start_ts is not None and daily.iloc[i]["date"] < start_ts: continue
        if end_ts is not None and daily.iloc[i]["date"] > end_ts: break
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
    metrics.update({"stock":stock_id,"equity":eq,"trades_detail":trades,"benchmarks":bench,"daily":daily,"hold_days":hold_days})
    return metrics


def portfolio_backtest(stocks, initial_capital, top_n, fee=0.001425, tax=0.003, slippage=0.0015, rebalance_days=20, progress_cb=None, max_workers=3):
    data={}
    def load_one(s):
        src=prepare_pit_sources(s,1500); d=add_technical_indicators(src["daily"])
        return s, src if not d.empty else None
    loaded = parallel_map(stocks, load_one, max_workers=max_workers)
    for idx, item in enumerate(loaded):
        s, src = item
        if src is not None: data[s]=src
        if progress_cb: progress_cb((idx+1)/len(stocks))
    if not data:return None
    mkt=get_yahoo_taiex(); all_dates=sorted(set().union(*[set(pd.to_datetime(src["daily"]["date"])) for src in data.values()])); all_dates=pd.DatetimeIndex(all_dates)
    value=float(initial_capital); equity=[]; holdings={}; pending_weights=None; pending_date=None; last_rebalance=-999; turnover_cost_total=0
    for i,date in enumerate(all_dates):
        if i<120: continue

        # 用前一個交易日收盤產生訊號，下一交易日才套用權重，避免 look-ahead。
        if pending_weights is not None and pending_date is not None and date >= pending_date:
            new_weights=pending_weights
            sells=sum(max(0,holdings.get(s,0)-new_weights.get(s,0)) for s in set(new_weights)|set(holdings))
            turnover=sum(abs(new_weights.get(s,0)-holdings.get(s,0)) for s in set(new_weights)|set(holdings))
            cost=value*turnover*(fee+slippage)+value*sells*tax
            value=max(0,value-cost); turnover_cost_total+=cost; holdings=new_weights
            pending_weights=None; pending_date=None

        if i-last_rebalance>=rebalance_days:
            scores={}; reg=market_regime(date,mkt)
            for s,src in data.items():
                snap=calculate_stock_snapshot(s,date,src,reg)
                if snap is not None: scores[s]=snap["買進分"]
            ranked=[s for s,v in sorted(scores.items(),key=lambda z:z[1],reverse=True) if v>=65][:top_n]
            pending_weights={s:1/len(ranked) for s in ranked} if ranked else {}
            pending_date=all_dates[min(i+1, len(all_dates)-1)]
            last_rebalance=i

        day_ret=0.0
        for s,w in holdings.items():
            d=data[s]["daily"].copy(); d["date"]=pd.to_datetime(d["date"]); idxs=d.index[d["date"]==date]
            if len(idxs):
                j=idxs[0]
                if j>0:
                    p0=safe_float(d.iloc[j-1]["close"]); p1=safe_float(d.iloc[j]["close"])
                    if p0>0: day_ret+=w*(p1/p0-1)
        value=value*(1+day_ret); equity.append((date,value))
    eq=pd.Series(dict(equity)); bench=get_benchmarks(); metrics=performance_metrics(eq,[],benchmark=bench); metrics.update({"equity":eq,"trades_detail":[],"holdings":holdings,"turnover_cost_total":turnover_cost_total,"benchmarks":bench,"rebalance_days":rebalance_days}); return metrics


def walk_forward_test(stocks, initial_capital, fee, tax, slippage, hold_days=10, train_years=2, test_years=1):
    """真正的 rolling OOS 報告。
    本版不做參數最佳化，因此 train 區只用來建立時間切點；test 區完全封存，
    並以同一套 Point-in-Time 買進分引擎做測試。
    """
    rows=[]
    if not stocks:return pd.DataFrame()
    for s in stocks:
        try:
            src=prepare_pit_sources(s, 1500)
            d=src["daily"].copy()
            if d.empty: continue
            d["date"]=pd.to_datetime(d["date"], errors="coerce")
            first, last=d["date"].min(), d["date"].max()
            cursor=first + pd.DateOffset(years=train_years)
            fold=1
            while cursor + pd.DateOffset(years=test_years) <= last:
                test_start=cursor
                test_end=min(cursor + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), last)
                r=backtest_single(
                    s, initial_capital, fee, tax, slippage, hold_days=hold_days,
                    start_date=test_start, end_date=test_end
                )
                if r:
                    rows.append({
                        "股票":s, "Fold":fold,
                        "訓練區間":f"{first:%Y-%m-%d} ~ {(test_start-pd.Timedelta(days=1)):%Y-%m-%d}",
                        "OOS測試區間":f"{test_start:%Y-%m-%d} ~ {test_end:%Y-%m-%d}",
                        "CAGR":r.get("cagr",np.nan)*100,
                        "MDD":r.get("mdd",np.nan)*100,
                        "Sharpe":r.get("sharpe",np.nan),
                        "OOS勝率":r.get("win_rate",np.nan),
                        "交易次數":r.get("trades",0),
                    })
                cursor=cursor + pd.DateOffset(years=test_years)
                fold+=1
        except Exception as e:
            _log_api_error("walk_forward_test", s, e)
    return pd.DataFrame(rows)

# =========================
# 8. UI Sidebar（只留下每個分頁都會用到的東西：自選股 + 大盤狀態）
# =========================
st.sidebar.subheader("📌 自選觀察名單")
st.sidebar.caption("同步套用到「股票分析」與「歷史驗證」，代碼限 4 碼。")

if st.session_state["watchlist_codes"]:
    st.sidebar.markdown('<div class="chip-list">', unsafe_allow_html=True)
    _remove_watch_idx = None
    for _i, _code in enumerate(st.session_state["watchlist_codes"]):
        _wc1, _wc2 = st.sidebar.columns([5, 1])
        _wc1.markdown(f'<div class="chip-item"><span class="chip-dot"></span>{_code}</div>', unsafe_allow_html=True)
        if _wc2.button("✕", key=f"del_watch_{_i}_{_code}", help="從觀察名單移除"):
            _remove_watch_idx = _i
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    if _remove_watch_idx is not None:
        st.session_state["watchlist_codes"].pop(_remove_watch_idx)
        st.rerun()
else:
    st.sidebar.caption("目前名單是空的，在下面新增第一檔代碼。")

with st.sidebar.form("add_watch_form", clear_on_submit=True):
    _wa1, _wa2 = st.columns([5, 1.6])
    _new_code = _wa1.text_input("新增代碼", max_chars=4, label_visibility="collapsed", placeholder="＋ 輸入 4 碼代號")
    _submitted = _wa2.form_submit_button("加入")
    if _submitted and _new_code.strip():
        _cc = clean_stock_list(_new_code)
        if not _cc:
            st.sidebar.warning("代碼格式不正確，需為 4 碼數字。")
        elif _cc[0] in st.session_state["watchlist_codes"]:
            st.sidebar.info(f"{_cc[0]} 已經在名單中了。")
        else:
            st.session_state["watchlist_codes"].append(_cc[0])
            st.rerun()

stocks = clean_stock_list("\n".join(st.session_state["watchlist_codes"]))
st.session_state["watchlist_codes"] = stocks


# 回測參數的預設值（實際輸入元件移到「⚙️ 系統設定」分頁）
_settings_defaults = {"initial_capital": 1_000_000, "fee": 0.001425, "tax": 0.003, "slippage": 0.0015, "top_n": 5, "hold_days": 10, "scan_workers": 3}
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
st.sidebar.markdown("""
<div class="legend-card">
  <div class="legend-title">🎨 顏色說明（台股標準）</div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--tw-up)"></span><span><b style="color:var(--tw-up)">紅色</b>＝上漲／正報酬</span></div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--tw-down)"></span><span><b style="color:var(--tw-down)">綠色</b>＝下跌／負報酬</span></div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--tw-flat)"></span><span>灰色＝平盤／無變化</span></div>
  <div style="height:1px;background:var(--border-c);margin:9px 0"></div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--decision-buy)"></span><span>決策：可買</span></div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--decision-watch)"></span><span>決策：觀察</span></div>
  <div class="legend-row"><span class="legend-swatch" style="background:var(--decision-stop)"></span><span>決策：不可買</span></div>
</div>
""", unsafe_allow_html=True)


def render_settings_tab():
    """把 Token / API 診斷 / 回測費率／投組持股數 全部集中在這一個分頁，
    一般使用者完全不需要打開設定頁；日常只要使用「盤中即時」或「盤後深度」。"""
    st.subheader("🔑 FinMind Token")
    st.caption("免費註冊 FinMind 帳號即可取得 Token，額度會從 300 次/hr 提高到 600 次/hr。輸入後按下方按鈕套用並儲存到本機，下次開啟 App 會自動帶入。若要部署到雲端，請改用 secrets，不要把 Token 寫進程式碼或 Git。")
    st.markdown("🔗 [前往 FinMind 官網免費註冊 / 索取 Token](https://finmindtrade.com/analysis/#/data/api_token)")
    user_token_input = st.text_input(
        "輸入 FinMind Token (選填)",
        value=st.session_state.get("token_applied", ""),
        type="password",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✔️ 確認 Token / 套用並儲存"):
            _clean = user_token_input.strip()
            st.session_state["token_applied"] = _clean
            if _clean:
                save_token_to_disk(_clean)
                st.success("Token 已套用並儲存到本機，下次開啟 App 會自動帶入。")
            else:
                clear_saved_token()
            st.rerun()
    with c2:
        if st.button("🗑️ 清除已儲存 Token"):
            st.session_state["token_applied"] = ""
            clear_saved_token()
            st.success("已清除本機儲存的 Token。")
            st.rerun()
    with c3:
        if st.button("🧹 清除快取"):
            st.cache_data.clear()
            st.session_state["market_scan_out"] = None
            st.session_state["market_scan_candidates"] = None
            st.session_state["market_scan_top5"] = None
            st.session_state["market_scan_saved_at"] = None
            clear_saved_scan()
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
    c3, c4 = st.columns(2)
    with c3:
        st.session_state["cfg_hold_days"] = st.slider("📅 單股持有天數", 3, 30, st.session_state["cfg_hold_days"], 1, help="單股回測的 TIME 出場門檻。")
    with c4:
        st.session_state["cfg_scan_workers"] = st.slider("⚡ 平行掃描執行緒", 1, 6, st.session_state["cfg_scan_workers"], 1, help="提高速度也會增加同時 API 請求；遇到額度/連線問題可降回 1。")

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

    _flush_api_errors()
    if st.session_state["api_errors"]:
        st.caption("最近的錯誤紀錄：")
        for err in reversed(st.session_state["api_errors"][-10:]):
            st.text(f"[{err['time']}] {err['api']}({err['stock_id']}) → {err['error']}")
    else:
        st.caption("目前尚無錯誤紀錄。")


# =========================
# 9. Dashboard + Tabs
# =========================
# 這一區只讀快取/低成本資料，避免首頁載入就大量消耗 FinMind。
try:
    _dash_live = get_intraday_market_snapshot()
except Exception:
    _dash_live = pd.DataFrame()

_dash_up = int((_dash_live.get("即時漲跌%", pd.Series(dtype=float)) > 0).sum()) if not _dash_live.empty else 0
_dash_down = int((_dash_live.get("即時漲跌%", pd.Series(dtype=float)) < 0).sum()) if not _dash_live.empty else 0
_dash_flat = int((_dash_live.get("即時漲跌%", pd.Series(dtype=float)) == 0).sum()) if not _dash_live.empty else 0
_dash_turnover = pd.to_numeric(_dash_live.get("成交金額", pd.Series(dtype=float)), errors="coerce").sum() if not _dash_live.empty else np.nan
_dash_latest = "—"
try:
    _tw = get_yahoo_taiex()
    if _tw is not None and len(_tw):
        _dash_latest = f"{float(_tw.iloc[-1]):,.0f}"
except Exception:
    pass

st.markdown(f"""
<div class="market-overview">
  <div class="overview-card"><div class="overview-label">大盤位階</div><div class="overview-value">{regime.get('score',50):.0f}<span style="font-size:12px;color:var(--text-sub)"> / 100</span></div><div class="overview-sub {'tw-up' if regime.get('score',50)>=60 else 'tw-down'}">{regime.get('regime','UNKNOWN')} · {regime.get('message','')}</div></div>
  <div class="overview-card"><div class="overview-label">台股即時樣本</div><div class="overview-value">{len(_dash_live):,}</div><div class="overview-sub"><span class="tw-up">上漲 {_dash_up:,}</span> · <span class="tw-down">下跌 {_dash_down:,}</span></div></div>
  <div class="overview-card"><div class="overview-label">成交金額</div><div class="overview-value">{(_dash_turnover/1e8):,.0f}<span style="font-size:12px;color:var(--text-sub)"> 億</span></div><div class="overview-sub">交易所快照 · 免 FinMind</div></div>
  <div class="overview-card"><div class="overview-label">上漲 / 下跌</div><div class="overview-value"><span class="tw-up">{_dash_up:,}</span> / <span class="tw-down">{_dash_down:,}</span></div><div class="overview-sub">平盤 {_dash_flat:,}</div></div>
  <div class="overview-card"><div class="overview-label">^TWII 最新</div><div class="overview-value">{_dash_latest}</div><div class="overview-sub">Yahoo benchmark / 市場位階</div></div>
</div>
<div class="scanner-launch-grid">
  <div class="scanner-launch live"><div class="scanner-launch-title">⚡ 盤中即時掃描</div><div class="scanner-launch-sub">即時行情 → 全市場快速篩選 → 套用最近盤後基準買進分。原則上 0 FinMind。</div><span class="launch-badge">約 15 秒行情快取</span></div>
  <div class="scanner-launch eod"><div class="scanner-launch-title">🌙 盤後深度掃描</div><div class="scanner-launch-sub">完整 PIT 研究 → 基本面 × 估值 × 籌碼 × 技術 → 建立明日研究池。</div><span class="launch-badge">FinMind 深度研究</span></div>
</div>
""", unsafe_allow_html=True)

tab_intraday, tab_eod, tab_stock, tab_holdings, tab_verify, tab_help, tab_settings = st.tabs([
    "⚡ 盤中即時", "🌙 盤後深度", "🔍 股票分析", "🩺 庫存健康", "📜 歷史驗證", "📖 使用說明", "⚙️ 系統設定"
])

# --- TAB：使用說明 ---
with tab_help:
    st.subheader("📖 Quant Compass V10.2 實戰使用說明")
    st.caption("給第一次使用的人：不用懂量化，也能知道這個系統在做什麼、什麼時候按哪個按鈕。")

    st.info("""
    **每天只需要三步：**

    🌙 盤後深度：找明天值得研究的股票  
    ⚡ 盤中即時：確認市場是否真的有買盤  
    🧠 策略健康：查看這套方法長期是否有效

    系統不是預測股票，而是用固定規則找出條件較佳的股票。
    """)

    st.markdown("""
    <div class="terminal-grid">
      <div class="terminal-card"><div class="tc-label">你可以把它想成</div><div class="tc-value">台股研究助理</div><div class="tc-sub">每天幫你從市場找出值得研究的股票，不直接替你下單。</div></div>
      <div class="terminal-card"><div class="tc-label">主要問題</div><div class="tc-value">今天看誰？</div><div class="tc-sub">盤後找明日名單，盤中確認市場真的有沒有動能。</div></div>
      <div class="terminal-card"><div class="tc-label">核心分數</div><div class="tc-value">買進分</div><div class="tc-sub">越高代表目前條件越完整；不是勝率，也不是保證獲利。</div></div>
      <div class="terminal-card"><div class="tc-label">資料原則</div><div class="tc-value">PIT</div><div class="tc-sub">歷史驗證不使用當時尚未知道的未來資料。</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧭 第一步：每天晚上要做什麼？", expanded=True):
        st.markdown("""
        **按「🌙 盤後深度」→「執行盤後深度掃描」**

        系統會依序做：
        1. 從上市＋上櫃股票中建立掃描清單。
        2. 先看流動性與技術條件，避免把 FinMind 額度浪費在冷門股票。
        3. 從候選股中挑出前段班。
        4. 一次批次抓基本面、營收、估值、法人與日 K。
        5. 用同一套「買進分」排序。
        6. 產生「明日最值得看」與「明日可優先研究」名單。

        **簡單說：晚上是在回答「明天有哪些股票值得我看？」**
        """)

    with st.expander("⚡ 第二步：隔天盤中要做什麼？"):
        st.markdown("""
        **按「⚡ 盤中即時」→「一鍵掃描現在市場」**。

        盤中主要看即時行情、成交量、漲跌與動能，並套用昨晚留下的研究結果。
        它的目的不是重新做一遍財報研究，而是回答：

        **「昨晚看好的股票，今天市場真的有沒有在買？」**

        盤中模式原則上不重新打完整 FinMind 研究資料，所以可以比盤後更頻繁使用。
        """)

    with st.expander("🧠 第三步：買進分到底是什麼？"):
        st.markdown("""
        買進分是把多個條件整理成一個容易排序的分數，包含：

        **基本面 × 估值 × 籌碼 × 技術 × 突破 × 市場環境**。

        例如 90 分代表「目前條件整體很強」，80 分代表「條件也不錯」，但**90 分不是 90% 勝率**。
        所以實際操作還要一起看：**風險、資料品質、狀態、是否過熱、是否接近漲停**。
        """)

    with st.expander("🎨 第四步：台股顏色怎麼看？"):
        st.markdown("""
        **價格／報酬：** 🔴 紅色＝上漲、🟢 綠色＝下跌、⚪ 灰色＝平盤。

        **決策：** 🟢 可買、🟡 觀察、🔴 不買。

        **風險：** 🟢 低風險、🟡 中風險、🔴 高風險。

        所以看到紅色時，要先看它是「價格上漲」還是「高風險」；兩者的意義不同。
        """)

    
    with st.expander("🧠 第六步：策略健康檢查怎麼看？", expanded=False):
        st.markdown("""
        系統會自動保存每天盤中與盤後產生的選股訊號。

        後續會追蹤：
        - 買進後 5 日結果
        - 買進後 10 日結果
        - 買進後 20 日結果

        用來回答：

        **「這套買進分邏輯過去到底有沒有優勢？」**

        不需要重新掃描全市場，也不會大量消耗 FinMind Token。
        """)

    with st.expander("⚠️ 第五步：看到『可買』是不是就一定要買？"):
        st.markdown("""
        **不是。**「可買」代表模型認為條件同時成立的程度較高，不代表未來一定上漲。

        實戰建議依序確認：
        **買進分 → 風險 → 資料品質 → 是否過熱 → 盤中動能 → 最後才由你決定是否下單。**

        這套系統目前定位是**量化研究與交易決策輔助**，不是自動下單機器人。
        """)

    with st.expander("🔑 FinMind Token 是做什麼？怎麼省？"):
        st.markdown("""
        FinMind 主要負責盤後研究資料，例如日 K、營收、財報、PER/PBR、法人資料。

        V10 已經把盤後完整研究改成**批次抓取**：候選股票的完整資料一次分批取得，再逐檔計算分數，避免以前每一檔股票重複打 5 個 API。

        **建議：**有 Token 就放在「⚙️ 系統設定」；盤中掃描則盡量不消耗 FinMind。
        """)

    with st.expander("📊 最後：這個系統每天真正要看的只有什麼？"):
        st.markdown("""
        如果你不想看一堆數字，只看這 6 個：

        **① 買進分　② 狀態　③ 風險　④ 資料品質　⑤ 近 1／5／20 日漲跌　⑥ 盤中動能**

        其他指標都是「需要深入研究時再打開」。
        """)

# --- TAB：系統設定（程式碼故意寫在最前面執行，讓改設定當下就對其他分頁生效，
#     即使它在畫面上排在最後一個分頁也一樣）---
with tab_settings:
    st.subheader("⚙️ 系統設定")
    st.caption("Token、API 診斷、回測費率、投組持股數，都集中在這裡——不影響「盤中即時／盤後深度」的日常使用。")
    render_settings_tab()

    st.divider()
    with st.expander("📖 系統說明"):
        st.markdown("""
        **V10.0 最終版重點：**
        * **買進分是唯一的主分數。** 綜合分、起漲分等內部因子仍在計算，但只出現在「股票分析」的詳細分析裡，不會同時丟兩個分數給你。
        * **雙掃描器。** 盤中即時與盤後深度共用同一套核心研究引擎，但資料層分開；盤中優先 0 FinMind，盤後才做完整深度分析。
        * **全市場掃描是真的全市場。** 不再是股票代碼排序後直接切前 N 檔；優先用交易所官方 OpenAPI 的「今日全市場成交金額」快照（免費、不吃 FinMind 額度）排出最活躍的候選名單，抓不到快照時才退回全市場均勻隨機取樣。
        * **FinMind 額度更省。** 初篩後的完整研究改成批次抓取 5 個 dataset，不再每檔股票重打 5 次；日K／營收／財報／PER/PBR／法人買賣也保留快取。
        * **盤後更穩。** 多檔分析不再共用同一個 DataLoader；每個 worker 使用獨立 client，降低「18/18 分析完成但全部沒有有效資料」的情況。
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
hold_days = st.session_state["cfg_hold_days"]
scan_workers = st.session_state["cfg_scan_workers"]


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
        <div class="pick-sub">{row['決策']} ・ {row.get('狀態','')} ・ 風險 {row.get('風險','')} ・ 資料 {row.get('資料品質','—')} ・ 現價 {row['現價']} ・ 今日 <span class="{'tw-up' if safe_float(row.get('近1日漲跌%'),0)>0 else 'tw-down' if safe_float(row.get('近1日漲跌%'),0)<0 else 'tw-flat'}">{format_num(row.get('近1日漲跌%'), 1, '%')}</span> ・ 5日 <span class="{'tw-up' if safe_float(row.get('近5日漲跌%'),0)>0 else 'tw-down' if safe_float(row.get('近5日漲跌%'),0)<0 else 'tw-flat'}">{format_num(row.get('近5日漲跌%'), 1, '%')}</span> ・ 20日 <span class="{'tw-up' if safe_float(row.get('近20日漲跌%'),0)>0 else 'tw-down' if safe_float(row.get('近20日漲跌%'),0)<0 else 'tw-flat'}">{format_num(row.get('近20日漲跌%'), 1, '%')}</span></div>
        <div class="pick-sub">風險調整優先級：<b>{row.get('優先級', row.get('買進分', 0)):.0f}</b> / 100</div>
        <div class="pick-reason">{reasons_html}</div>
    </div>
    """, unsafe_allow_html=True)


def scan_column_config():
    cfg = {}
    if hasattr(st, "column_config"):
        cfg["買進分"] = st.column_config.ProgressColumn("買進分", min_value=0, max_value=100, format="%.0f", help="0–100 條件強度；不是未來報酬率或勝率。")
        cfg["優先級"] = st.column_config.ProgressColumn("風險調整優先級", min_value=0, max_value=100, format="%.0f", help="買進分扣除風險、市況與過熱懲罰後的研究排序分。")
        cfg["決策"] = st.column_config.TextColumn("決策", width="small")
        cfg["狀態"] = st.column_config.TextColumn("狀態", width="medium")
        cfg["風險"] = st.column_config.TextColumn("風險", width="small")
        cfg["資料品質"] = st.column_config.TextColumn("資料品質", width="small")
        cfg["近1日漲跌%"] = st.column_config.NumberColumn("近1日漲跌%", format="%.1f%%")
        cfg["近5日漲跌%"] = st.column_config.NumberColumn("近5日漲跌%", format="%.1f%%")
        cfg["近20日漲跌%"] = st.column_config.NumberColumn("近20日漲跌%", format="%.1f%%")
        cfg["量比"] = st.column_config.NumberColumn("量比", format="%.2fx")
        cfg["趨勢"] = st.column_config.LineChartColumn("20日趨勢", width="medium", help="近 20 個交易日累積相對走勢")
        cfg["現價"] = st.column_config.NumberColumn("現價", format="%.2f")
    return cfg


def show_scan_dataframe(df):
    if df is None or df.empty: return
    shown = df.copy()
    order = [c for c in ["名稱"] + MAIN_TABLE_COLS if c in shown.columns]
    shown = shown[order]
    st.dataframe(style_scan_table(shown), use_container_width=True, hide_index=True, column_config=scan_column_config())

MAIN_TABLE_COLS = ["股票代碼", "現價", "買進分", "優先級", "決策", "狀態", "風險", "資料品質", "近1日漲跌%", "近5日漲跌%", "近20日漲跌%", "量比", "漲停狀態", "趨勢", "說明"]

# --- TAB：盤中即時掃描 ---
with tab_intraday:
    st.subheader("⚡ 盤中即時掃描")
    st.caption("盤中模式只讀交易所行情快照 + 已保存的盤後研究結果；原則上 0 FinMind Token。每次重新整理約 15 秒更新一次行情。")
    u = get_stock_universe()
    if u.empty:
        st.error("無法取得股票清單，請到「⚙️ 系統設定」檢查 API。")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("市場股票", len(u))
        cached_eod = load_saved_scan()
        eod_time = cached_eod.get("saved_at", "尚無") if isinstance(cached_eod, dict) else "尚無"
        c2.metric("最近盤後研究", eod_time)
        c3.metric("FinMind 原則", "0 次")
        if st.button("⚡ 一鍵掃描現在市場", type="primary"):
            with st.spinner("連線交易所並掃描市場中…"):
                try:
                    live = run_intraday_scan(u, top_n=30)
                    save_intraday_scan(live)
                    st.session_state["intraday_scan_out"] = live
                    st.session_state["intraday_scan_saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                except Exception as exc:
                    _log_api_error("run_intraday_scan", "-", exc)
                    st.error("盤中行情取得失敗；請稍後再試。這次沒有呼叫 FinMind。")
        if st.session_state.get("intraday_scan_out") is None:
            cached_live = load_intraday_scan()
            if isinstance(cached_live, dict):
                st.session_state["intraday_scan_out"] = cached_live.get("out")
                st.session_state["intraday_scan_saved_at"] = cached_live.get("saved_at")
        live = st.session_state.get("intraday_scan_out")
        if isinstance(live, pd.DataFrame) and not live.empty:
            saved_at = st.session_state.get("intraday_scan_saved_at", "")
            if saved_at: st.caption(f"🕒 最近一次盤中掃描：{saved_at}")
            st.subheader("🔥 盤中最強候選")
            cols = [c for c in ["排名","股票代碼","名稱","即時價","即時漲跌%","近1日漲跌%","近5日漲跌%","成交金額","基準買進分","盤中動能分","即時調整分","盤中訊號","風險","狀態"] if c in live.columns]
            st.dataframe(style_intraday_table(live[cols]), use_container_width=True, hide_index=True)
            st.info("「基準買進分」來自最近一次盤後深度研究；「即時調整分」只用盤中行情做動態調整。因此盤中掃描不會重新呼叫完整 FinMind 研究資料。若尚無盤後結果，基準分暫以 50 計。")
        else:
            st.info("尚未執行盤中掃描。按上方「一鍵掃描現在市場」即可。")

# --- TAB：盤後深度掃描 ---
with tab_eod:
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

    regime_score = safe_float(regime.get("score"), 50)
    regime_color = "var(--accent-green)" if regime.get("regime") == "BULL" else ("var(--accent-yellow)" if regime.get("regime") == "NEUTRAL" else "var(--accent-red)")
    st.markdown(f"""
    <div class="terminal-grid">
      <div class="terminal-card"><div class="tc-label">MARKET REGIME</div><div class="tc-value" style="color:{regime_color};">{regime.get("regime","UNKNOWN")}</div><div class="tc-sub">{regime.get("message","")}</div></div>
      <div class="terminal-card"><div class="tc-label">MARKET SCORE</div><div class="tc-value">{regime_score:.0f}<span class="tc-unit">/100</span></div><div class="tc-sub">MA20 / MA60 · MACD · ADX</div></div>
      <div class="terminal-card"><div class="tc-label">DECISION RULE</div><div class="tc-value">多因子</div><div class="tc-sub">基本面 × 估值 × 籌碼 × 技術</div></div>
      <div class="terminal-card"><div class="tc-label">DATA POLICY</div><div class="tc-value">PIT</div><div class="tc-sub">歷史訊號不使用未來日期資料</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("盤後深度模式：完整更新研究資料，產生明日觀察名單。先看條件強度，再看風險與資料品質。買進分不是未來報酬率，也不是勝率。")

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

        cfg = effective_scan_config(strength_choice)
        scan_size = len(uni) if cfg["prefilter"] is None else min(cfg["prefilter"], len(uni))
        # 初篩每檔 1 次日 K；完整 shortlist 改成 5 個批次 dataset request。
        est_calls = scan_size + 5
        st.caption(f"「{strength_choice}」盤後先掃約 {scan_size} 檔，再取前 {cfg['topk']} 檔做完整分析；"
                   f"完整分析改用 5 個批次資料請求，不再逐檔重打 5 次 FinMind。")

        _quota_limit = 600 if st.session_state.get("token_applied") or has_finmind_secret() else 300
        if est_calls > _quota_limit:
            st.warning(f"⚠️ 預估本次約 {est_calls} 次 FinMind 請求，超過目前上限 {_quota_limit}。"
                       f"系統已自動限制無 Token 模式的標準/深度掃描；如需完整 400 檔初篩，請到「⚙️ 系統設定」套用 Token。")
        else:
            st.caption(f"（預估約 {est_calls} 次 API；完整分析採批次抓取，Token 消耗比舊版大幅降低。）")

        if st.button("🌙 執行盤後深度掃描", type="primary"):
            scan_list = build_scan_list(uni, strength_choice)
            pre_rows = []
            with st.status(f"🔎 正在執行市場掃描… 0/{len(scan_list)}", expanded=False) as scan_status:
                scan_status.write(f"預計掃描 {len(scan_list)} 檔，平行執行緒 {scan_workers}。")
                pre_results = []
                def _prefilter_one(sid):
                    return market_prefilter(sid)
                with ThreadPoolExecutor(max_workers=max(1, min(scan_workers, len(scan_list)))) as pool:
                    futures = {pool.submit(_prefilter_one, sid): sid for sid in scan_list}
                    done = 0
                    for future in as_completed(futures):
                        sid = futures[future]; done += 1
                        try:
                            r = future.result()
                            if r: pre_results.append(r)
                        except Exception as exc:
                            _log_api_error("market_prefilter", sid, exc)
                        scan_status.update(label=f"🔎 初篩中… {done}/{len(scan_list)}")
                _flush_api_errors()

                if pre_results:
                    pre_df = pd.DataFrame(pre_results).sort_values("初篩分", ascending=False)
                    st.caption(f"流動性過濾後剩 {len(pre_df)} / {len(scan_list)} 檔值得繼續分析。")
                    shortlist = pre_df.head(cfg["topk"])["股票代碼"].tolist()
                    final_rows = []
                    # ★ V10.1 穩定性修正：先一次批次抓完整研究資料，再逐檔使用同一個核心評分引擎。
                    with st.status(f"🧠 批次研究資料中… 0/5", expanded=False) as final_status:
                        batch_sources = prepare_pit_sources_batch(shortlist, 1500)
                        final_status.update(label="🧠 批次研究資料完成… 5/5")
                    _flush_api_errors()

                    with st.status(f"🧠 完整分析中… 0/{len(shortlist)}", expanded=False) as final_status:
                        for done, sid in enumerate(shortlist, start=1):
                            try:
                                sources = batch_sources.get(sid, {})
                                if not sources.get("daily", pd.DataFrame()).empty:
                                    result = calculate_stock_snapshot(sid, pd.Timestamp(datetime.now().date()), sources, regime)
                                else:
                                    result = None
                                if result:
                                    final_rows.append(result)
                                else:
                                    st.toast(f"⚠️ {sid} 無法產生完整分數，請看 API 診斷", icon="⚠️")
                            except Exception as exc:
                                _log_api_error("calculate_stock_snapshot", sid, exc)
                            final_status.update(label=f"🧠 完整分析中… {done}/{len(shortlist)}")
                    _flush_api_errors()

                    if final_rows:
                        out = pd.DataFrame(final_rows).sort_values("買進分", ascending=False)
                        name_map = universe_df.set_index("stock_id")["stock_name"].to_dict() if "stock_name" in universe_df.columns else {}
                        out.insert(1, "名稱", out["股票代碼"].map(name_map).fillna(""))
                        candidates = out[out["決策"].isin(["🟢 可買"])]
                        _saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        st.session_state["market_scan_out"] = out
                        st.session_state["market_scan_candidates"] = candidates
                        st.session_state["market_scan_top5"] = out.head(5)
                        st.session_state["market_scan_saved_at"] = _saved_at
                        # 存到本機檔案：就算關掉 App 或重開機，這份結果也會留著，
                        # 直到你下一次按「執行盤後深度掃描」才會被覆蓋掉。
                        save_scan_to_disk({
                            "out": out, "candidates": candidates, "top5": out.head(5), "saved_at": _saved_at
                        })
                        try:
                            pd.to_pickle({"out": out, "candidates": candidates, "top5": out.head(5), "saved_at": _saved_at}, EOD_SCAN_CACHE_FILE)
                        except Exception:
                            pass
                        append_research_snapshot(out, _saved_at, regime.get("regime"), regime.get("score"))
                    else:
                        st.error("完整分析階段沒有取得有效資料。請至「⚙️ 系統設定」檢查 API 診斷紀錄。")
                else:
                    st.error("初篩沒有取得任何有效資料（可能全數被流動性門檻濾掉）。請至「⚙️ 系統設定」檢查 API 診斷紀錄。")

        if st.session_state.get("market_scan_out") is not None:
            out_df = st.session_state["market_scan_out"]
            top5_df = st.session_state.get("market_scan_top5")
            _saved_at = st.session_state.get("market_scan_saved_at")
            if _saved_at:
                st.caption(f"🕓 目前顯示的是 {_saved_at} 的盤後深度掃描結果（重開 App 也不會消失，按「執行盤後深度掃描」才會更新）。")

            if top5_df is not None and not top5_df.empty:
                st.subheader("🔥 明日最值得看")
                for rank, (_, row) in enumerate(top5_df.iterrows(), start=1):
                    render_pick_card(row, rank)

            st.subheader("📋 盤後深度結果")
            st.caption("先看「買進分」判斷條件強度，再看「風險／資料品質／風險調整優先級」決定研究順序；不要把買進分直接當成勝率。")
            show_cols = ["名稱"] + MAIN_TABLE_COLS if "名稱" in out_df.columns else MAIN_TABLE_COLS
            show_scan_dataframe(out_df[show_cols])

            cands_df = st.session_state["market_scan_candidates"]
            st.subheader("🟢 明日可優先研究")
            if cands_df.empty:
                st.info("這次掃描沒有股票同時通過所有買進條件——今天先觀察就好。")
            else:
                cols2 = ["名稱"] + MAIN_TABLE_COLS if "名稱" in cands_df.columns else MAIN_TABLE_COLS
                show_scan_dataframe(cands_df[cols2])

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
                render_factor_bars(result)

    else:
        if st.button("🚀 掃描自選清單", type="primary"):
            if not stocks:
                st.error("請先在側邊欄加入至少一個有效的 4 碼股票代碼。")
            else:
                rows = []
                with st.status(f"🔎 自選股分析中… 0/{len(stocks)}", expanded=False) as wl_status:
                    with ThreadPoolExecutor(max_workers=max(1, min(scan_workers, len(stocks)))) as pool:
                        futures = {pool.submit(calculate_stock, stock, regime["regime"], regime): stock for stock in stocks}
                        done = 0
                        for future in as_completed(futures):
                            stock = futures[future]; done += 1
                            try:
                                result = future.result()
                                if result: rows.append(result)
                                else: st.toast(f"⚠️ {stock} 資料抓取失敗", icon="⚠️")
                            except Exception as exc:
                                _log_api_error("calculate_stock", stock, exc)
                                st.toast(f"⚠️ {stock} 分析失敗", icon="⚠️")
                            wl_status.update(label=f"🔎 自選股分析中… {done}/{len(stocks)}")
                _flush_api_errors()

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
            show_scan_dataframe(out)
            st.subheader("🎯 真正候選池")
            if cands.empty: st.info("目前自選股中沒有同時通過所有買進條件的標的。")
            else: show_scan_dataframe(cands)

# --- TAB：庫存健康 ---
with tab_holdings:
    st.subheader("🩺 庫存健康檢查")
    st.caption("輸入你實際持有的股票、股數與成本，AI 會依趨勢強度與波動度自動估算每一檔的動態停損防線與停利目標，一鍵檢查現在該停損、加碼、攤平還是出清。這份清單會存在本機，下次開啟自動帶回來。")

    with st.expander("🤖 AI 怎麼算停損／停利？（不想手動猜百分比，直接看這裡）", expanded=False):
        st.markdown("""
- **停損防線是「動態」的，不是死板固定 %**：用「近期高點 − ATR（真實波動幅度）× 趨勢係數」跟「成本價 × 你的下限%」兩者取比較不吃虧的一個。
  股票趨勢越強（ADX 越高），給的緩衝空間越大，避免正常回檔就被洗出場、錯過主升段；趨勢走弱或盤整時則自動收緊，虧損不會放給它擴大。
- **停利目標用「風險倍數」動態算，不用你猜要設多少 %**：目標價會隨這檔股票自己的波動度自動放大縮小。到第一目標建議先落袋一小部分，到第二目標建議了結一半左右，**剩餘部位改用移動停損顧著就好、不用全部出清**——這是為了解決「賣飛」的問題。
- **攤平只在條件仍偏多、且沒跌破 AI 停損防線時才會建議**，且會提醒「最多攤平一次、攤平後要重設停損」，避免掉進「無底洞式攤平」。
- 以上全部是量化規則＋技術指標算出來的參考建議，**不是投資建議**，最終判斷跟資金控管仍要自己做。
        """)

    st.markdown("##### 持有部位")
    _hh1, _hh2, _hh3, _hh4 = st.columns([2.2, 2, 2, 0.8])
    _hh1.markdown('<div class="row-editor-head">股票代碼</div>', unsafe_allow_html=True)
    _hh2.markdown('<div class="row-editor-head">持有股數</div>', unsafe_allow_html=True)
    _hh3.markdown('<div class="row-editor-head">持有成本（每股）</div>', unsafe_allow_html=True)
    _hh4.markdown('<div class="row-editor-head">&nbsp;</div>', unsafe_allow_html=True)

    _remove_holding_idx = None
    for _i, _row in enumerate(st.session_state["holdings_rows"]):
        with st.container(border=True):
            rc1, rc2, rc3, rc4 = st.columns([2.2, 2, 2, 0.8])
            _code_val = rc1.text_input("股票代碼", value=_row.get("code", ""), max_chars=4,
                                        key=f"h_code_{_row['id']}", label_visibility="collapsed",
                                        placeholder="4 碼代號")
            _shares_val = rc2.number_input("持有股數", value=float(_row.get("shares", 0) or 0), min_value=0.0,
                                            step=1000.0, format="%.0f", key=f"h_shares_{_row['id']}",
                                            label_visibility="collapsed")
            _cost_val = rc3.number_input("持有成本", value=float(_row.get("cost", 0.0) or 0.0), min_value=0.0,
                                          step=0.1, format="%.2f", key=f"h_cost_{_row['id']}",
                                          label_visibility="collapsed")
            _row["code"] = _code_val.strip()
            _row["shares"] = _shares_val
            _row["cost"] = _cost_val
            if rc4.button("🗑️", key=f"h_del_{_row['id']}", help="刪除這一列", use_container_width=True):
                _remove_holding_idx = _i

    if _remove_holding_idx is not None:
        st.session_state["holdings_rows"].pop(_remove_holding_idx)
        if not st.session_state["holdings_rows"]:
            st.session_state["holdings_rows"].append({"id": str(uuid.uuid4()), "code": "", "shares": 0.0, "cost": 0.0})
        st.rerun()

    if st.button("＋ 新增一檔持股", use_container_width=True):
        st.session_state["holdings_rows"].append({"id": str(uuid.uuid4()), "code": "", "shares": 0.0, "cost": 0.0})
        st.rerun()

    holdings_df = holdings_df_from_rows(st.session_state["holdings_rows"])

    st.markdown("##### 風險偏好")
    st.caption("決定「成本下限」這條保底防線；真正的停損／停利價仍會依每檔股票當下的趨勢與波動度動態調整。")
    profile_keys = list(RISK_PROFILE_PRESETS.keys())
    chosen_profile = st.radio("風險偏好", profile_keys, index=1, horizontal=True,
                               key="risk_profile_choice", label_visibility="collapsed")
    rp_cols = st.columns(3)
    for i, key in enumerate(profile_keys):
        preset = RISK_PROFILE_PRESETS[key]
        active_cls = "active" if key == chosen_profile else ""
        icon = "🛡️" if key == "保守" else ("⚖️" if key == "平衡" else "🚀")
        rp_cols[i].markdown(f"""
<div class="risk-profile-card {active_cls}">
    <div class="rp-title">{icon} {key}</div>
    <div class="rp-desc">{preset['desc']}</div>
    <div class="rp-num">成本下限 −{preset['base_stop_loss_pct']:.0f}%</div>
</div>
""", unsafe_allow_html=True)

    stop_loss_pct = RISK_PROFILE_PRESETS[chosen_profile]["base_stop_loss_pct"]
    with st.expander("⚙️ 進階：手動微調成本下限（一般不用調）", expanded=False):
        stop_loss_pct = st.slider("成本下限（%）", min_value=3, max_value=25, value=int(stop_loss_pct),
                                   help="未實現損益跌破這個百分比時，一定會被視為停損防線的下限，即使 AI 動態算出的緩衝更寬鬆也一樣。")

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        run_check = st.button("🩺 開始健康檢查", type="primary", use_container_width=True)
    with bcol2:
        if st.button("💾 儲存這份庫存清單", use_container_width=True):
            save_holdings_to_disk(holdings_df)
            st.success("庫存清單已儲存到本機。")

    if run_check:
        _clean_holdings = holdings_df.dropna(subset=["股票代碼"])
        _clean_holdings = _clean_holdings[_clean_holdings["股票代碼"].astype(str).str.strip() != ""]
        if _clean_holdings.empty:
            st.warning("請先在上面輸入至少一檔庫存（股票代碼、股數、成本）。")
        else:
            save_holdings_to_disk(holdings_df)
            results = []
            with st.status(f"🔎 正在檢查 {len(_clean_holdings)} 檔庫存…", expanded=False) as hstatus:
                for i, row in _clean_holdings.iterrows():
                    sid = str(row["股票代碼"]).strip().zfill(4)
                    cost = safe_float(row.get("持有成本"))
                    shares = safe_float(row.get("持有股數"), 0)
                    try:
                        stock_row = calculate_stock(sid, regime["regime"], regime)
                        if stock_row is None:
                            results.append({"股票代碼": sid, "操作建議": "⚠️ 查無資料", "理由": ["抓不到這檔股票的資料，請確認代碼是否正確。"]})
                        else:
                            r = evaluate_holding_action(stock_row, cost, shares, stop_loss_pct)
                            r["股票代碼"] = sid
                            results.append(r)
                    except Exception as exc:
                        _log_api_error("evaluate_holding_action", sid, exc)
                        results.append({"股票代碼": sid, "操作建議": "⚠️ 檢查失敗", "理由": ["這檔股票暫時檢查失敗，稍後再試一次。"]})
                    hstatus.update(label=f"🔎 正在檢查… {i+1}/{len(_clean_holdings)}")
                _flush_api_errors()
            st.session_state["holdings_health_res"] = results

    if st.session_state.get("holdings_health_res"):
        results = st.session_state["holdings_health_res"]
        action_order = {"🔻 建議停損": 0, "🚪 建議出清": 1, "🎯 建議獲利了結一部分": 2, "🎯 可考慮部分獲利了結": 3,
                         "➕ 可考慮攤平": 4, "📈 可考慮加碼": 5, "🤝 續抱觀察": 6,
                         "⚠️ 資料不足": 7, "⚠️ 查無資料": 7, "⚠️ 檢查失敗": 7}
        results_sorted = sorted(results, key=lambda r: action_order.get(r.get("操作建議"), 9))

        n_stop = sum(1 for r in results if r.get("操作建議") in ("🔻 建議停損", "🚪 建議出清"))
        n_profit = sum(1 for r in results if "獲利了結" in r.get("操作建議", ""))
        n_add = sum(1 for r in results if r.get("操作建議") in ("📈 可考慮加碼", "➕ 可考慮攤平"))
        conf_vals = [r.get("續抱信心分") for r in results if not pd.isna(r.get("續抱信心分", np.nan))]
        avg_conf = f"{np.mean(conf_vals):.0f}" if conf_vals else "—"

        st.markdown(f"""
<div class="stat-chip-row">
    <div class="stat-chip"><div class="sc-label">庫存檔數</div><div class="sc-value">{len(results)}</div></div>
    <div class="stat-chip"><div class="sc-label">⚠️ 需要注意（停損／出清）</div><div class="sc-value" style="color:var(--accent-red);">{n_stop}</div></div>
    <div class="stat-chip"><div class="sc-label">🎯 可考慮獲利了結</div><div class="sc-value" style="color:var(--accent-green);">{n_profit}</div></div>
    <div class="stat-chip"><div class="sc-label">📈 可考慮加碼／攤平</div><div class="sc-value" style="color:var(--accent-blue);">{n_add}</div></div>
</div>
""", unsafe_allow_html=True)

        for r in results_sorted:
            action = r.get("操作建議", "")
            if "停損" in action or "出清" in action:
                badge_bg, badge_border, badge_color = "rgba(255,69,58,0.15)", "var(--accent-red)", "var(--accent-red)"
            elif "獲利了結" in action or "加碼" in action or "攤平" in action:
                badge_bg, badge_border, badge_color = "rgba(48,209,88,0.15)", "var(--accent-green)", "var(--accent-green)"
            elif "續抱" in action:
                badge_bg, badge_border, badge_color = "rgba(10,132,255,0.15)", "var(--accent-blue)", "var(--accent-blue)"
            else:
                badge_bg, badge_border, badge_color = "rgba(255,214,10,0.15)", "var(--accent-yellow)", "var(--accent-yellow)"

            price = r.get("現價", np.nan)
            pnl = r.get("損益%", np.nan)
            has_price = not pd.isna(price) if price is not None else False

            meta_bits = []
            if has_price:
                meta_bits.append(f"現價 <b>{price:.2f}</b>")
            if pnl is not None and not pd.isna(pnl):
                pnl_color = "var(--accent-green)" if pnl >= 0 else "var(--accent-red)"
                meta_bits.append(f"損益 <b style='color:{pnl_color};'>{pnl:+.1f}%</b>")
            mv = r.get("市值", np.nan)
            if mv is not None and not pd.isna(mv):
                meta_bits.append(f"市值 <b>{mv:,.0f}</b>")
            upl = r.get("未實現損益", np.nan)
            if upl is not None and not pd.isna(upl):
                upl_color = "var(--accent-green)" if upl >= 0 else "var(--accent-red)"
                meta_bits.append(f"未實現損益 <b style='color:{upl_color};'>{upl:+,.0f}</b>")
            meta_html = "　·　".join(meta_bits)

            confidence = r.get("續抱信心分", np.nan)
            confidence_html = ""
            if confidence is not None and not pd.isna(confidence):
                cf_color = "var(--accent-green)" if confidence >= 70 else ("var(--accent-yellow)" if confidence >= 40 else "var(--accent-red)")
                confidence_html = f"""
<div class="confidence-row">
    <div class="confidence-label">續抱信心分</div>
    <div class="confidence-track"><div class="confidence-fill" style="width:{confidence:.0f}%; background:{cf_color};"></div></div>
    <div class="confidence-num" style="color:{cf_color};">{confidence:.0f}</div>
</div>"""

            targets_html = ""
            stop_v, t1, t2 = r.get("建議停損價", np.nan), r.get("目標價1", np.nan), r.get("目標價2", np.nan)
            if any(v is not None and not pd.isna(v) for v in [stop_v, t1, t2]):
                def _box(label, v):
                    val_str = f"{v:.2f}" if (v is not None and not pd.isna(v)) else "—"
                    return f'<div class="price-target-box"><div class="pt-label">{label}</div><div class="pt-value">{val_str}</div></div>'
                targets_html = f"""
<div class="price-target-row">
    {_box("🛑 AI 停損防線", stop_v)}
    {_box("🎯 目標價 1（1.5R）", t1)}
    {_box("🎯 目標價 2（3R）", t2)}
</div>"""

            reasons_html = "".join(f"<li>{x}</li>" for x in r.get("理由", []))

            st.markdown(f"""
<div class="holding-card">
    <div class="hc-top">
        <div class="hc-name">{r.get('股票代碼','')}<span class="hc-badge" style="background:{badge_bg}; border-color:{badge_border}; color:{badge_color};">{action}</span></div>
    </div>
    <div class="hc-meta">{meta_html}</div>
    {confidence_html}
    {targets_html}
    <ul class="hc-reasons">{reasons_html}</ul>
</div>
""", unsafe_allow_html=True)

        st.caption("以上為量化規則參考建議（依買進分、趨勢強度 ADX、波動度 ATR 與你選擇的風險偏好動態計算），不是投資建議，實際操作請自行判斷並留意資金控管。")

with tab_verify:
    st.caption("策略健康檢查：追蹤每日實際產生的選股訊號，確認買進分是否在真實市場中有效。")
    sub_strategy = st.container()

    def build_equity_benchmark_figure(result, title="資產曲線 vs Benchmark"):
        eq = result.get("equity", pd.Series(dtype=float))
        fig = go.Figure()
        if eq is not None and len(eq):
            base = float(eq.iloc[0])
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy / Portfolio", line=dict(width=2.5)))
            benches = result.get("benchmarks", {}) or {}
            for name, series in benches.items():
                b = pd.Series(series).reindex(eq.index).ffill().dropna()
                if len(b) > 1 and b.iloc[0] != 0:
                    norm = b / b.iloc[0] * base
                    label = "^TWII / 大盤" if name == "^TWII" else "0050" if name == "0050.TW" else name
                    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=label, line=dict(dash="dot")))
        fig.update_layout(title=title, template="plotly_dark", height=430, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
        return fig

    def build_backtest_technical_figure(result):
        from plotly.subplots import make_subplots
        daily = result.get("daily", pd.DataFrame()).copy()
        eq = result.get("equity", pd.Series(dtype=float))
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.40,0.25,0.18,0.17], subplot_titles=("資產曲線 vs Benchmark", "價格 / MA20 / MA60", "MACD", "KD"))
        if eq is not None and len(eq):
            fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Strategy",line=dict(width=2.5)),row=1,col=1)
            for name, series in (result.get("benchmarks", {}) or {}).items():
                b=pd.Series(series).reindex(eq.index).ffill().dropna()
                if len(b)>1 and b.iloc[0]!=0:
                    label="^TWII / 大盤" if name=="^TWII" else "0050" if name=="0050.TW" else name
                    fig.add_trace(go.Scatter(x=b.index,y=b/b.iloc[0]*float(eq.iloc[0]),mode="lines",name=label,line=dict(dash="dot")),row=1,col=1)
        if not daily.empty:
            dd=daily.tail(260)
            fig.add_trace(go.Scatter(x=dd["date"],y=dd["close"],name="Close",mode="lines"),row=2,col=1)
            for col in ["MA20","MA60"]:
                if col in dd: fig.add_trace(go.Scatter(x=dd["date"],y=dd[col],name=col,mode="lines"),row=2,col=1)
            if "MACD" in dd:
                hist=(dd["MACD"]-dd["MACD_signal"]).fillna(0)
                fig.add_trace(go.Bar(x=dd["date"],y=hist,name="MACD Hist",opacity=0.55),row=3,col=1)
                fig.add_trace(go.Scatter(x=dd["date"],y=dd["MACD"],name="MACD",mode="lines"),row=3,col=1)
                fig.add_trace(go.Scatter(x=dd["date"],y=dd["MACD_signal"],name="Signal",mode="lines"),row=3,col=1)
            if "K" in dd and "D" in dd:
                fig.add_trace(go.Scatter(x=dd["date"],y=dd["K"],name="K",mode="lines"),row=4,col=1)
                fig.add_trace(go.Scatter(x=dd["date"],y=dd["D"],name="D",mode="lines"),row=4,col=1)
        fig.update_layout(title=f"{result.get('stock','')} 回測與技術面", template="plotly_dark", height=1000, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
        return fig

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
                with st.status(f"📉 {selected} Point-in-Time 回測中…", expanded=False):
                    st.session_state["single_backtest_res"]=backtest_single(selected,initial_capital,fee,tax,slippage,hold_days=hold_days)
            result=st.session_state.get("single_backtest_res")
            if result:
                metric_grid(result); ai_explain(result)
                bc=st.columns(4)
                bc[0].metric("^TWII 超額", f"{result.get('alpha_TWII',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_TWII',np.nan)) else "N/A")
                bc[1].metric("0050 超額", f"{result.get('alpha_0050_TW',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_0050_TW',np.nan)) else "N/A")
                bc[2].metric("Beta / TWII", f"{result.get('beta_TWII',np.nan):.2f}" if not pd.isna(result.get('beta_TWII',np.nan)) else "N/A")
                bc[3].metric("持有天數", f"{result.get('hold_days',hold_days)} 天")
                fig=build_backtest_technical_figure(result)
                st.plotly_chart(fig,use_container_width=True)
                with st.expander("📋 交易明細"): st.dataframe(pd.DataFrame(result.get("trades_detail",[])),use_container_width=True,hide_index=True)
                with st.expander("🧠 AI 研究摘要"):
                    st.write("這個策略不是單看技術訊號，而是用目前系統的買進分與市場位階做歷史判斷；每個歷史日只使用當日以前的資料。")

    with sub_portfolio:
        st.subheader("💼 投組回測：真實成本 + 換股成本")
        if len(stocks)>=2 and st.button("💼 執行投資組合回測",type="primary"):
            with st.status("💼 計算共同資金池、換股與交易成本…", expanded=False):
                st.session_state["portfolio_backtest_res"]=portfolio_backtest(stocks,initial_capital,top_n,fee,tax,slippage,max_workers=scan_workers)
        result=st.session_state.get("portfolio_backtest_res")
        if result:
            metric_grid(result); ai_explain(result,"投組解讀")
            bc=st.columns(2); bc[0].metric("^TWII 超額", f"{result.get('alpha_TWII',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_TWII',np.nan)) else "N/A"); bc[1].metric("0050 超額", f"{result.get('alpha_0050_TW',np.nan)*100:.2f}%" if not pd.isna(result.get('alpha_0050_TW',np.nan)) else "N/A")
            fig=build_equity_benchmark_figure(result, title="共同資金池資產曲線 vs Benchmark")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("投組換股時會依實際權重變化估算手續費、滑價與賣出證交稅；Benchmark 以相同期間起始值標準化，方便直接比較。")

    with sub_wf:
        st.subheader("🧪 Walk-Forward / Out-of-Sample")
        st.markdown("**目的：** 不讓同一段歷史同時扮演訓練與驗證角色。現在會依訓練年數建立 rolling 時間切點，測試區間完全封存；本版不在訓練區自動調參，因此不把報告式訓練期冒充成最佳化結果。")
        if st.button("🧪 執行 OOS 驗證",type="primary"):
            with st.spinner("執行多標的 OOS 驗證..."):
                st.session_state["wf_res"]=walk_forward_test(stocks,initial_capital,fee,tax,slippage,hold_days=hold_days)
        wf=st.session_state.get("wf_res")
        if wf is not None and not wf.empty:
            st.dataframe(wf,use_container_width=True,hide_index=True)
            st.success("OOS 報表完成。訓練區只用於建立時間切點，測試區完全獨立；本版不自動最佳化參數，避免把資料探勘結果誤當成真實 OOS。")

    with sub_strategy:
        st.subheader("🧠 策略健康檢查")
        st.caption("此功能不重新掃描全市場，而是使用每日盤後/盤中留下的訊號紀錄，追蹤這套買進分邏輯在真實市場中的表現。")

        log = load_research_log()

        if log.empty:
            st.info("目前沒有歷史訊號紀錄。請先執行盤後深度掃描，系統會逐日保存選股結果，未來自動驗證。")
        else:
            st.success(f"目前累積訊號：{len(log):,} 筆")

            if st.button("🔍 執行策略健康檢查", type="primary"):
                with st.status("正在分析歷史訊號後續表現…", expanded=False):
                    detail, cal = build_forward_calibration(log, max_samples=len(log))
                    st.session_state["health_detail"] = detail
                    st.session_state["health_table"] = cal

            detail = st.session_state.get("health_detail", pd.DataFrame())
            cal = st.session_state.get("health_table", pd.DataFrame())

            if detail is not None and not detail.empty:

                st.markdown("### 📊 這套選股邏輯過去表現")

                c1, c2, c3 = st.columns(3)

                for col, days in zip([c1, c2, c3], ["5D", "10D", "20D"]):
                    win_col = f"{days}勝率"
                    ret_col = f"{days}平均報酬"

                    if win_col in detail.columns:
                        win = detail[win_col].mean()
                        ret = detail[ret_col].mean() if ret_col in detail.columns else 0

                        col.metric(
                            f"{days} 後",
                            f"勝率 {win:.1f}%",
                            f"平均 {ret:.2f}%"
                        )

                st.markdown("### 🎯 買進分是否有效")

                if cal is not None and not cal.empty:
                    keep_cols = [
                        c for c in [
                            "分數區間",
                            "樣本數",
                            "10D勝率",
                            "10D平均報酬",
                            "20D勝率",
                            "20D平均報酬"
                        ]
                        if c in cal.columns
                    ]

                    if keep_cols:
                        st.dataframe(
                            cal[keep_cols].round(2),
                            use_container_width=True,
                            hide_index=True
                        )

                st.markdown("### 🚦 目前策略狀態")

                drift = strategy_drift_report(detail)

                if drift.get("status") == "DRIFT":
                    st.error("🔴 策略可能失效，需要重新檢查市場環境")
                elif drift.get("status") == "WATCH":
                    st.warning("🟡 策略需要觀察")
                else:
                    st.success("🟢 策略目前維持正常")

                with st.expander("📋 查看詳細訊號紀錄"):
                    st.dataframe(
                        detail.sort_values("訊號日", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("尚未建立驗證資料，請先累積盤後掃描訊號。")


# footer
st.divider()
st.caption("台股量化羅盤 Quant Compass V10.2 Strategy Health Final · Smart Real-Time Scanner · 台股標準配色：紅漲綠跌 · Point-in-Time · Unified Buy Score · Realistic Costs · Benchmark · OOS Framework")
