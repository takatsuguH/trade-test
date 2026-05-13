"""株式自動取引ダッシュボード — Streamlit アプリ"""
import sys
sys.path.insert(0, ".")

import uuid
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from src.data.fetcher import get_stock_data
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended, calculate_short_signals
from src.strategies.composite import generate_composite_signal, merge_all_signals
from src.risk.manager import RiskManager
from src.backtest import run_backtest
from src.analysis.correlation import analyze_correlations, INDICATOR_META
from src.indicators.signal_generator import generate_ext_signals, build_ext_composite
from src.optimization.searcher import find_best_combination
from src.db import storage as db
import json as _json

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="株式自動取引ダッシュボード",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

db.init_db()

# ─────────────────────────────────────────────
# ダークテーマ カスタムCSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ベースカラー */
:root {
    --bg-primary: #080c14;
    --bg-secondary: #0d1321;
    --bg-card: #111827;
    --bg-card-hover: #1a2236;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-red: #ff4060;
    --accent-orange: #ff8c00;
    --text-primary: #e8eaf0;
    --text-secondary: #8892a4;
    --border-color: #1e2d40;
    --glow-cyan: 0 0 8px rgba(0, 212, 255, 0.4);
    --glow-green: 0 0 8px rgba(0, 255, 136, 0.4);
}

/* アプリ全体背景 */
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0, 212, 255, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0, 255, 136, 0.03) 0%, transparent 50%) !important;
}

/* メインコンテンツエリア */
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
}

/* サイドバー */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

/* ヘッダー・テキスト */
h1, h2, h3 {
    color: var(--text-primary) !important;
    letter-spacing: 0.03em;
}
h1 {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}
p, label, .stMarkdown {
    color: var(--text-secondary) !important;
}

/* メトリクスカード */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ボタン */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    box-shadow: var(--glow-cyan) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 255, 136, 0.15)) !important;
    border-color: var(--accent-green) !important;
    color: var(--accent-green) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.35), rgba(0, 255, 136, 0.25)) !important;
    box-shadow: var(--glow-green) !important;
}

/* 入力フォーム */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* セレクトボックス */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* スライダー */
.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: var(--border-color) !important;
}

/* チェックボックス */
.stCheckbox > label > div:first-child {
    border-color: var(--accent-cyan) !important;
}
.stCheckbox > label > div[data-checked="true"] {
    background: var(--accent-cyan) !important;
}

/* タブ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 1px solid var(--border-color) !important;
    gap: 4px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 255, 136, 0.1)) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* エクスパンダー */
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}
.stExpander > div > div > div > summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* データフレーム */
.stDataFrame {
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* 区切り線 */
hr {
    border-color: var(--border-color) !important;
    opacity: 0.5 !important;
}

/* アラートボックス */
.stAlert {
    border-radius: 8px !important;
    border-left-width: 3px !important;
}

/* プログレスバー */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
}

/* スピナー */
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}

/* サイドバーヘッダー */
[data-testid="stSidebar"] h2 {
    color: var(--accent-cyan) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    -webkit-text-fill-color: var(--accent-cyan) !important;
}

/* キャプション */
.stCaptionContainer, .stCaption {
    color: var(--text-secondary) !important;
    opacity: 0.7;
}

/* トグル */
.stToggle > label > div {
    background: var(--border-color) !important;
}
.stToggle > label > div[data-checked="true"] {
    background: var(--accent-cyan) !important;
}

/* スクロールバー */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)

st.title("📈 株式自動取引ダッシュボード")

# ─────────────────────────────────────────────
# 会社名辞書（主要日本株）+ yfinance フォールバック
# ─────────────────────────────────────────────
_JP_NAMES: dict[str, str] = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "6861.T": "キーエンス",
    "7974.T": "任天堂",
    "4063.T": "信越化学工業",
    "8306.T": "三菱UFJフィナンシャルG",
    "6098.T": "リクルートHD",
    "4519.T": "中外製薬",
    "8035.T": "東京エレクトロン",
    "6367.T": "ダイキン工業",
    "9432.T": "日本電信電話（NTT）",
    "7751.T": "キヤノン",
    "4502.T": "武田薬品工業",
    "6501.T": "日立製作所",
    "6702.T": "富士通",
    "6752.T": "パナソニックHD",
    "7267.T": "本田技研工業",
    "6301.T": "コマツ",
    "9433.T": "KDDI",
    "4661.T": "オリエンタルランド",
    "8058.T": "三菱商事",
    "8031.T": "三井物産",
    "7011.T": "三菱重工業",
    "9022.T": "JR東海",
    "8766.T": "東京海上HD",
    "6723.T": "ルネサスエレクトロニクス",
    "6954.T": "ファナック",
    "9735.T": "セコム",
    "9983.T": "ファーストリテイリング",
    "4568.T": "第一三共",
    "8411.T": "みずほフィナンシャルG",
    "8316.T": "三井住友フィナンシャルG",
    "9020.T": "JR東日本",
    "7832.T": "バンダイナムコHD",
    "6645.T": "オムロン",
    "4543.T": "テルモ",
    "2914.T": "日本たばこ産業（JT）",
    "3382.T": "セブン＆アイHD",
    "8001.T": "伊藤忠商事",
    # 米国株（英語のまま）
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta",
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_name(ticker: str) -> str:
    """会社名を返す。辞書にあれば即返し、なければyfinanceで取得（英語）。"""
    code = ticker.strip().upper()
    if code in _JP_NAMES:
        return _JP_NAMES[code]
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or "不明"
    except Exception:
        return "?"


@st.cache_data(ttl=3600, show_spinner=False)
def get_short_data(ticker: str) -> dict:
    """
    yfinance から空売り情報を取得する。
    shortRatio     : 空売りカバー日数（売り残 / 平均日次出来高）
    shortPercentOfFloat : 浮動株に対する空売り比率（%）
    日本株では取得できない場合が多いため available フラグで判定する。
    """
    try:
        info = yf.Ticker(ticker).info
        sr = info.get("shortRatio")
        sp = info.get("shortPercentOfFloat")
        if sp is not None:
            sp = round(sp * 100, 2)
        return {
            "short_ratio": sr,
            "short_pct_float": sp,
            "available": sr is not None or sp is not None,
        }
    except Exception:
        return {"short_ratio": None, "short_pct_float": None, "available": False}


# ─────────────────────────────────────────────
# 追加指標グループ定義（サイドバー用）
# ─────────────────────────────────────────────
_SIDEBAR_EXTRA_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("モメンタム系", [
        ("ストキャスティクス %K", "STOCH_K"),
        ("ストキャスティクス %D", "STOCH_D"),
        ("CCI（商品チャネル）", "CCI"),
        ("ウィリアムズ %R", "WILLIAMS_R"),
        ("ROC（変化率）", "ROC"),
        ("モメンタム", "MOM"),
        ("アルティメットオシレーター", "UO"),
        ("CMO", "CMO"),
    ]),
    ("トレンド系", [
        ("ADX（方向性指数）", "ADX"),
        ("+DI（上昇方向性）", "PLUS_DI"),
        ("-DI（下降方向性）", "MINUS_DI"),
        ("アルーンオシレーター", "AROON_OSC"),
        ("アルーン上昇", "AROON_UP"),
        ("アルーン下降", "AROON_DOWN"),
        ("TRIX", "TRIX"),
        ("DPO（デトレンド）", "DPO"),
        ("マスインデックス", "MASS_INDEX"),
        ("コポックカーブ", "COPPOCK"),
    ]),
    ("移動平均・価格系", [
        ("DEMA（二重EMA）", "DEMA"),
        ("TEMA（三重EMA）", "TEMA"),
        ("HMA（ハル移動平均）", "HMA"),
        ("VWAP", "VWAP"),
        ("パラボリックSAR", "PSAR"),
        ("一目 転換線", "ICHI_TENKAN"),
        ("一目 基準線", "ICHI_KIJUN"),
        ("一目 先行スパンA", "ICHI_SPAN_A"),
        ("一目 先行スパンB", "ICHI_SPAN_B"),
    ]),
    ("バンド系", [
        ("ケルトナー中心線", "KC_MID"),
        ("ケルトナー上限", "KC_UPPER"),
        ("ケルトナー下限", "KC_LOWER"),
        ("ドンチャン中心", "DC_MID"),
        ("ドンチャン上限", "DC_UPPER"),
        ("ドンチャン下限", "DC_LOWER"),
    ]),
    ("出来高・ボラ系", [
        ("OBV（出来高加重）", "OBV"),
        ("MFI（資金流量指数）", "MFI"),
        ("CMF（チャイキン資金流）", "CMF"),
        ("フォースインデックス", "FORCE_INDEX"),
        ("EOM（移動容易性）", "EOM"),
        ("ATR（真の値幅）", "ATR"),
    ]),
    ("空売り・スクイーズ", [
        ("空売り圧力指数", "SELL_PRESSURE"),
        ("スクイーズスコア", "SQUEEZE_SCORE"),
    ]),
]

# ─────────────────────────────────────────────
# 拡張指標パラメータ仕様（col_name → [(key, label, type, min, max, step)]）
# type: "int" or "float"
# ─────────────────────────────────────────────
_COL_PARAMS: dict[str, list[tuple]] = {
    "STOCH_K":     [("stoch_k", "K期間", "int", 2, 50, 1),
                    ("stoch_d", "D期間", "int", 2, 20, 1)],
    "STOCH_D":     [],
    "CCI":         [("cci_period", "期間", "int", 5, 100, 1)],
    "WILLIAMS_R":  [("williams_period", "期間", "int", 5, 50, 1)],
    "ROC":         [("roc_period", "期間", "int", 2, 50, 1)],
    "MOM":         [("mom_period", "期間", "int", 2, 50, 1)],
    "UO":          [("uo_p1", "短期", "int", 2, 20, 1),
                    ("uo_p2", "中期", "int", 5, 30, 1),
                    ("uo_p3", "長期", "int", 10, 60, 1)],
    "CMO":         [("cmo_period", "期間", "int", 5, 50, 1)],
    "ADX":         [("adx_period", "期間", "int", 5, 50, 1)],
    "PLUS_DI":     [],
    "MINUS_DI":    [],
    "AROON_OSC":   [("aroon_period", "期間", "int", 5, 100, 1)],
    "AROON_UP":    [],
    "AROON_DOWN":  [],
    "TRIX":        [("trix_period", "期間", "int", 5, 50, 1)],
    "DPO":         [("dpo_period", "期間", "int", 5, 100, 1)],
    "MASS_INDEX":  [("mass_fast", "Fast EMA", "int", 2, 20, 1),
                    ("mass_slow", "ウィンドウ", "int", 10, 50, 1)],
    "COPPOCK":     [("coppock_roc1", "ROC1", "int", 5, 30, 1),
                    ("coppock_roc2", "ROC2", "int", 5, 30, 1),
                    ("coppock_wma", "WMA", "int", 3, 20, 1)],
    "DEMA":        [("dema_period", "期間", "int", 5, 100, 1)],
    "TEMA":        [("tema_period", "期間", "int", 5, 100, 1)],
    "HMA":         [("hma_period", "期間", "int", 5, 100, 1)],
    "VWAP":        [],
    "PSAR":        [("psar_af",   "AF開始",   "float", 0.01, 0.1,  0.01),
                    ("psar_step", "AFステップ", "float", 0.01, 0.1,  0.01),
                    ("psar_max",  "AF最大",    "float", 0.1,  0.5,  0.05)],
    "ICHI_TENKAN": [("ichi_tenkan",   "転換線",     "int", 5,  20,  1),
                    ("ichi_kijun",    "基準線",     "int", 10, 50,  1),
                    ("ichi_senkou_b", "先行スパンB", "int", 30, 100, 1)],
    "ICHI_KIJUN":  [],
    "ICHI_SPAN_A": [],
    "ICHI_SPAN_B": [],
    "KC_MID":      [("keltner_period", "期間", "int",   5,   50,  1),
                    ("keltner_mult",   "乗数", "float", 1.0, 4.0, 0.5)],
    "KC_UPPER":    [],
    "KC_LOWER":    [],
    "DC_MID":      [("donchian_period", "期間", "int", 5, 100, 1)],
    "DC_UPPER":    [],
    "DC_LOWER":    [],
    "OBV":         [],
    "MFI":         [("mfi_period", "期間", "int", 5, 50, 1)],
    "CMF":         [("cmf_period", "期間", "int", 5, 50, 1)],
    "FORCE_INDEX": [("force_period", "期間", "int", 2, 30, 1)],
    "EOM":         [("eom_period", "期間", "int", 5, 50, 1)],
    "ATR":         [("atr_period", "期間", "int", 5, 50, 1)],
    "SELL_PRESSURE": [],
    "SQUEEZE_SCORE": [],
}


# ─────────────────────────────────────────────
# ヘルパー：銘柄別設定の取得・構築
# ─────────────────────────────────────────────
def _get_ticker_settings(ticker: str) -> dict:
    """セッション状態（ライブ値）または DB から銘柄の設定を取得する。"""
    pfx = f"cfg_{ticker}"
    if st.session_state.get(f"_cfg_init_{ticker}"):
        s: dict = {}
        for k, v in db.DEFAULT_SETTINGS.items():
            if k != "extra_checked":
                s[k] = st.session_state.get(f"{pfx}_{k}", v)
        s["extra_checked"] = [
            col_name
            for _, grp_items in _SIDEBAR_EXTRA_GROUPS
            for _, col_name in grp_items
            if st.session_state.get(f"{pfx}_ext_{col_name}", False)
        ]
        return s
    loaded = db.load_settings(ticker)
    return {**db.DEFAULT_SETTINGS, **loaded}


def _build_indicator_config(s: dict) -> dict:
    return {
        "use_ma":    s["use_ma"],    "ma_short":   s["ma_short"],  "ma_long":  s["ma_long"],
        "use_rsi":   s["use_rsi"],   "rsi_period": s["rsi_period"], "rsi_ob":  s["rsi_ob"],  "rsi_os": s["rsi_os"],
        "use_macd":  s["use_macd"],  "macd_fast":  s["macd_fast"],  "macd_slow": s["macd_slow"], "macd_sig": s["macd_sig"],
        "use_bb":    s["use_bb"],    "bb_period":  s["bb_period"],  "bb_std":  s["bb_std"],
    }


def _build_active_indicators(s: dict) -> list[str]:
    return (
        (["MA"]   if s["use_ma"]   else []) +
        (["RSI"]  if s["use_rsi"]  else []) +
        (["MACD"] if s["use_macd"] else []) +
        (["BB"]   if s["use_bb"]   else [])
    )


def _build_ext_params(s: dict) -> dict:
    return {
        "stochastic":          {"k": s["stoch_k"],        "d": s["stoch_d"]},
        "cci":                 {"period": s["cci_period"]},
        "williams_r":          {"period": s["williams_period"]},
        "roc":                 {"period": s["roc_period"]},
        "momentum":            {"period": s["mom_period"]},
        "adx":                 {"period": s["adx_period"]},
        "atr":                 {"period": s["atr_period"]},
        "mfi":                 {"period": s["mfi_period"]},
        "cmf":                 {"period": s["cmf_period"]},
        "dema":                {"period": s["dema_period"]},
        "tema":                {"period": s["tema_period"]},
        "hma":                 {"period": s["hma_period"]},
        "aroon":               {"period": s["aroon_period"]},
        "trix":                {"period": s["trix_period"]},
        "dpo":                 {"period": s["dpo_period"]},
        "force_index":         {"period": s["force_period"]},
        "eom":                 {"period": s["eom_period"]},
        "keltner":             {"period": s["keltner_period"], "mult": s["keltner_mult"]},
        "donchian":            {"period": s["donchian_period"]},
        "psar":                {"af_start": s["psar_af"], "af_step": s["psar_step"], "af_max": s["psar_max"]},
        "ultimate_oscillator": {"p1": s["uo_p1"], "p2": s["uo_p2"], "p3": s["uo_p3"]},
        "cmo":                 {"period": s["cmo_period"]},
        "ichimoku":            {"tenkan": s["ichi_tenkan"], "kijun": s["ichi_kijun"], "senkou_b": s["ichi_senkou_b"]},
        "mass_index":          {"fast": s["mass_fast"], "slow": s["mass_slow"]},
        "coppock":             {"roc1": s["coppock_roc1"], "roc2": s["coppock_roc2"], "wma_period": s["coppock_wma"]},
    }


# ─────────────────────────────────────────────
# サイドバー設定
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("銘柄設定")

    # ── 動的銘柄リスト（DB連携） ──
    if "ticker_items" not in st.session_state:
        codes = db.load_stocks()
        if not codes:
            codes = ["7203.T", "6758.T", "9984.T"]
            for _c in codes:
                db.add_stock(_c)
        st.session_state.ticker_items = [
            {"id": str(uuid.uuid4()), "code": _c} for _c in codes
        ]

    st.caption("日本株は末尾に .T を付けてください")
    remove_id = None
    for item in st.session_state.ticker_items:
        uid = item["id"]
        col_code, col_name, col_del = st.columns([3, 5, 1])

        code = col_code.text_input(
            uid,
            value=item["code"],
            key=f"ti_{uid}",
            label_visibility="collapsed",
            placeholder="例: 7203.T",
        )
        new_code = code.strip()
        if new_code != item["code"]:
            if item["code"]:
                db.remove_stock(item["code"])
            if new_code:
                db.add_stock(new_code)
            item["code"] = new_code

        if item["code"]:
            name = get_company_name(item["code"])
            col_name.markdown(
                f"<div style='padding-top:6px;font-size:0.78rem;color:#aaa'>{name}</div>",
                unsafe_allow_html=True,
            )

        if col_del.button("✕", key=f"td_{uid}", help="削除"):
            remove_id = uid

    if remove_id:
        removed = next((x for x in st.session_state.ticker_items if x["id"] == remove_id), None)
        if removed and removed["code"]:
            db.remove_stock(removed["code"])
        st.session_state.ticker_items = [
            x for x in st.session_state.ticker_items if x["id"] != remove_id
        ]
        st.rerun()

    if st.button("＋ 銘柄を追加", use_container_width=True):
        st.session_state.ticker_items.append({"id": str(uuid.uuid4()), "code": ""})
        st.rerun()

    tickers = [item["code"] for item in st.session_state.ticker_items if item["code"]]

    period = st.selectbox(
        "取得期間",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
        format_func=lambda x: {"1mo": "1ヶ月", "3mo": "3ヶ月", "6mo": "6ヶ月", "1y": "1年", "2y": "2年"}[x],
    )

    st.divider()
    st.header("テクニカル指標設定")

    # ── 設定を編集する銘柄を選択 ──
    _settings_opts = tickers if tickers else [""]
    settings_ticker = st.selectbox(
        "設定を編集する銘柄",
        options=_settings_opts,
        format_func=lambda x: (f"{x}  {get_company_name(x)}" if x else "(銘柄なし)"),
    )

    if settings_ticker:
        pfx = f"cfg_{settings_ticker}"
        _init_key = f"_cfg_init_{settings_ticker}"

        # 初回のみ DB から session_state へ展開
        if not st.session_state.get(_init_key):
            _s0 = db.load_settings(settings_ticker)
            _s0 = {**db.DEFAULT_SETTINGS, **_s0}
            for _k, _v in db.DEFAULT_SETTINGS.items():
                if _k != "extra_checked":
                    _sk = f"{pfx}_{_k}"
                    if _sk not in st.session_state:
                        st.session_state[_sk] = _s0.get(_k, _v)
            _ec0 = _s0.get("extra_checked", [])
            for _, _grp in _SIDEBAR_EXTRA_GROUPS:
                for _, _cn in _grp:
                    _ek = f"{pfx}_ext_{_cn}"
                    if _ek not in st.session_state:
                        st.session_state[_ek] = _cn in _ec0
            st.session_state[_init_key] = True

        # ── メイン4指標 ──
        _use_ma = st.checkbox("移動平均（MA）", key=f"{pfx}_use_ma")
        if _use_ma:
            _c1, _c2 = st.columns(2)
            _c1.number_input("短期", min_value=2,  max_value=50,  step=1, key=f"{pfx}_ma_short")
            _c2.number_input("長期", min_value=5,  max_value=200, step=1, key=f"{pfx}_ma_long")

        _use_rsi = st.checkbox("RSI", key=f"{pfx}_use_rsi")
        if _use_rsi:
            st.slider("RSI 期間", 5, 30, key=f"{pfx}_rsi_period")
            _c1, _c2 = st.columns(2)
            _c1.number_input("買われすぎ", min_value=60, max_value=90, step=1, key=f"{pfx}_rsi_ob")
            _c2.number_input("売られすぎ", min_value=10, max_value=40, step=1, key=f"{pfx}_rsi_os")

        _use_macd = st.checkbox("MACD", key=f"{pfx}_use_macd")
        if _use_macd:
            _c1, _c2, _c3 = st.columns(3)
            _c1.number_input("Fast",   min_value=3,  max_value=50,  step=1, key=f"{pfx}_macd_fast")
            _c2.number_input("Slow",   min_value=5,  max_value=100, step=1, key=f"{pfx}_macd_slow")
            _c3.number_input("Signal", min_value=2,  max_value=30,  step=1, key=f"{pfx}_macd_sig")

        _use_bb = st.checkbox("ボリンジャーバンド", key=f"{pfx}_use_bb")
        if _use_bb:
            _c1, _c2 = st.columns(2)
            _c1.number_input("BB期間",  min_value=5,   max_value=50,  step=1,   key=f"{pfx}_bb_period")
            _c2.number_input("標準偏差", min_value=1.0, max_value=3.0, step=0.5,
                             format="%.1f", key=f"{pfx}_bb_std")

        # ── 追加指標（27種）＋ パラメータ設定 ──
        st.divider()
        with st.expander("📊 追加テクニカル指標（27種）"):
            st.caption("指標をチェックするとパラメータを設定できます")
            for _grp_name, _grp_items in _SIDEBAR_EXTRA_GROUPS:
                st.markdown(f"**{_grp_name}**")
                for _label, _col_name in _grp_items:
                    _meta = INDICATOR_META.get(_col_name, {})
                    _is_checked = st.checkbox(
                        _label,
                        key=f"{pfx}_ext_{_col_name}",
                        help=_meta.get("desc", ""),
                    )
                    if _is_checked:
                        _params_spec = _COL_PARAMS.get(_col_name, [])
                        if _params_spec:
                            _n = len(_params_spec)
                            _pcols = st.columns(_n)
                            for _pi, (_pk, _plbl, _ptyp, _pmn, _pmx, _pstep) in enumerate(_params_spec):
                                _wk = f"{pfx}_{_pk}"
                                if _ptyp == "float":
                                    _pcols[_pi].number_input(
                                        _plbl,
                                        min_value=float(_pmn), max_value=float(_pmx),
                                        step=float(_pstep), format="%.2f", key=_wk,
                                    )
                                else:
                                    _pcols[_pi].number_input(
                                        _plbl,
                                        min_value=int(_pmn), max_value=int(_pmx),
                                        step=int(_pstep), key=_wk,
                                    )

        # ── 保存ボタン ──
        if st.button("💾 この銘柄の設定を保存", key=f"_save_{settings_ticker}",
                     use_container_width=True, type="primary"):
            _save_s = {
                _k: st.session_state.get(f"{pfx}_{_k}", db.DEFAULT_SETTINGS[_k])
                for _k in db.DEFAULT_SETTINGS if _k != "extra_checked"
            }
            _save_s["extra_checked"] = [
                _cn
                for _, _grp in _SIDEBAR_EXTRA_GROUPS
                for _, _cn in _grp
                if st.session_state.get(f"{pfx}_ext_{_cn}", False)
            ]
            db.save_settings(settings_ticker, _save_s)
            for _ck in list(st.session_state.keys()):
                if (_ck.startswith(f"df_ext_{settings_ticker}_") or
                        _ck.startswith(f"corr_cache_{settings_ticker}_") or
                        _ck.startswith(f"opt_{settings_ticker}_")):
                    del st.session_state[_ck]
            st.success("設定を保存しました！")

    st.divider()
    st.header("リスク管理")
    initial_cash = st.number_input("初期資金（円）", value=1_000_000, step=100_000, min_value=100_000)
    stop_loss = st.slider("損切りライン（%）", 1, 30, 5)
    take_profit = st.slider("利確ライン（%）", 1, 50, 10)
    max_pos = st.slider("最大投資割合（%）", 10, 100, 100)
    rebuy_dip = st.slider(
        "買い戻し下落率（%）", 0, 20, 0,
        help="売却後、売値よりこの%以上下がったときのみ買い戻す。0=シグナルが出次第即時再エントリー",
    )

    st.divider()
    st.header("自動更新")
    auto_refresh = st.toggle("リアルタイム更新", value=False)
    refresh_sec = st.select_slider(
        "更新間隔",
        options=[30, 60, 120, 300],
        value=60,
        format_func=lambda x: f"{x}秒",
    )

# 自動更新（ページ全体を再実行）
if auto_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

risk = RiskManager(
    stop_loss_pct=stop_loss,
    take_profit_pct=take_profit,
    max_position_pct=max_pos,
    rebuy_dip_pct=rebuy_dip,
)


# ─────────────────────────────────────────────
# チャート生成関数
# ─────────────────────────────────────────────
# 拡張指標のチャート用パレット（最大20色）
_EXT_COLORS = [
    "#e91e63", "#00bcd4", "#8bc34a", "#ff5722", "#9c27b0",
    "#03a9f4", "#ffeb3b", "#795548", "#607d8b", "#f44336",
    "#4caf50", "#ff9800", "#673ab7", "#009688", "#ffc107",
    "#3f51b5", "#cddc39", "#ff4081", "#00e5ff", "#76ff03",
]


def create_chart(
    df: pd.DataFrame,
    ticker: str,
    ind_cfg: dict,
    ext_overlays: list[str] | None = None,
    ext_oscillators: list[str] | None = None,
) -> go.Figure:
    ext_overlays = [c for c in (ext_overlays or []) if c in df.columns]
    ext_oscillators = [c for c in (ext_oscillators or []) if c in df.columns]

    # ── サブプロット構成 ──
    subplot_rows = 1
    row_heights = [0.5]
    subplot_titles = [f"{ticker} 株価チャート"]

    if ind_cfg.get("use_rsi") and "RSI" in df.columns:
        subplot_rows += 1
        row_heights.append(0.15)
        subplot_titles.append("RSI")

    if ind_cfg.get("use_macd") and "MACD" in df.columns:
        subplot_rows += 1
        row_heights.append(0.15)
        subplot_titles.append("MACD")

    for col in ext_oscillators:
        meta = INDICATOR_META.get(col, {})
        subplot_rows += 1
        row_heights.append(0.15)
        subplot_titles.append(meta.get("name", col))

    # 高さ正規化
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=subplot_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── ローソク足 ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="株価",
        increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)

    # ── 移動平均（既存） ──
    if ind_cfg.get("use_ma") and "MA_short" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_short"],
            name=f"MA({ind_cfg['ma_short']})",
            line=dict(color="#ff9800", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_long"],
            name=f"MA({ind_cfg['ma_long']})",
            line=dict(color="#2196f3", width=1.5)), row=1, col=1)

    # ── ボリンジャーバンド（既存） ──
    if ind_cfg.get("use_bb") and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB上限",
            line=dict(color="rgba(156,39,176,0.4)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB下限",
            line=dict(color="rgba(156,39,176,0.4)", width=1),
            fill="tonexty", fillcolor="rgba(156,39,176,0.05)", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB中心",
            line=dict(color="rgba(156,39,176,0.6)", width=1, dash="dot")), row=1, col=1)

    # ── 拡張オーバーレイ（価格チャートに重ねる） ──
    for i, col in enumerate(ext_overlays):
        meta = INDICATOR_META.get(col, {})
        color = _EXT_COLORS[i % len(_EXT_COLORS)]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=meta.get("name", col),
            line=dict(color=color, width=1.5, dash="dot"),
        ), row=1, col=1)

    # ── 買い・売りシグナルマーカー ──
    buys = df[df["order"] > 0]
    sells = df[df["order"] < 0]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"] * 0.99, mode="markers", name="買いシグナル",
            marker=dict(symbol="triangle-up", color="#ff5252", size=12,
                        line=dict(width=1, color="white")),
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.01, mode="markers", name="売りシグナル",
            marker=dict(symbol="triangle-down", color="#40c4ff", size=12,
                        line=dict(width=1, color="white")),
        ), row=1, col=1)

    current_row = 2

    # ── RSIパネル ──
    if ind_cfg.get("use_rsi") and "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#9c27b0", width=1.5)), row=current_row, col=1)
        fig.add_hrect(y0=ind_cfg["rsi_ob"], y1=100,
            fillcolor="rgba(239,83,80,0.1)", line_width=0, row=current_row, col=1)
        fig.add_hrect(y0=0, y1=ind_cfg["rsi_os"],
            fillcolor="rgba(38,166,154,0.1)", line_width=0, row=current_row, col=1)
        fig.add_hline(y=ind_cfg["rsi_ob"],
            line=dict(color="#ef5350", width=1, dash="dash"), row=current_row, col=1)
        fig.add_hline(y=ind_cfg["rsi_os"],
            line=dict(color="#26a69a", width=1, dash="dash"), row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # ── MACDパネル ──
    if ind_cfg.get("use_macd") and "MACD" in df.columns:
        hist_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=hist_colors, opacity=0.7), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#2196f3", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal",
            line=dict(color="#ff9800", width=1.5)), row=current_row, col=1)
        current_row += 1

    # ── 拡張オシレーターパネル（1指標1行） ──
    for i, col in enumerate(ext_oscillators):
        meta = INDICATOR_META.get(col, {})
        color = _EXT_COLORS[i % len(_EXT_COLORS)]
        rng = meta.get("range")

        if col in ("OBV", "FORCE_INDEX", "EOM", "MOM", "ATR", "COPPOCK"):
            # バー表示が見やすい指標
            vals = df[col].fillna(0)
            bar_colors = [color if v >= 0 else "#607d8b" for v in vals]
            fig.add_trace(go.Bar(x=df.index, y=vals, name=meta.get("name", col),
                marker_color=bar_colors, opacity=0.8), row=current_row, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=meta.get("name", col),
                line=dict(color=color, width=1.5)), row=current_row, col=1)

        # 固定レンジの指標は基準線を追加
        if rng:
            mid = (rng[0] + rng[1]) / 2
            fig.add_hline(y=mid, line=dict(color="gray", width=0.8, dash="dot"),
                          row=current_row, col=1)
            fig.update_yaxes(range=[rng[0], rng[1]], row=current_row, col=1)

        current_row += 1

    fig.update_layout(
        height=max(500, 420 + subplot_rows * 130),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def create_comparison_chart(curve1: pd.DataFrame, curve2: pd.DataFrame,
                            label1: str, label2: str, initial_cash: float) -> go.Figure:
    """2つのバックテストのポートフォリオ推移を重ねて表示する。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve1["date"], y=curve1["value"],
        name=label1, line=dict(color="#2196f3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=curve2["date"], y=curve2["value"],
        name=label2, line=dict(color="#ff9800", width=2),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.05)",
    ))
    fig.add_hline(y=initial_cash, line=dict(color="gray", dash="dash"), annotation_text="初期資金")
    fig.update_layout(
        height=280, template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        title="ポートフォリオ推移 比較",
        yaxis_tickformat=",",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def create_portfolio_chart(portfolio_curve: pd.DataFrame, initial_cash: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_curve["date"], y=portfolio_curve["value"],
        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
        line=dict(color="#2196f3", width=2),
        name="ポートフォリオ",
    ))
    fig.add_hline(y=initial_cash, line=dict(color="gray", dash="dash"), annotation_text="初期資金")
    fig.update_layout(
        height=200, template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        title="ポートフォリオ推移",
        yaxis_tickformat=",",
    )
    return fig


# ─────────────────────────────────────────────
# メインコンテンツ（銘柄タブ）
# ─────────────────────────────────────────────
if not tickers:
    st.warning("サイドバーで銘柄コードを入力してください。")
    st.stop()

tabs = st.tabs([f"📊 {t}" for t in tickers])

for tab, ticker in zip(tabs, tickers):
    with tab:
        # 銘柄別設定を取得
        _s = _get_ticker_settings(ticker)
        ic = _build_indicator_config(_s)
        active = _build_active_indicators(_s)
        ext_params = _build_ext_params(_s)
        extra_cols = _s.get("extra_checked", [])
        extra_overlays = [c for c in extra_cols
                          if INDICATOR_META.get(c, {}).get("type") == "overlay"]
        extra_oscillators = [c for c in extra_cols
                             if INDICATOR_META.get(c, {}).get("type") != "overlay"]

        with st.spinner(f"{ticker} のデータを取得中..."):
            try:
                df = get_stock_data(ticker, period=period)
            except Exception as e:
                st.error(f"データ取得エラー: {e}")
                continue

        # 指標計算 → 複合シグナル生成
        df = calculate_all(df, ic)
        df = calculate_short_signals(df)

        df_ext = None
        if extra_cols:
            _ext_sig = _json.dumps(ext_params, sort_keys=True)
            _ext_sig_key = f"df_ext_sig_{ticker}_{period}"
            ext_df_key = f"df_ext_{ticker}_{period}"
            if st.session_state.get(_ext_sig_key) != _ext_sig or ext_df_key not in st.session_state:
                st.session_state[ext_df_key] = calculate_extended(df, ext_params)
                st.session_state[_ext_sig_key] = _ext_sig
            df_ext = st.session_state[ext_df_key]
            for _col in df_ext.columns:
                if _col not in df.columns:
                    df[_col] = df_ext[_col]
            df, _ext_sig_cols = generate_ext_signals(df, extra_cols)
            df = merge_all_signals(df, active, _ext_sig_cols)
        else:
            df = generate_composite_signal(df, active)

        # バックテスト実行
        result = run_backtest(df, initial_cash=initial_cash, risk=risk)

        # ─── メトリクス ───
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        price_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

        signal_label = {1: "🟢 買いシグナル", -1: "🔴 売りシグナル", 0: "⚪ ホールド"}
        current_signal = int(latest.get("composite_signal", 0))
        vote = int(latest.get("vote_sum", 0))
        trade_count = len(result["trades"])
        ret_pct = result["total_return_pct"]

        company_name = get_company_name(ticker)
        st.markdown(f"### {company_name}　<span style='font-size:0.9rem;color:#aaa'>{ticker}</span>",
                    unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("現在株価", f"{latest['Close']:,.1f}円", f"{price_change:+.2f}%")
        total_vote_count = len(active) + len(extra_cols)
        col2.metric("シグナル", signal_label[current_signal], f"投票: {vote:+d}/{total_vote_count}")
        col3.metric("リターン", f"{ret_pct:.2f}%",
                    f"{result['final_value'] - result['initial_cash']:+,.0f}円")
        col4.metric("最大DD", f"{result['max_drawdown_pct']:.2f}%")
        col5.metric("勝率", f"{result['win_rate_pct']:.1f}%")
        col6.metric("取引回数", f"{trade_count}回")

        # ─── 空売り・スクイーズ分析 ───
        sell_pressure_now = float(df["SELL_PRESSURE"].iloc[-1]) if "SELL_PRESSURE" in df.columns else 0.0
        squeeze_now       = float(df["SQUEEZE_SCORE"].iloc[-1])  if "SQUEEZE_SCORE" in df.columns else 0.0
        has_short_alert   = sell_pressure_now > 0.65 or squeeze_now > 0.55

        with st.expander("📉 空売り・スクイーズ分析", expanded=has_short_alert):
            sd = get_short_data(ticker)

            sc1, sc2, sc3, sc4 = st.columns(4)

            # yfinance 空売りデータ
            if sd["available"]:
                sr_val = sd["short_ratio"]
                sp_val = sd["short_pct_float"]
                sc1.metric(
                    "空売りカバー日数",
                    f"{sr_val:.1f} 日" if sr_val else "—",
                    "スクイーズリスク高" if sr_val and sr_val > 5 else ("中程度" if sr_val and sr_val > 2 else None),
                )
                sc2.metric(
                    "空売り比率（浮動株比）",
                    f"{sp_val:.1f}%" if sp_val else "—",
                    "高水準" if sp_val and sp_val > 10 else None,
                )
            else:
                sc1.metric("空売りカバー日数", "データなし", help="米国株のみ取得可能")
                sc2.metric("空売り比率", "データなし")

            # 技術的シグナル
            pressure_label = "危険" if sell_pressure_now > 0.65 else ("警戒" if sell_pressure_now > 0.4 else "正常")
            squeeze_label  = "高" if squeeze_now > 0.55 else ("中" if squeeze_now > 0.35 else "低")
            sc3.metric("空売り圧力", f"{sell_pressure_now:.2f}", pressure_label)
            sc4.metric("スクイーズスコア", f"{squeeze_now:.2f}", f"発生確率: {squeeze_label}")

            # アラートメッセージ
            if sell_pressure_now > 0.65:
                st.warning(
                    "⚠️ **空売り圧力が高い状態です。** "
                    "大量売りを伴う下落が続いており、機関投資家による空売り積み増しの可能性があります。"
                    " 新規買いは慎重に。既存保有株の損切りラインを確認してください。"
                )
            if squeeze_now > 0.55:
                st.success(
                    "🚀 **ショートスクイーズの兆候を検知しました。** "
                    "下落トレンド後の大量買いが急増しています。"
                    " 空売り勢の踏み上げによる急騰が起こりやすい局面です。買いチャンスの可能性があります。"
                )
            if sd["available"] and sd.get("short_ratio") and sd["short_ratio"] > 5:
                st.info(
                    f"ℹ️ 空売りカバー日数 {sd['short_ratio']:.1f} 日 — "
                    "売り残が多く、価格が上昇すると踏み上げが加速する可能性があります。"
                )

            # SELL_PRESSURE / SQUEEZE_SCORE のチャート
            if "SELL_PRESSURE" in df.columns and "SQUEEZE_SCORE" in df.columns:
                fig_ss = go.Figure()
                fig_ss.add_trace(go.Scatter(
                    x=df.index, y=df["SELL_PRESSURE"],
                    name="空売り圧力", line=dict(color="#f44336", width=1.5),
                ))
                fig_ss.add_trace(go.Scatter(
                    x=df.index, y=df["SQUEEZE_SCORE"],
                    name="スクイーズスコア", line=dict(color="#4caf50", width=1.5),
                ))
                fig_ss.add_hline(y=0.65, line=dict(color="#f44336", dash="dash", width=1),
                                 annotation_text="売り圧閾値")
                fig_ss.add_hline(y=0.55, line=dict(color="#4caf50", dash="dash", width=1),
                                 annotation_text="スクイーズ閾値")
                fig_ss.update_layout(
                    height=200, template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="空売り圧力 / スクイーズスコア（0〜1）",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_ss, width="stretch")

            st.caption(
                "空売り圧力：大出来高を伴う下落の累積強度（0〜1）。"
                "　スクイーズスコア：下落後の急騰＋大出来高の強度（0〜1）。"
                "　空売りカバー日数・空売り比率はyfinanceから取得（主に米国株対応）。"
            )

        # ─── 相関アラート（取引10回超 かつ リターンマイナス） ───
        ext_overlays_on: list[str] = list(extra_overlays)
        ext_oscillators_on: list[str] = list(extra_oscillators)
        corr_results: list[dict] = []

        if trade_count > 10 and ret_pct < 0:
            st.error(
                f"⚠️ アラート: 取引回数 {trade_count}回 でリターンがマイナス ({ret_pct:.2f}%) です。"
                " 現在の指標との相関が低い可能性があります。",
                icon="🚨",
            )

            # ── Step1: 相関分析（初回のみ実行してキャッシュ） ──
            cache_key = f"corr_cache_{ticker}_{period}"
            if cache_key not in st.session_state:
                with st.spinner("25種類以上の指標で相関分析中..."):
                    df_ext_for_corr = df_ext if df_ext is not None else calculate_extended(df, ext_params)
                    cr = analyze_correlations(df_ext_for_corr, forward_days=5, min_corr=0.15)
                st.session_state[cache_key] = {"df_ext": df_ext_for_corr, "corr": cr}

            cached = st.session_state[cache_key]
            df_ext = cached["df_ext"]
            corr_results = cached["corr"]

            if corr_results:
                candidate_cols = [r["col"] for r in corr_results]
                overlays_meta    = [r for r in corr_results if r["type"] == "overlay"]
                oscillators_meta = [r for r in corr_results if r["type"] != "overlay"]
                overlay_col_set  = {r["col"] for r in overlays_meta}

                # ── Step2: 最良組み合わせ探索（初回のみ実行してキャッシュ） ──
                opt_key = f"opt_{ticker}_{period}"
                if opt_key not in st.session_state:
                    prog_bar = st.progress(0, text="最良の指標組み合わせを探索中...")
                    def _cb(v):
                        prog_bar.progress(min(v, 0.99), text=f"探索中... {v*100:.0f}%")
                    best_cols, best_ret, method = find_best_combination(
                        df_ext, candidate_cols, progress_cb=_cb
                    )
                    prog_bar.empty()
                    st.session_state[opt_key] = {
                        "cols": best_cols, "ret": best_ret, "method": method
                    }

                opt = st.session_state[opt_key]
                rec_cols: list[str] = opt["cols"]
                rec_ret: float      = opt["ret"]
                rec_method: str     = opt["method"]

                # ── Step3: 推奨チェックボックスの初期値をセット（初回のみ） ──
                applied_key = f"rec_applied_{ticker}_{period}"
                if not st.session_state.get(applied_key):
                    for col in rec_cols:
                        ov_key  = f"ext_ov_{ticker}_{col}"
                        osc_key = f"ext_osc_{ticker}_{col}"
                        if col in overlay_col_set:
                            st.session_state.setdefault(ov_key, True)
                        else:
                            st.session_state.setdefault(osc_key, True)
                    st.session_state[applied_key] = True

                # ── 推奨バナー ──
                rec_names = [INDICATOR_META.get(c, {}).get("name", c) for c in rec_cols]
                rec_names_str = "　+　".join(rec_names)
                sign = "+" if rec_ret >= 0 else ""
                st.success(
                    f"✨ **推奨の組み合わせ** （{rec_method}）\n\n"
                    f"**{rec_names_str}**\n\n"
                    f"→ 推定リターン **{sign}{rec_ret:.2f}%**　"
                    f"（現在 {ret_pct:.2f}% から **{rec_ret - ret_pct:+.2f}%** 改善）",
                )

                # ── 推奨に戻すボタン ──
                if st.button("⭐ 推奨の組み合わせに戻す", key=f"restore_rec_{ticker}"):
                    # 全チェックをクリアして推奨だけをオンに
                    for r in corr_results:
                        col = r["col"]
                        st.session_state.pop(f"ext_ov_{ticker}_{col}", None)
                        st.session_state.pop(f"ext_osc_{ticker}_{col}", None)
                    for col in rec_cols:
                        if col in overlay_col_set:
                            st.session_state[f"ext_ov_{ticker}_{col}"] = True
                        else:
                            st.session_state[f"ext_osc_{ticker}_{col}"] = True
                    st.rerun()

                def corr_badge(c: float) -> str:
                    a = abs(c)
                    return "🔴 強" if a >= 0.4 else ("🟡 中" if a >= 0.25 else "🟢 弱")

                st.subheader("📊 将来リターンと相関が高い指標")
                st.caption("⭐ = 推奨組み合わせに含まれる指標")

                if overlays_meta:
                    st.markdown("**価格チャートに重ねるタイプ（移動平均系）**")
                    cols3 = st.columns(3)
                    for i, rec in enumerate(overlays_meta):
                        col  = rec["col"]
                        key  = f"ext_ov_{ticker}_{col}"
                        star = " ⭐" if col in rec_cols else ""
                        direction = "上昇連動" if rec["corr"] > 0 else "下落連動"
                        label = f"{rec['name']}{star}  {corr_badge(rec['corr'])} {rec['corr']:+.3f} ({direction})"
                        if cols3[i % 3].checkbox(label, key=key, help=rec["desc"]):
                            ext_overlays_on.append(col)
                            if col in df_ext.columns:
                                df[col] = df_ext[col]

                if oscillators_meta:
                    st.markdown("**別パネルで表示するタイプ（オシレーター）**")
                    cols3 = st.columns(3)
                    for i, rec in enumerate(oscillators_meta):
                        col  = rec["col"]
                        key  = f"ext_osc_{ticker}_{col}"
                        star = " ⭐" if col in rec_cols else ""
                        direction = "上昇連動" if rec["corr"] > 0 else "下落連動"
                        label = f"{rec['name']}{star}  {corr_badge(rec['corr'])} {rec['corr']:+.3f} ({direction})"
                        if cols3[i % 3].checkbox(label, key=key, help=rec["desc"]):
                            ext_oscillators_on.append(col)
                            if col in df_ext.columns:
                                df[col] = df_ext[col]

                # ── バックテスト実行ボタン ──
                all_checked = ext_overlays_on + ext_oscillators_on
                st.divider()
                run_col, clear_col, reset_col, _ = st.columns([2, 1, 1, 2])

                if run_col.button(
                    f"▶ 選択した {len(all_checked)} 指標でバックテスト実行",
                    key=f"run_ext_{ticker}",
                    disabled=(len(all_checked) == 0),
                    type="primary",
                ):
                    with st.spinner("バックテスト実行中..."):
                        df_bt, sig_cols = generate_ext_signals(df_ext, all_checked)
                        df_bt = build_ext_composite(df_bt, sig_cols)
                        ext_result = run_backtest(
                            df_bt, initial_cash, risk,
                            signal_col="ext_composite_signal",
                            order_col="ext_order",
                        )
                    st.session_state[f"ext_bt_{ticker}"] = {
                        "result": ext_result,
                        "checked": all_checked,
                        "names": [INDICATOR_META.get(c, {}).get("name", c) for c in all_checked],
                        "is_recommended": set(all_checked) == set(rec_cols),
                    }

                if clear_col.button("クリア", key=f"clear_ext_{ticker}"):
                    st.session_state.pop(f"ext_bt_{ticker}", None)
                    st.rerun()

                if reset_col.button("再探索", key=f"reopt_{ticker}",
                                    help="期間・設定変更後に再探索する"):
                    for k in [cache_key, opt_key, applied_key]:
                        st.session_state.pop(k, None)
                    st.rerun()

            else:
                st.info("相関しきい値(0.15)以上の指標は見つかりませんでした。")

        # ─── 拡張バックテスト結果 ───
        ext_bt_state = st.session_state.get(f"ext_bt_{ticker}")
        if ext_bt_state:
            er = ext_bt_state["result"]
            names_str = "、".join(ext_bt_state["names"][:5])
            if len(ext_bt_state["names"]) > 5:
                names_str += f" 他{len(ext_bt_state['names'])-5}件"

            is_rec = ext_bt_state.get("is_recommended", False)
            badge = " ⭐ 推奨組み合わせ" if is_rec else ""
            st.subheader(f"🔁 拡張バックテスト結果（{names_str}）{badge}")

            # 比較メトリクス
            c1, c2, c3, c4 = st.columns(4)
            delta_ret = er["total_return_pct"] - ret_pct
            delta_dd  = er["max_drawdown_pct"] - result["max_drawdown_pct"]
            delta_win = er["win_rate_pct"] - result["win_rate_pct"]
            delta_tr  = len(er["trades"]) - trade_count

            c1.metric("リターン（新）",  f"{er['total_return_pct']:.2f}%",  f"{delta_ret:+.2f}% vs 元")
            c2.metric("最大DD（新）",    f"{er['max_drawdown_pct']:.2f}%",  f"{delta_dd:+.2f}% vs 元")
            c3.metric("勝率（新）",      f"{er['win_rate_pct']:.1f}%",      f"{delta_win:+.1f}% vs 元")
            c4.metric("取引回数（新）",  f"{len(er['trades'])}回",          f"{delta_tr:+d}回 vs 元")

            # ポートフォリオ推移比較
            if not er["portfolio_curve"].empty and not result["portfolio_curve"].empty:
                fig_cmp = create_comparison_chart(
                    result["portfolio_curve"], er["portfolio_curve"],
                    "元の指標", f"拡張指標（{len(ext_bt_state['checked'])}種）",
                    initial_cash,
                )
                st.plotly_chart(fig_cmp, width="stretch")

            # 取引履歴（拡張）
            if not er["trades"].empty:
                with st.expander(f"拡張バックテスト取引履歴（{len(er['trades'])}件）"):
                    td = er["trades"].copy()
                    td["date"]   = pd.to_datetime(td["date"]).dt.strftime("%Y-%m-%d")
                    td["price"]  = td["price"].map("{:,.1f}円".format)
                    td["profit"] = td["profit"].apply(lambda x: f"{x:+,.0f}円" if pd.notna(x) else "-")
                    td.columns   = ["日付", "種別", "株価", "株数", "損益"]
                    st.dataframe(td, width="stretch", hide_index=True)

        st.divider()

        # ─── チャート ───
        ext_overlays_on    = list(dict.fromkeys(ext_overlays_on))
        ext_oscillators_on = list(dict.fromkeys(ext_oscillators_on))
        fig = create_chart(df, ticker, ic,
                           ext_overlays=ext_overlays_on,
                           ext_oscillators=ext_oscillators_on)
        st.plotly_chart(fig, width="stretch")

        # ─── ポートフォリオ推移 ───
        if not result["portfolio_curve"].empty:
            fig_pf = create_portfolio_chart(result["portfolio_curve"], initial_cash)
            st.plotly_chart(fig_pf, width="stretch")

        # ─── 取引履歴 ───
        if not result["trades"].empty:
            with st.expander(f"取引履歴 ({len(result['trades'])}件)"):
                trades_display = result["trades"].copy()
                trades_display["date"] = pd.to_datetime(trades_display["date"]).dt.strftime("%Y-%m-%d")
                trades_display["price"] = trades_display["price"].map("{:,.1f}円".format)
                trades_display["profit"] = trades_display["profit"].apply(
                    lambda x: f"{x:+,.0f}円" if pd.notna(x) else "-"
                )
                trades_display.columns = ["日付", "種別", "株価", "株数", "損益"]
                st.dataframe(trades_display, width="stretch", hide_index=True)

        # ─── 指標テーブル（直近10日）───
        with st.expander("指標データ（直近10日）"):
            show_cols = ["Close"]
            if _s.get("use_ma"):
                show_cols += ["MA_short", "MA_long"]
            if _s.get("use_rsi"):
                show_cols += ["RSI"]
            if _s.get("use_macd"):
                show_cols += ["MACD", "MACD_sig"]
            if _s.get("use_bb"):
                show_cols += ["BB_upper", "BB_mid", "BB_lower"]
            show_cols += ["composite_signal", "vote_sum"]
            available = [c for c in show_cols if c in df.columns]
            tail_df = df[available].tail(10).copy()
            tail_df.index = tail_df.index.strftime("%Y-%m-%d")
            st.dataframe(tail_df.round(2), width="stretch")

# ─────────────────────────────────────────────
# フッター
# ─────────────────────────────────────────────
st.divider()
if auto_refresh:
    from datetime import datetime
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {refresh_sec}秒ごとに自動更新中")
else:
    st.caption("自動更新はオフです。サイドバーの「リアルタイム更新」をオンにすると定期更新されます。")
