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

from src.data.fetcher import get_stock_data as _get_stock_data_raw
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended, calculate_short_signals
from src.strategies.composite import generate_composite_signal, merge_all_signals
from src.strategies.context_strategy import generate_context_signal
from src.risk.manager import RiskManager
from src.backtest import run_backtest
from src.analysis.correlation import analyze_correlations, INDICATOR_META
from src.indicators.signal_generator import generate_ext_signals, build_ext_composite
from src.optimization.searcher import find_best_combination
from src.optimization.timeframe_detector import analyze_timeframe
from src.optimization.rsi_detector import analyze_rsi
from src.optimization.macd_detector import analyze_macd
from src.db import storage as db
from src.data.edinet import check_api_connection, find_latest_filing, DOC_TYPE_LABELS
from src.indicators.fundamental import (
    get_fundamental_data,
    calculate_fundamental_signals,
    DEFAULT_FUND_SETTINGS,
    FUND_LABELS,
    FUND_UNITS,
)
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


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """株価データ取得（5分キャッシュ）。同一セッション内で一貫したデータを返す。"""
    return _get_stock_data_raw(ticker, period, interval)


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

/* Plotly モードバー（ツールバー） */
.modebar {
    background: rgba(13, 19, 33, 0.85) !important;
    border-radius: 4px !important;
}
.modebar-btn path {
    fill: var(--text-secondary) !important;
}
.modebar-btn:hover path {
    fill: var(--accent-cyan) !important;
}
.modebar-btn.active path {
    fill: var(--accent-cyan) !important;
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


@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_data_cached(ticker: str) -> dict:
    """ファンダメンタルデータをキャッシュ付きで取得（24時間キャッシュ）。"""
    return get_fundamental_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def check_edinet_connection() -> bool:
    """EDINET API接続確認（1時間キャッシュ）。"""
    return check_api_connection()


def _get_fund_settings(ticker: str) -> dict:
    """セッション状態またはDBからファンダメンタル設定を取得する。"""
    pfx = f"fund_{ticker}"
    # ウィジェットキーが実際に存在する場合のみセッション状態を使用
    if st.session_state.get(f"_fund_init_{ticker}") and f"{pfx}_use_roe" in st.session_state:
        return {
            _k: st.session_state.get(f"{pfx}_{_k}", _v)
            for _k, _v in DEFAULT_FUND_SETTINGS.items()
        }
    loaded = db.load_fund_settings(ticker)
    return {**DEFAULT_FUND_SETTINGS, **loaded}


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
    "SELL_PRESSURE": [
        ("sell_pressure_danger",  "危険閾値", "float", 0.1, 0.99, 0.05),
        ("sell_pressure_caution", "警戒閾値", "float", 0.1, 0.99, 0.05),
    ],
    "SQUEEZE_SCORE": [
        ("squeeze_high", "高閾値", "float", 0.1, 0.99, 0.05),
        ("squeeze_mid",  "中閾値", "float", 0.1, 0.99, 0.05),
    ],
}


# ─────────────────────────────────────────────
# ヘルパー：銘柄別設定の取得・構築
# ─────────────────────────────────────────────
def _get_company_name(ticker: str) -> str:
    """企業名を取得してセッションにキャッシュする（取得失敗時はtickerをそのまま返す）。"""
    cache_key = f"_company_name_{ticker}"
    if cache_key not in st.session_state:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            name = ticker
        st.session_state[cache_key] = name
    return st.session_state[cache_key]


def _save_all_settings(ticker: str) -> None:
    """全設定をDBへ即時保存するコールバック（あらゆる設定変更時に呼ぶ）。"""
    pfx = f"cfg_{ticker}"
    if f"{pfx}_use_ma" not in st.session_state:
        return  # ウィジェット未描画時はスキップ
    _s = {
        _k: st.session_state.get(f"{pfx}_{_k}", db.DEFAULT_SETTINGS[_k])
        for _k in db.DEFAULT_SETTINGS if _k != "extra_checked"
    }
    _s["extra_checked"] = [
        _cn
        for _, _grp in _SIDEBAR_EXTRA_GROUPS
        for _, _cn in _grp
        if st.session_state.get(f"{pfx}_ext_{_cn}", False)
    ]
    db.save_settings(ticker, _s)


def _save_all_fund_settings(ticker: str) -> None:
    """ファンダメンタル設定を即時DBへ保存するコールバック。"""
    pfx = f"fund_{ticker}"
    if f"{pfx}_use_roe" not in st.session_state:
        return
    _fs = {
        _k: st.session_state.get(f"{pfx}_{_k}", _v)
        for _k, _v in DEFAULT_FUND_SETTINGS.items()
    }
    db.save_fund_settings(ticker, _fs)


def _auto_save_setting(ticker: str, setting_key: str) -> None:
    """後方互換のため残す。全設定を保存する。"""
    _save_all_settings(ticker)


def _turn_off_timeframe_diag(ticker: str) -> None:
    """MA値手動変更時に時間軸診断トグルをOFFにし、全設定をDBへ保存する。"""
    pfx = f"cfg_{ticker}"
    st.session_state[f"{pfx}_show_timeframe_diagnosis"] = False
    _save_all_settings(ticker)


def _turn_off_rsi_diag(ticker: str) -> None:
    """RSI値手動変更時にRSI診断トグルをOFFにし、全設定をDBへ保存する。"""
    pfx = f"cfg_{ticker}"
    st.session_state[f"{pfx}_show_rsi_diagnosis"] = False
    _save_all_settings(ticker)


def _turn_off_macd_diag(ticker: str) -> None:
    """MACDパラメータ手動変更時にMACD診断トグルをOFFにし、全設定をDBへ保存する。"""
    pfx = f"cfg_{ticker}"
    st.session_state[f"{pfx}_show_macd_diagnosis"] = False
    _save_all_settings(ticker)


def _is_diag_effective(diag_result: dict) -> bool:
    """診断結果が有効（best_combined > 現在の手動設定）かチェック。"""
    if "error" in diag_result:
        return False
    configs = diag_result.get("configs", [])
    if not configs:
        return False
    best = diag_result.get("best_combined", {})
    baseline = diag_result.get("baseline_return_pct", None)
    if baseline is None:
        # 古いキャッシュ（baseline未収録）: フォールバックで旧ロジック
        return any(c.get("return_pct", 0) > 0 for c in configs)
    return best.get("return_pct", 0) > baseline


def _diag_confirm_message(diag_result: dict) -> str:
    """確認UIのメッセージ文を返す（「現在の方が良い」vs「全マイナス」で出し分け）。"""
    best_ret  = diag_result.get("best_combined", {}).get("return_pct", 0)
    baseline  = diag_result.get("baseline_return_pct", None)
    if baseline is not None and best_ret > 0 and best_ret < baseline:
        return (
            f"⚠️ 現在の設定（リターン {baseline:+.1f}%）より診断推奨値（{best_ret:+.1f}%）が"
            "低い状態です。切り替えると悪化する可能性がありますが、有効にしますか？"
        )
    return "⚠️ 全プリセットでリターンがマイナスです。それでも有効にしますか？"


def _apply_diag_and_clear_caches(ticker: str, updates: dict) -> None:
    """診断推奨値をDBに保存し、_cfg_init_ とキャッシュをリセットする。
    updates に show_XXX_diagnosis=True を含めることで、トグル状態も一括保存できる。
    session_state にウィジェットがある場合はそこを base にして、ない場合はDBから読む。
    NOTE: _save_all_settings は呼ばない（古い session_state 値で updates を上書きされるのを防ぐ）。
    """
    pfx = f"cfg_{ticker}"
    if f"{pfx}_use_ma" in st.session_state:
        base = {_k: st.session_state.get(f"{pfx}_{_k}", db.DEFAULT_SETTINGS[_k])
                for _k in db.DEFAULT_SETTINGS if _k != "extra_checked"}
        base["extra_checked"] = [
            _cn for _, _grp in _SIDEBAR_EXTRA_GROUPS for _, _cn in _grp
            if st.session_state.get(f"{pfx}_ext_{_cn}", False)
        ]
    else:
        base = {**db.DEFAULT_SETTINGS, **db.load_settings(ticker)}
    base.update(updates)
    db.save_settings(ticker, base)
    st.session_state.pop(f"_cfg_init_{ticker}", None)
    for _ck in [k for k in list(st.session_state.keys())
                if ticker in k and any(k.startswith(p) for p in
                   ("df_", "bt_summary_", "opt_", "corr_cache_", "df_ext_"))]:
        st.session_state.pop(_ck, None)


def _on_toggle_timeframe_diag(ticker: str) -> None:
    """時間軸診断トグル切替コールバック: ON→キャッシュがあれば即時適用、なければ描画後適用フラグをセット。"""
    pfx = f"cfg_{ticker}"
    is_on = bool(st.session_state.get(f"{pfx}_show_timeframe_diagnosis", False))
    if is_on:
        # 診断適用前の手動設定をスナップショットとしてsession_stateに保存
        st.session_state[f"{pfx}_snap_tf_ma_short"] = int(
            st.session_state.get(f"{pfx}_ma_short", db.DEFAULT_SETTINGS["ma_short"]))
        st.session_state[f"{pfx}_snap_tf_ma_long"] = int(
            st.session_state.get(f"{pfx}_ma_long", db.DEFAULT_SETTINGS["ma_long"]))
        for _k in [k for k in list(st.session_state.keys()) if k.startswith(f"tf_result_{ticker}_")]:
            st.session_state.pop(_k, None)
        st.session_state.pop(f"_diag_inline_confirm_{ticker}_timeframe", None)
        # キャッシュがあれば即時適用してフォームに反映、なければ描画後に診断して適用
        _period = st.session_state.get("_period", "1y")
        _cached = db.load_diagnosis_cache(ticker, "timeframe", _period)
        if _cached is not None and _cached.get("best_combined", {}).get("short"):
            _b = _cached["best_combined"]
            _apply_diag_and_clear_caches(ticker, {
                "ma_short": _b["short"], "ma_long": _b["long"], "use_ma": True,
            })
        else:
            st.session_state[f"_diag_auto_apply_{ticker}_timeframe"] = True
            _save_all_settings(ticker)
    else:
        # 診断OFFでスナップショットが残っていれば元の設定に復元
        snap_short = st.session_state.get(f"{pfx}_snap_tf_ma_short")
        snap_long  = st.session_state.get(f"{pfx}_snap_tf_ma_long")
        if snap_short is not None and snap_long is not None:
            st.session_state[f"{pfx}_ma_short"] = int(snap_short)
            st.session_state[f"{pfx}_ma_long"]  = int(snap_long)
            st.session_state[f"{pfx}_snap_tf_ma_short"] = None
            st.session_state[f"{pfx}_snap_tf_ma_long"]  = None
        _save_all_settings(ticker)


def _on_toggle_rsi_diag(ticker: str) -> None:
    """RSI診断トグル切替コールバック: ON→キャッシュがあれば即時適用、なければ描画後適用フラグをセット。"""
    pfx = f"cfg_{ticker}"
    is_on = bool(st.session_state.get(f"{pfx}_show_rsi_diagnosis", False))
    if is_on:
        st.session_state[f"{pfx}_snap_rsi_ob"] = int(
            st.session_state.get(f"{pfx}_rsi_ob", db.DEFAULT_SETTINGS["rsi_ob"]))
        st.session_state[f"{pfx}_snap_rsi_os"] = int(
            st.session_state.get(f"{pfx}_rsi_os", db.DEFAULT_SETTINGS["rsi_os"]))
        for _k in [k for k in list(st.session_state.keys()) if k.startswith(f"rsi_result_{ticker}_")]:
            st.session_state.pop(_k, None)
        st.session_state.pop(f"_diag_inline_confirm_{ticker}_rsi", None)
        _period = st.session_state.get("_period", "1y")
        _cached = db.load_diagnosis_cache(ticker, "rsi", _period)
        if _cached is not None and _cached.get("best_combined", {}).get("ob"):
            _b = _cached["best_combined"]
            _apply_diag_and_clear_caches(ticker, {
                "rsi_ob": _b["ob"], "rsi_os": _b["os"],
            })
        else:
            st.session_state[f"_diag_auto_apply_{ticker}_rsi"] = True
            _save_all_settings(ticker)
    else:
        snap_ob = st.session_state.get(f"{pfx}_snap_rsi_ob")
        snap_os = st.session_state.get(f"{pfx}_snap_rsi_os")
        if snap_ob is not None and snap_os is not None:
            st.session_state[f"{pfx}_rsi_ob"] = int(snap_ob)
            st.session_state[f"{pfx}_rsi_os"] = int(snap_os)
            st.session_state[f"{pfx}_snap_rsi_ob"] = None
            st.session_state[f"{pfx}_snap_rsi_os"] = None
        _save_all_settings(ticker)


def _on_toggle_macd_diag(ticker: str) -> None:
    """MACD診断トグル切替コールバック: ON→キャッシュがあれば即時適用、なければ描画後適用フラグをセット。"""
    pfx = f"cfg_{ticker}"
    is_on = bool(st.session_state.get(f"{pfx}_show_macd_diagnosis", False))
    if is_on:
        st.session_state[f"{pfx}_snap_macd_fast"] = int(
            st.session_state.get(f"{pfx}_macd_fast", db.DEFAULT_SETTINGS["macd_fast"]))
        st.session_state[f"{pfx}_snap_macd_slow"] = int(
            st.session_state.get(f"{pfx}_macd_slow", db.DEFAULT_SETTINGS["macd_slow"]))
        st.session_state[f"{pfx}_snap_macd_sig"]  = int(
            st.session_state.get(f"{pfx}_macd_sig",  db.DEFAULT_SETTINGS["macd_sig"]))
        for _k in [k for k in list(st.session_state.keys()) if k.startswith(f"macd_result_{ticker}_")]:
            st.session_state.pop(_k, None)
        st.session_state.pop(f"_diag_inline_confirm_{ticker}_macd", None)
        _period = st.session_state.get("_period", "1y")
        _cached = db.load_diagnosis_cache(ticker, "macd", _period)
        if _cached is not None and _cached.get("best_combined", {}).get("fast"):
            _b = _cached["best_combined"]
            _apply_diag_and_clear_caches(ticker, {
                "macd_fast": _b["fast"], "macd_slow": _b["slow"], "macd_sig": _b["sig"],
            })
        else:
            st.session_state[f"_diag_auto_apply_{ticker}_macd"] = True
            _save_all_settings(ticker)
    else:
        snap_fast = st.session_state.get(f"{pfx}_snap_macd_fast")
        snap_slow = st.session_state.get(f"{pfx}_snap_macd_slow")
        snap_sig  = st.session_state.get(f"{pfx}_snap_macd_sig")
        if snap_fast is not None and snap_slow is not None and snap_sig is not None:
            st.session_state[f"{pfx}_macd_fast"] = int(snap_fast)
            st.session_state[f"{pfx}_macd_slow"] = int(snap_slow)
            st.session_state[f"{pfx}_macd_sig"]  = int(snap_sig)
            st.session_state[f"{pfx}_snap_macd_fast"] = None
            st.session_state[f"{pfx}_snap_macd_slow"] = None
            st.session_state[f"{pfx}_snap_macd_sig"]  = None
        _save_all_settings(ticker)


def _get_ticker_settings(ticker: str, db_only: bool = False) -> dict:
    """セッション状態（ライブ値）または DB から銘柄の設定を取得する。
    db_only=True の場合は常に DB から読む（ハッシュ計算など一貫性が必要な箇所で使用）。
    """
    pfx = f"cfg_{ticker}"
    if not db_only and st.session_state.get(f"_cfg_init_{ticker}") and f"{pfx}_use_ma" in st.session_state:
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


def _get_effective_settings(ticker: str, period: str, db_only: bool = False) -> dict:
    """診断トグルON時に診断推奨値で上書きした有効設定を返す。
    db_only=True の場合は session_state を参照しない（ハッシュ一貫性確保用）。
    """
    _s = _get_ticker_settings(ticker, db_only=db_only)
    if _s.get("show_timeframe_diagnosis", False):
        _tf_cache = db.load_diagnosis_cache(ticker, "timeframe", period)
        if _tf_cache:
            _b = _tf_cache.get("best_combined", {})
            if _b.get("short") and _b.get("long"):
                _s = {**_s, "ma_short": _b["short"], "ma_long": _b["long"]}
    if _s.get("show_rsi_diagnosis", False):
        _rsi_cache = db.load_diagnosis_cache(ticker, "rsi", period)
        if _rsi_cache:
            _b = _rsi_cache.get("best_combined", {})
            if _b.get("ob") and _b.get("os"):
                _s = {**_s, "rsi_ob": _b["ob"], "rsi_os": _b["os"]}
    if _s.get("show_macd_diagnosis", False):
        _macd_cache = db.load_diagnosis_cache(ticker, "macd", period)
        if _macd_cache:
            _b = _macd_cache.get("best_combined", {})
            if _b.get("fast") and _b.get("slow") and _b.get("sig"):
                _s = {**_s, "macd_fast": _b["fast"], "macd_slow": _b["slow"], "macd_sig": _b["sig"]}
    return _s


def _build_indicator_config(s: dict) -> dict:
    return {
        "use_ma":    s["use_ma"],    "ma_short":   s["ma_short"],  "ma_long":  s["ma_long"],
        "use_rsi":   s["use_rsi"],   "rsi_period": s["rsi_period"], "rsi_ob":  s["rsi_ob"],  "rsi_os": s["rsi_os"],
        "use_macd":  s["use_macd"],  "macd_fast":  s["macd_fast"],  "macd_slow": s["macd_slow"], "macd_sig": s["macd_sig"],
        "use_bb":    s["use_bb"],    "bb_period":  s["bb_period"],  "bb_std":  s["bb_std"],
    }


def _build_diag_indicator_config(s: dict) -> dict:
    """診断用indicator_config: 診断前の手動設定値をベースラインとして使用する。
    rsi_ob/os は診断で変更されないため上書きしない。
    MA/MACD はスナップショットがある場合（診断適用済み）は診断前の手動値を使用。
    """
    cfg = _build_indicator_config(s)
    # MA: スナップショットがあれば診断前の手動値を使用
    if s.get("snap_tf_ma_short") is not None:
        cfg["ma_short"] = int(s["snap_tf_ma_short"])
        cfg["ma_long"]  = int(s["snap_tf_ma_long"])
    # RSI: スナップショットがあれば診断前の手動値を使用
    if s.get("snap_rsi_ob") is not None:
        cfg["rsi_ob"] = int(s["snap_rsi_ob"])
        cfg["rsi_os"] = int(s["snap_rsi_os"])
    # MACD: スナップショットがあれば診断前の手動値を使用
    if s.get("snap_macd_fast") is not None:
        cfg["macd_fast"] = int(s["snap_macd_fast"])
        cfg["macd_slow"] = int(s["snap_macd_slow"])
        cfg["macd_sig"]  = int(s["snap_macd_sig"])
    return cfg


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


def _ensure_ticker_items() -> None:
    """銘柄リストを DB から session_state に読み込む。"""
    if "ticker_items" not in st.session_state:
        codes = db.load_stocks()
        if not codes:
            codes = ["7203.T", "6758.T", "9984.T"]
            for _c in codes:
                db.add_stock(_c)
        st.session_state.ticker_items = [
            {"id": str(uuid.uuid4()), "code": _c} for _c in codes
        ]


def _prepare_ticker_df_and_backtest(
    ticker: str, period: str, db_only: bool = False
) -> tuple[pd.DataFrame | None, dict | None, dict]:
    """データ取得・指標計算・バックテスト。失敗時は (None, None, {})。
    db_only=True の場合は session_state を参照しない（サイドバー事前計算用）。
    """
    _s = _get_effective_settings(ticker, period, db_only=db_only)
    ic = _build_indicator_config(_s)
    active = _build_active_indicators(_s)
    ext_params = _build_ext_params(_s)
    extra_cols = _s.get("extra_checked", [])
    extra_overlays = [
        c for c in extra_cols if INDICATOR_META.get(c, {}).get("type") == "overlay"
    ]
    extra_oscillators = [
        c for c in extra_cols if INDICATOR_META.get(c, {}).get("type") != "overlay"
    ]

    try:
        df = get_stock_data(ticker, period=period)
    except Exception:
        return None, None, {}

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
        df_ext = st.session_state[ext_df_key].copy()  # コピーで mutation を防止
        for _col in df_ext.columns:
            if _col not in df.columns:
                df[_col] = df_ext[_col]
        df, _ext_sig_cols = generate_ext_signals(df, extra_cols, params=_s)
        if _s.get("use_context_strategy", False):
            df = generate_context_signal(df, active, _ext_sig_cols,
                                         score_threshold=_s.get("context_score_threshold", 5),
                                         rsi_ob=_s.get("rsi_ob", 70),
                                         rsi_os=_s.get("rsi_os", 30))
        else:
            df = merge_all_signals(df, active, _ext_sig_cols)
    else:
        if _s.get("use_context_strategy", False):
            df = generate_context_signal(df, active,
                                         score_threshold=_s.get("context_score_threshold", 5),
                                         rsi_ob=_s.get("rsi_ob", 70),
                                         rsi_os=_s.get("rsi_os", 30))
        else:
            df = generate_composite_signal(df, active)

    _fund_settings = _get_fund_settings(ticker)
    _fund_data = get_fundamental_data_cached(ticker)
    _fund_result = calculate_fundamental_signals(_fund_data, _fund_settings)
    _fund_score = _fund_result["score"]
    _fund_count = _fund_result["enabled_count"]
    _fund_integrate = _s.get("fund_integrate", False)

    if _fund_integrate and _fund_count > 0 and "vote_sum" in df.columns:
        _use_ctx = _s.get("use_context_strategy", False)
        if _use_ctx:
            _threshold = _s.get("context_score_threshold", 5)
        else:
            _tech_col_count = len(active) + len(extra_cols)
            _threshold = max(1, _tech_col_count / 2)
        df["vote_sum"] = df["vote_sum"] + _fund_score
        df["composite_signal"] = 0
        df.loc[df["vote_sum"] >= _threshold, "composite_signal"] = 1
        df.loc[df["vote_sum"] <= -_threshold, "composite_signal"] = -1
        df["order"] = df["composite_signal"].diff()

    initial_cash = int(_s.get("initial_cash", 1_000_000))
    _max_shares = int(_s.get("max_shares", 0))
    risk = RiskManager(
        stop_loss_pct=_s.get("stop_loss", 5),
        take_profit_pct=_s.get("take_profit", 10),
        max_position_pct=_s.get("max_pos", 100),
        rebuy_dip_pct=_s.get("rebuy_dip", 0),
    )
    result = run_backtest(df, initial_cash=initial_cash, risk=risk, max_shares=_max_shares)

    ctx = {
        "_s": _s,
        "ic": ic,
        "active": active,
        "extra_cols": extra_cols,
        "extra_overlays": extra_overlays,
        "extra_oscillators": extra_oscillators,
        "df_ext": df_ext,
        "_fund_count": _fund_count,
        "_fund_signals": _fund_result["signals"],
        "_fund_integrate": _fund_integrate,
        "initial_cash": initial_cash,
    }
    return df, result, ctx


def _refresh_portfolio_summaries(tickers: list[str], period: str) -> None:
    """登録銘柄すべてのバックテスト結果をサイドバー集計用に更新する。
    DB から設定を読むことでハッシュ振動を防ぐ。
    """
    for t in tickers:
        _df, _res, _ = _prepare_ticker_df_and_backtest(t, period, db_only=True)
        if _res is not None:
            st.session_state[f"bt_summary_{t}"] = {
                "initial_cash": _res["initial_cash"],
                "final_value": _res["final_value"],
                "total_return_pct": _res["total_return_pct"],
                "current_position": _res["current_position"],
            }


def _invalidate_stale_summaries(period: str) -> None:
    """設定変更を検知してbt_summary_・df_ext_キャッシュをサイドバー表示前に無効化する。"""
    for item in st.session_state.get("ticker_items", []):
        t = item.get("code", "")
        if not t:
            continue
        # db_only=True で常にDBから読む → 銘柄切替によるハッシュ振動を防ぐ
        _s = _get_effective_settings(t, period, db_only=True)
        _active = _build_active_indicators(_s)
        _ic = _build_indicator_config(_s)
        _ext_p = _build_ext_params(_s)
        _hash = (
            f"period={period}"
            f"|{','.join(sorted(_active))}"
            f"|ic={_json.dumps(_ic, sort_keys=True)}"
            f"|ctx={_s.get('use_context_strategy', False)}"
            f"|thr={_s.get('context_score_threshold', 5)}"
            f"|fund={_s.get('fund_integrate', False)}"
            f"|risk={_s.get('stop_loss', 5)}/{_s.get('take_profit', 10)}"
            f"/{_s.get('max_pos', 100)}/{_s.get('rebuy_dip', 0)}"
            f"|cash={_s.get('initial_cash', 1_000_000)}"
            f"|ext={_json.dumps(_ext_p, sort_keys=True)}"
            f"|checked={','.join(sorted(_s.get('extra_checked', [])))}"
            f"|tf_diag={_s.get('show_timeframe_diagnosis', False)}"
            f"|rsi_diag_on={_s.get('show_rsi_diagnosis', False)}"
            f"|macd_diag={_s.get('show_macd_diagnosis', False)}"
        )
        _hash_key = f"_sig_hash_{t}"
        if st.session_state.get(_hash_key) != _hash:
            st.session_state.pop(f"bt_summary_{t}", None)
            for _k in [k for k in list(st.session_state.keys())
                       if k.startswith(f"df_ext_{t}_")]:
                st.session_state.pop(_k, None)
            st.session_state[_hash_key] = _hash


_ensure_ticker_items()

# DBから取得期間を先読みしてsession_stateにセット
# （サイドバー描画前にbt_summary_を正しい期間で計算するために必要）
if "_period" not in st.session_state:
    st.session_state["_period"] = db.load_global_settings().get("period", "1y")

# 設定変更を検知してbt_summary_を無効化（サイドバー事前計算の前に実行）
_invalidate_stale_summaries(st.session_state.get("_period", "1y"))

# サイドバー一覧用：未計算銘柄のbt_summaryを事前取得
_sb_period = st.session_state.get("_period", "1y")
_sb_all_tickers = [
    item["code"] for item in st.session_state.get("ticker_items", [])
    if item.get("code")
]
_sb_missing = [t for t in _sb_all_tickers if f"bt_summary_{t}" not in st.session_state]
if _sb_missing:
    _refresh_portfolio_summaries(_sb_missing, _sb_period)

# MACDパラメータ診断: キャッシュ未保存の銘柄を起動時にバックグラウンドで実行
_macd_diag_missing = [
    t for t in _sb_all_tickers
    if db.load_diagnosis_cache(t, "macd", _sb_period) is None
]
if _macd_diag_missing:
    for _t in _macd_diag_missing:
        try:
            _t_df   = get_stock_data(_t, period=_sb_period)
            _t_s    = {**db.DEFAULT_SETTINGS, **db.load_settings(_t)}
            _t_cash = _t_s.get("initial_cash", 1_000_000)
            _t_cfg  = _build_diag_indicator_config(_t_s)
            _t_result = analyze_macd(_t_df, initial_cash=_t_cash, indicator_config=_t_cfg)
            db.save_diagnosis_cache(_t, "macd", _sb_period, _t_result)
        except Exception:
            pass

# ─────────────────────────────────────────────
# サイドバー設定
# ─────────────────────────────────────────────
with st.sidebar:
    st.toggle("📊 基本情報パネル", value=True, key="_show_right_panel")
    st.divider()

    # ── ポートフォリオ一覧 ──
    if _sb_all_tickers:
        # 合計集計（bt_summary が取得できている銘柄のみ）
        _sb_total_ic = 0
        _sb_total_fv = 0
        _sb_valid_n = 0
        _sb_rows_html = ""
        for _sbt in _sb_all_tickers:
            _sbts = st.session_state.get(f"bt_summary_{_sbt}", {})
            if _sbts:
                _sb_ic = _sbts.get("initial_cash", 0)
                _sb_fv = _sbts.get("final_value", 0)
                _sb_profit = _sb_fv - _sb_ic
                _sb_ret = _sbts.get("total_return_pct", 0.0)
                _sb_total_ic += _sb_ic
                _sb_total_fv += _sb_fv
                _sb_valid_n += 1
                _sb_color = "#4caf50" if _sb_profit >= 0 else "#f44336"
                _sb_sign = "+" if _sb_profit >= 0 else ""
                _sb_rows_html += (
                    f"<div style='margin-bottom:8px;padding-bottom:8px;"
                    f"border-bottom:1px solid #333;'>"
                    f"<div style='font-weight:700;font-size:0.8rem;margin-bottom:2px'>{_sbt}</div>"
                    f"<div style='font-size:0.72rem;color:#aaa;'>初期: ¥{_sb_ic:,.0f}</div>"
                    f"<div style='font-size:0.72rem;color:{_sb_color};'>"
                    f"損益: {_sb_sign}¥{_sb_profit:,.0f}"
                    f"&nbsp;({_sb_sign}{_sb_ret:.2f}%)</div>"
                    f"</div>"
                )
            else:
                _sb_rows_html += (
                    f"<div style='margin-bottom:8px;padding-bottom:8px;"
                    f"border-bottom:1px solid #333;'>"
                    f"<div style='font-weight:700;font-size:0.8rem;color:#888'>{_sbt}</div>"
                    f"<div style='font-size:0.7rem;color:#666'>— 未計算</div>"
                    f"</div>"
                )

        # 合計サマリー HTML
        _sb_total_profit = _sb_total_fv - _sb_total_ic
        _sb_total_ret = ((_sb_total_fv / _sb_total_ic) - 1) * 100 if _sb_total_ic > 0 else 0.0
        _sb_sum_color = "#4caf50" if _sb_total_profit >= 0 else "#f44336"
        _sb_sum_sign = "+" if _sb_total_profit >= 0 else ""
        _sb_summary_html = (
            f"<div style='background:#1e1e2e;border-radius:6px;padding:8px 10px;"
            f"margin-bottom:10px;border:1px solid #444;'>"
            f"<div style='font-size:0.7rem;color:#888;margin-bottom:4px;font-weight:600'>"
            f"📋 ポートフォリオ合計（{_sb_valid_n}銘柄）</div>"
            f"<div style='font-size:0.72rem;color:#aaa;'>初期投資合計: ¥{_sb_total_ic:,.0f}</div>"
            f"<div style='font-size:0.8rem;font-weight:700;color:{_sb_sum_color};margin-top:3px;'>"
            f"損益合計: {_sb_sum_sign}¥{_sb_total_profit:,.0f}</div>"
            f"<div style='font-size:0.72rem;color:{_sb_sum_color};'>"
            f"合計リターン: {_sb_sum_sign}{_sb_total_ret:.2f}%</div>"
            f"</div>"
        )

        st.markdown(_sb_summary_html, unsafe_allow_html=True)
        with st.expander("銘柄別の損益を表示", expanded=False):
            st.markdown(
                f"<div style='padding:4px 0'>{_sb_rows_html}</div>",
                unsafe_allow_html=True,
            )
        st.divider()

    with st.expander("📈 銘柄設定"):
        _ensure_ticker_items()
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
                    # 新規銘柄を即時3診断してキャッシュ保存・有効性に応じてトグル設定
                    try:
                        _new_df    = get_stock_data(new_code, period=_sb_period)
                        _new_s     = {**db.DEFAULT_SETTINGS, **db.load_settings(new_code)}
                        _new_cash  = _new_s.get("initial_cash", 1_000_000)
                        _new_cfg   = _build_diag_indicator_config(_new_s)

                        _tf_res   = analyze_timeframe(_new_df, initial_cash=_new_cash, indicator_config=_new_cfg)
                        db.save_diagnosis_cache(new_code, "timeframe", _sb_period, _tf_res)

                        _rsi_res  = analyze_rsi(_new_df, initial_cash=_new_cash, indicator_config=_new_cfg)
                        db.save_diagnosis_cache(new_code, "rsi", _sb_period, _rsi_res)

                        _macd_res = analyze_macd(_new_df, initial_cash=_new_cash, indicator_config=_new_cfg)
                        db.save_diagnosis_cache(new_code, "macd", _sb_period, _macd_res)

                        _init_s = dict(_new_s)

                        _b_tf = _tf_res.get("best_combined", {})
                        if _b_tf.get("short") and _b_tf.get("long"):
                            _init_s["show_timeframe_diagnosis"] = True
                            _init_s["ma_short"] = _b_tf["short"]
                            _init_s["ma_long"]  = _b_tf["long"]
                            _init_s["use_ma"]   = True

                        _b_rsi = _rsi_res.get("best_combined", {})
                        if _b_rsi.get("ob") and _b_rsi.get("os"):
                            _init_s["show_rsi_diagnosis"] = True
                            _init_s["rsi_ob"] = _b_rsi["ob"]
                            _init_s["rsi_os"] = _b_rsi["os"]

                        _b_macd = _macd_res.get("best_combined", {})
                        if _b_macd.get("fast") and _b_macd.get("slow") and _b_macd.get("sig"):
                            _init_s["show_macd_diagnosis"] = True
                            _init_s["macd_fast"] = _b_macd["fast"]
                            _init_s["macd_slow"] = _b_macd["slow"]
                            _init_s["macd_sig"]  = _b_macd["sig"]

                        db.save_settings(new_code, _init_s)
                    except Exception:
                        pass
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

        _period_opts = ["1mo", "3mo", "6mo", "1y", "2y"]
        if "_period" not in st.session_state:
            st.session_state["_period"] = db.load_global_settings().get("period", "1y")
        period = st.selectbox(
            "取得期間",
            _period_opts,
            index=_period_opts.index(st.session_state["_period"]),
            format_func=lambda x: {"1mo": "1ヶ月", "3mo": "3ヶ月", "6mo": "6ヶ月", "1y": "1年", "2y": "2年"}[x],
            key="_period",
        )
        db.save_global_settings({"period": period})

    # expander外でも tickers / active_ticker / period を常時計算
    tickers = [item["code"] for item in st.session_state.get("ticker_items", []) if item["code"]]
    if "active_ticker" not in st.session_state or st.session_state.get("active_ticker") not in tickers:
        st.session_state.active_ticker = tickers[0] if tickers else ""
    period = st.session_state.get("_period", "1y")

    with st.expander("⚙️ テクニカル指標設定"):
        # ── 設定を編集する銘柄を選択（ダッシュボードと双方向同期）──
        _at = st.session_state.get("active_ticker", "")
        _at_idx = tickers.index(_at) if _at in tickers else 0
        settings_ticker = st.selectbox(
            "設定を編集する銘柄",
            options=tickers if tickers else [""],
            index=_at_idx,
            format_func=lambda x: (f"{x}  {get_company_name(x)}" if x else "(銘柄なし)"),
        )
        # サイドバー変更 → active_ticker へ反映
        if settings_ticker and settings_ticker != st.session_state.get("active_ticker"):
            st.session_state.active_ticker = settings_ticker

        if settings_ticker:
            pfx = f"cfg_{settings_ticker}"
            _init_key = f"_cfg_init_{settings_ticker}"

            # 初回、または銘柄切替でウィジェット session_state が消えたときに DB から再展開
            _needs_init = (
                not st.session_state.get(_init_key)
                or f"{pfx}_use_ma" not in st.session_state
            )
            if _needs_init:
                _s0 = db.load_settings(settings_ticker)
                _s0 = {**db.DEFAULT_SETTINGS, **_s0}
                # on_change で即時保存しているため DB は常に最新値。
                # 常にDBから上書きすることで session_state/DB の乖離によるハッシュ振動を防ぐ。
                for _k, _v in db.DEFAULT_SETTINGS.items():
                    if _k != "extra_checked":
                        _sk = f"{pfx}_{_k}"
                        st.session_state[_sk] = _s0.get(_k, _v)
                _ec0 = _s0.get("extra_checked", [])
                for _, _grp in _SIDEBAR_EXTRA_GROUPS:
                    for _, _cn in _grp:
                        _ek = f"{pfx}_ext_{_cn}"
                        if _ek not in st.session_state:
                            st.session_state[_ek] = _cn in _ec0
                st.session_state[_init_key] = True

            # ── メイン4指標 ──
            _use_ma = st.checkbox("移動平均（MA）", key=f"{pfx}_use_ma",
                                  on_change=_save_all_settings, args=(settings_ticker,))
            if _use_ma:
                _tf_on = st.session_state.get(f"{pfx}_show_timeframe_diagnosis", True)
                _ma_label_short = ":red[短期]" if _tf_on else "短期"
                _ma_label_long  = ":red[長期]" if _tf_on else "長期"
                _c1, _c2 = st.columns(2)
                _c1.number_input(_ma_label_short, min_value=2,  max_value=50,  step=1, key=f"{pfx}_ma_short",
                                 on_change=_turn_off_timeframe_diag, args=(settings_ticker,))
                _c2.number_input(_ma_label_long,  min_value=5,  max_value=200, step=1, key=f"{pfx}_ma_long",
                                 on_change=_turn_off_timeframe_diag, args=(settings_ticker,))
            st.toggle(
                "🕐 時間軸適合診断を表示",
                key=f"{pfx}_show_timeframe_diagnosis",
                on_change=_on_toggle_timeframe_diag,
                args=(settings_ticker,),
                help="ONにすると時間軸適合診断（MA最適化）を表示・自動適用します。",
            )
            if st.session_state.get(f"_diag_confirm_{settings_ticker}_timeframe"):
                _tf_cm = db.load_diagnosis_cache(settings_ticker, "timeframe", st.session_state.get("_period", "1y"))
                st.warning(_diag_confirm_message(_tf_cm or {}))
                _cc1, _cc2 = st.columns(2)
                if _cc1.button("はい", key=f"confirm_tf_{settings_ticker}"):
                    _period_now = st.session_state.get("_period", "1y")
                    _conf_cache = db.load_diagnosis_cache(settings_ticker, "timeframe", _period_now)
                    _conf_best  = (_conf_cache or {}).get("best_combined", {})
                    _conf_upd   = {"show_timeframe_diagnosis": True}
                    if _conf_best.get("short") and _conf_best.get("long"):
                        _conf_upd["ma_short"] = _conf_best["short"]
                        _conf_upd["ma_long"]  = _conf_best["long"]
                        _conf_upd["use_ma"]   = True
                    _apply_diag_and_clear_caches(settings_ticker, _conf_upd)
                    st.session_state[f"_diag_confirm_{settings_ticker}_timeframe"] = False
                    st.rerun()
                if _cc2.button("いいえ", key=f"cancel_tf_{settings_ticker}"):
                    st.session_state[f"_diag_confirm_{settings_ticker}_timeframe"] = False
                    st.session_state[f"{pfx}_snap_tf_ma_short"] = None
                    st.session_state[f"{pfx}_snap_tf_ma_long"]  = None
                    _save_all_settings(settings_ticker)
                    st.rerun()

            _use_rsi = st.checkbox("RSI", key=f"{pfx}_use_rsi",
                                   on_change=_save_all_settings, args=(settings_ticker,))
            if _use_rsi:
                st.slider("RSI 期間", 5, 30, key=f"{pfx}_rsi_period",
                          on_change=_save_all_settings, args=(settings_ticker,))
                _rsi_on = st.session_state.get(f"{pfx}_show_rsi_diagnosis", True)
                _rsi_label_ob = ":red[買われすぎ]" if _rsi_on else "買われすぎ"
                _rsi_label_os = ":red[売られすぎ]" if _rsi_on else "売られすぎ"
                _c1, _c2 = st.columns(2)
                _c1.number_input(_rsi_label_ob, min_value=60, max_value=90, step=1, key=f"{pfx}_rsi_ob",
                                 on_change=_turn_off_rsi_diag, args=(settings_ticker,))
                _c2.number_input(_rsi_label_os, min_value=10, max_value=40, step=1, key=f"{pfx}_rsi_os",
                                 on_change=_turn_off_rsi_diag, args=(settings_ticker,))
            st.toggle(
                "📊 RSI閾値適合診断を表示",
                key=f"{pfx}_show_rsi_diagnosis",
                on_change=_on_toggle_rsi_diag,
                args=(settings_ticker,),
                help="ONにするとRSI閾値適合診断を表示・自動適用します。",
            )
            if st.session_state.get(f"_diag_confirm_{settings_ticker}_rsi"):
                _rsi_cm = db.load_diagnosis_cache(settings_ticker, "rsi", st.session_state.get("_period", "1y"))
                st.warning(_diag_confirm_message(_rsi_cm or {}))
                _rc1, _rc2 = st.columns(2)
                if _rc1.button("はい", key=f"confirm_rsi_{settings_ticker}"):
                    _period_now = st.session_state.get("_period", "1y")
                    _conf_cache = db.load_diagnosis_cache(settings_ticker, "rsi", _period_now)
                    _conf_best  = (_conf_cache or {}).get("best_combined", {})
                    _conf_upd   = {"show_rsi_diagnosis": True}
                    if _conf_best.get("ob") and _conf_best.get("os"):
                        _conf_upd["rsi_ob"] = _conf_best["ob"]
                        _conf_upd["rsi_os"] = _conf_best["os"]
                    _apply_diag_and_clear_caches(settings_ticker, _conf_upd)
                    st.session_state[f"_diag_confirm_{settings_ticker}_rsi"] = False
                    st.rerun()
                if _rc2.button("いいえ", key=f"cancel_rsi_{settings_ticker}"):
                    st.session_state[f"_diag_confirm_{settings_ticker}_rsi"] = False
                    st.session_state[f"{pfx}_snap_rsi_ob"] = None
                    st.session_state[f"{pfx}_snap_rsi_os"] = None
                    _save_all_settings(settings_ticker)
                    st.rerun()

            _use_macd = st.checkbox("MACD", key=f"{pfx}_use_macd",
                                    on_change=_save_all_settings, args=(settings_ticker,))
            if _use_macd:
                _macd_on = st.session_state.get(f"{pfx}_show_macd_diagnosis", True)
                _macd_label_f = ":red[Fast]"   if _macd_on else "Fast"
                _macd_label_s = ":red[Slow]"   if _macd_on else "Slow"
                _macd_label_g = ":red[Signal]" if _macd_on else "Signal"
                _c1, _c2, _c3 = st.columns(3)
                _c1.number_input(_macd_label_f, min_value=3,  max_value=50,  step=1, key=f"{pfx}_macd_fast",
                                 on_change=_turn_off_macd_diag, args=(settings_ticker,))
                _c2.number_input(_macd_label_s, min_value=5,  max_value=100, step=1, key=f"{pfx}_macd_slow",
                                 on_change=_turn_off_macd_diag, args=(settings_ticker,))
                _c3.number_input(_macd_label_g, min_value=2,  max_value=30,  step=1, key=f"{pfx}_macd_sig",
                                 on_change=_turn_off_macd_diag, args=(settings_ticker,))

            st.toggle(
                "📊 MACDパラメータ診断を表示",
                key=f"{pfx}_show_macd_diagnosis",
                on_change=_on_toggle_macd_diag,
                args=(settings_ticker,),
                help="ONにするとMACDパラメータ適合診断を表示・自動適用します。",
            )
            if st.session_state.get(f"_diag_confirm_{settings_ticker}_macd"):
                _macd_cm = db.load_diagnosis_cache(settings_ticker, "macd", st.session_state.get("_period", "1y"))
                st.warning(_diag_confirm_message(_macd_cm or {}))
                _mc1, _mc2 = st.columns(2)
                if _mc1.button("はい", key=f"confirm_macd_{settings_ticker}"):
                    _period_now = st.session_state.get("_period", "1y")
                    _conf_cache = db.load_diagnosis_cache(settings_ticker, "macd", _period_now)
                    _conf_best  = (_conf_cache or {}).get("best_combined", {})
                    _conf_upd   = {"show_macd_diagnosis": True}
                    if _conf_best.get("fast") and _conf_best.get("slow") and _conf_best.get("sig"):
                        _conf_upd["macd_fast"] = _conf_best["fast"]
                        _conf_upd["macd_slow"] = _conf_best["slow"]
                        _conf_upd["macd_sig"]  = _conf_best["sig"]
                    _apply_diag_and_clear_caches(settings_ticker, _conf_upd)
                    st.session_state[f"_diag_confirm_{settings_ticker}_macd"] = False
                    st.rerun()
                if _mc2.button("いいえ", key=f"cancel_macd_{settings_ticker}"):
                    st.session_state[f"_diag_confirm_{settings_ticker}_macd"] = False
                    st.session_state[f"{pfx}_snap_macd_fast"] = None
                    st.session_state[f"{pfx}_snap_macd_slow"] = None
                    st.session_state[f"{pfx}_snap_macd_sig"]  = None
                    _save_all_settings(settings_ticker)
                    st.rerun()

            _use_bb = st.checkbox("ボリンジャーバンド", key=f"{pfx}_use_bb",
                                  on_change=_save_all_settings, args=(settings_ticker,))
            if _use_bb:
                _c1, _c2 = st.columns(2)
                _c1.number_input("BB期間",  min_value=5,   max_value=50,  step=1,   key=f"{pfx}_bb_period",
                                 on_change=_save_all_settings, args=(settings_ticker,))
                _c2.number_input("標準偏差", min_value=1.0, max_value=3.0, step=0.5,
                                 format="%.1f", key=f"{pfx}_bb_std",
                                 on_change=_save_all_settings, args=(settings_ticker,))

            # ── シグナル方式切り替え ──
            st.divider()
            _use_ctx = st.toggle(
                "🧠 コンテキスト方式（TREND/RANGE判定）",
                key=f"{pfx}_use_context_strategy",
                on_change=_save_all_settings, args=(settings_ticker,),
                help="ONにすると相場状態を先に判定し、重み付きスコアで売買判断します。OFFは従来の多数決方式。",
            )
            if _use_ctx:
                st.slider(
                    "スコア閾値（エントリーに必要な最低スコア）",
                    min_value=3, max_value=10, step=1,
                    key=f"{pfx}_context_score_threshold",
                    on_change=_save_all_settings, args=(settings_ticker,),
                    help="スコアの最大値は10（MA±3、MACD±3、RSI±2、BB±2）",
                )

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
                            on_change=_save_all_settings, args=(settings_ticker,),
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
                                            on_change=_save_all_settings, args=(settings_ticker,),
                                        )
                                    else:
                                        _pcols[_pi].number_input(
                                            _plbl,
                                            min_value=int(_pmn), max_value=int(_pmx),
                                            step=int(_pstep), key=_wk,
                                            on_change=_save_all_settings, args=(settings_ticker,),
                                        )

            # ── 投資・リスク設定 ──
            st.divider()
            st.markdown("**💰 投資・リスク設定**")
            st.number_input(
                "初期資金（円）", min_value=100_000, step=100_000,
                key=f"{pfx}_initial_cash",
                on_change=_save_all_settings, args=(settings_ticker,),
            )
            st.number_input(
                "最大株数（株）", min_value=0, step=100,
                key=f"{pfx}_max_shares",
                on_change=_save_all_settings, args=(settings_ticker,),
                help="0=制限なし（初期資金の投資割合で自動決定）",
            )
            _rc1, _rc2 = st.columns(2)
            _rc1.slider("損切りライン（%）", 1, 30, key=f"{pfx}_stop_loss",
                        on_change=_save_all_settings, args=(settings_ticker,))
            _rc2.slider("利確ライン（%）",   1, 50, key=f"{pfx}_take_profit",
                        on_change=_save_all_settings, args=(settings_ticker,))
            _rc1, _rc2 = st.columns(2)
            _rc1.slider("最大投資割合（%）",   10, 100, key=f"{pfx}_max_pos",
                        on_change=_save_all_settings, args=(settings_ticker,))
            _rc2.slider(
                "買い戻し下落率（%）", 0, 20, key=f"{pfx}_rebuy_dip",
                on_change=_save_all_settings, args=(settings_ticker,),
                help="0=シグナルが出次第即時再エントリー",
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
        else:
            settings_ticker = tickers[0] if tickers else ""

    with st.expander("📋 ファンダメンタル指標設定"):
        _fund_ticker = st.session_state.get("active_ticker", tickers[0] if tickers else "")

        if _fund_ticker:
            _fpfx = f"fund_{_fund_ticker}"
            _finit_key = f"_fund_init_{_fund_ticker}"

            # 初回、または銘柄切替でウィジェット session_state が消えたときに DB から再展開
            _fund_needs_init = (
                not st.session_state.get(_finit_key)
                or f"{_fpfx}_use_roe" not in st.session_state
            )
            if _fund_needs_init:
                _fs0 = {**DEFAULT_FUND_SETTINGS, **db.load_fund_settings(_fund_ticker)}
                for _fk, _fv in DEFAULT_FUND_SETTINGS.items():
                    _fsk = f"{_fpfx}_{_fk}"
                    if _fsk not in st.session_state:
                        st.session_state[_fsk] = _fs0.get(_fk, _fv)
                st.session_state[_finit_key] = True

            with st.expander("📋 ファンダメンタル閾値設定"):
                st.caption("各指標の買い／売りシグナル判定閾値を設定します")
                _FUND_ROWS = [
                    ("ROE（自己資本利益率）", "roe",        "use_roe",        "roe_buy",         "roe_sell",         "%",  0.0, 50.0, 0.5),
                    ("ROA（総資産利益率）",    "roa",        "use_roa",        "roa_buy",         "roa_sell",         "%",  0.0, 30.0, 0.5),
                    ("EPS成長率",             "eps_growth", "use_eps_growth", "eps_growth_buy",  "eps_growth_sell",  "%", -30.0, 100.0, 1.0),
                    ("売上高成長率",           "rev_growth", "use_rev_growth", "rev_growth_buy",  "rev_growth_sell",  "%", -30.0, 100.0, 1.0),
                    ("PER（株価収益率）",      "per",        "use_per",        "per_buy",         "per_sell",         "倍", 1.0, 100.0, 1.0),
                    ("PBR（株価純資産倍率）",  "pbr",        "use_pbr",        "pbr_buy",         "pbr_sell",         "倍", 0.1, 20.0, 0.1),
                    ("営業利益率",             "op_margin",  "use_op_margin",  "op_margin_buy",   "op_margin_sell",   "%",  0.0, 50.0, 0.5),
                    ("負債比率（D/E×100）",    "debt_ratio", "use_debt_ratio", "debt_ratio_buy",  "debt_ratio_sell",  "",   0.0, 500.0, 5.0),
                ]
                for _flbl, _fkey, _fuse, _fbuy, _fsell, _funit, _fmin, _fmax, _fstep in _FUND_ROWS:
                    _use_checked = st.checkbox(
                        _flbl, key=f"{_fpfx}_{_fuse}",
                        on_change=_save_all_fund_settings, args=(_fund_ticker,),
                    )
                    if _use_checked:
                        _fc1, _fc2 = st.columns(2)
                        if _fkey in ("per", "pbr", "debt_ratio"):
                            _fc1.number_input(
                                f"買い閾値（{_funit}以下）" if _funit else "買い閾値（以下）",
                                min_value=_fmin, max_value=_fmax, step=_fstep,
                                format="%.1f", key=f"{_fpfx}_{_fbuy}",
                                on_change=_save_all_fund_settings, args=(_fund_ticker,),
                            )
                            _fc2.number_input(
                                f"売り閾値（{_funit}以上）" if _funit else "売り閾値（以上）",
                                min_value=_fmin, max_value=_fmax, step=_fstep,
                                format="%.1f", key=f"{_fpfx}_{_fsell}",
                                on_change=_save_all_fund_settings, args=(_fund_ticker,),
                            )
                        else:
                            _fc1.number_input(
                                f"買い閾値（{_funit}以上）" if _funit else "買い閾値（以上）",
                                min_value=_fmin, max_value=_fmax, step=_fstep,
                                format="%.1f", key=f"{_fpfx}_{_fbuy}",
                                on_change=_save_all_fund_settings, args=(_fund_ticker,),
                            )
                            _fc2.number_input(
                                f"売り閾値（{_funit}以下）" if _funit else "売り閾値（以下）",
                                min_value=_fmin, max_value=_fmax, step=_fstep,
                                format="%.1f", key=f"{_fpfx}_{_fsell}",
                                on_change=_save_all_fund_settings, args=(_fund_ticker,),
                            )

            if st.button("💾 ファンダメンタル設定を保存", key=f"_fund_save_{_fund_ticker}",
                         use_container_width=True, type="primary"):
                _fsave = {
                    _fk: st.session_state.get(f"{_fpfx}_{_fk}", DEFAULT_FUND_SETTINGS[_fk])
                    for _fk in DEFAULT_FUND_SETTINGS
                }
                db.save_fund_settings(_fund_ticker, _fsave)
                st.success("ファンダメンタル設定を保存しました！")

    with st.expander("🔄 自動更新"):
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

# リスク管理は銘柄別設定から取得するためここでは生成しない


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
    trades: pd.DataFrame | None = None,
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
            marker=dict(symbol="triangle-up", color="#ff5252", size=10,
                        line=dict(width=1, color="white")),
            opacity=0.5,
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.01, mode="markers", name="売りシグナル",
            marker=dict(symbol="triangle-down", color="#40c4ff", size=10,
                        line=dict(width=1, color="white")),
            opacity=0.5,
        ), row=1, col=1)

    # ── 実際の売買マーカー ──
    if trades is not None and not trades.empty:
        _trade_styles = {
            "BUY":         dict(symbol="circle",        color="#00e676", size=13, label="実売買（買）"),
            "SELL":        dict(symbol="circle",        color="#40c4ff", size=13, label="実売買（売）"),
            "STOP_LOSS":   dict(symbol="x",             color="#ff1744", size=14, label="損切り"),
            "TAKE_PROFIT": dict(symbol="star",          color="#ffd600", size=14, label="利確"),
        }
        for _ttype, _style in _trade_styles.items():
            _t = trades[trades["type"] == _ttype]
            if _t.empty:
                continue
            # trade の date と df.index を照合して価格を取得
            _t_dates = _t["date"]
            _t_prices = _t["price"].values
            _t_shares = _t["shares"].values
            _t_profits = _t["profit"].values
            _is_buy = _ttype == "BUY"

            # ローソク足上下への配置
            _df_prices = []
            for _d, _p in zip(_t_dates, _t_prices):
                if _d in df.index:
                    _row = df.loc[_d]
                    _df_prices.append(_row["Low"] * 0.975 if _is_buy else _row["High"] * 1.025)
                else:
                    _df_prices.append(_p)

            # 売り系マーカー: 損益に応じて色を変える
            def _pf_color(pf):
                if pf is None or pd.isna(pf):
                    return _style["color"]
                return "#00e676" if pf >= 0 else "#ff1744"

            def _pf_label(pf):
                """マーカー上の短い損益テキスト (+15k / -8k など)"""
                if pf is None or pd.isna(pf):
                    return ""
                sign = "+" if pf >= 0 else ""
                if abs(pf) >= 1_000_000:
                    return f"{sign}{pf/1_000_000:.1f}M"
                elif abs(pf) >= 1_000:
                    return f"{sign}{pf/1_000:.0f}k"
                return f"{sign}{pf:.0f}"

            if _is_buy:
                _marker_colors = [_style["color"]] * len(_t)
                _marker_texts  = ["B"] * len(_t)
                _text_colors   = ["black"] * len(_t)
                _text_pos      = "middle center"
            else:
                _marker_colors = [_pf_color(pf) for pf in _t_profits]
                _short_label   = {"SELL": "S", "STOP_LOSS": "SL", "TAKE_PROFIT": "TP"}[_ttype]
                _marker_texts  = [
                    f"{_short_label} {_pf_label(pf)}" for pf in _t_profits
                ]
                _text_colors   = [
                    "#00e676" if (pf is not None and not pd.isna(pf) and pf >= 0) else "#ff1744"
                    for pf in _t_profits
                ]
                _text_pos = "top center"

            _hover = [
                (f"{_ttype}<br>価格: ¥{p:,.0f}<br>株数: {int(s)}"
                 + (f"<br>損益: ¥{pf:,.0f}" if pf is not None and not pd.isna(pf) else ""))
                for p, s, pf in zip(_t_prices, _t_shares, _t_profits)
            ]
            fig.add_trace(go.Scatter(
                x=_t_dates, y=_df_prices,
                mode="markers+text",
                name=_style["label"],
                text=_marker_texts,
                textposition=_text_pos,
                textfont=dict(size=8, color=_text_colors),
                marker=dict(
                    symbol=_style["symbol"],
                    color=_marker_colors,
                    size=_style["size"],
                    line=dict(width=1.5, color="white"),
                ),
                hovertext=_hover,
                hoverinfo="text",
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
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=80),
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
        margin=dict(l=0, r=0, t=40, b=60),
        title="ポートフォリオ推移 比較",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
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

# ── 銘柄ナビゲーションボタン（企業コード＋企業名、双方向同期） ──
if "active_ticker" not in st.session_state or st.session_state.active_ticker not in tickers:
    st.session_state.active_ticker = tickers[0]

_nav_cols = st.columns(len(tickers))
for _ni, (_nc, _nt) in enumerate(zip(_nav_cols, tickers)):
    _nn = get_company_name(_nt)
    _nshort = (_nn[:8] + "…") if len(_nn) > 8 else _nn
    _is_active = (st.session_state.active_ticker == _nt)
    if _nc.button(
        f"📊 {_nt}  \n{_nshort}",
        use_container_width=True,
        type="primary" if _is_active else "secondary",
        key=f"_nav_{_nt}",
    ):
        st.session_state.active_ticker = _nt
        st.rerun()

# ── 下部パネル：アクティブ銘柄のファンダメンタル情報を固定表示 ──
_panel_ticker = (
    st.session_state.get("active_ticker", tickers[0] if tickers else "")
    if st.session_state.get("_show_right_panel", True) else None
)
if _panel_ticker:
    _pd = get_fundamental_data_cached(_panel_ticker)
    _pfs = _get_fund_settings(_panel_ticker)
    _pfr = calculate_fundamental_signals(_pd, _pfs)
    _psigs = _pfr["signals"]
    _pscore = _pfr["score"]
    _pedinet = None
    if _panel_ticker.upper().endswith(".T"):
        _pcode4 = _panel_ticker.replace(".T", "").replace(".t", "")[:4]
        _pedinet = db.load_edinet_cache(_pcode4)

    def _sig_icon(k: str) -> str:
        s = _psigs.get(k, 0)
        return "🟢" if s > 0 else ("🔴" if s < 0 else "⚪")

    def _fmt(k: str, unit: str) -> str:
        v = _pd.get(k)
        return f"{v:.1f}{unit}" if v is not None else "—"

    _score_color = "#00ff88" if _pscore >= 2 else ("#ff4060" if _pscore <= -2 else "#8892a4")
    _pname = get_company_name(_panel_ticker)

    # ── 会社情報タイル ──
    _ci_html = (
        "<div style='flex-shrink:0;padding-right:16px;border-right:1px solid #1e2d40;min-width:100px'>"
        "<div style='font-size:0.6rem;color:#00d4ff;font-weight:700;margin-bottom:3px'>&#x1F4CA; 基本情報</div>"
        f"<div style='font-size:0.7rem;color:#e8eaf0;font-weight:600;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{_pname}</div>"
        f"<div style='font-size:0.58rem;color:#888'>{_panel_ticker}</div>"
        f"<div style='font-size:0.65rem;color:{_score_color};font-weight:600;margin-top:2px'>スコア: {_pscore:+d}</div>"
        "</div>"
    )

    # ── ポートフォリオタイル ──
    _pbts = st.session_state.get(f"bt_summary_{_panel_ticker}", {})
    _pf_html = ""
    if _pbts:
        _pic    = _pbts.get("initial_cash", 0)
        _pfv    = _pbts.get("final_value", 0)
        _pprofit = _pfv - _pic
        _pretpct = _pbts.get("total_return_pct", 0.0)
        _ppos   = _pbts.get("current_position", 0)
        _pc     = "#00ff88" if _pprofit >= 0 else "#ff4060"
        _psign  = "+" if _pprofit >= 0 else ""
        _pf_items = [
            ("初期投資額",   f"{_pic:,.0f}円",              "#e8eaf0", ""),
            ("現在の評価額", f"{_pfv:,.0f}円",              "#e8eaf0", ""),
            ("損益合計",     f"{_psign}{_pprofit:,.0f}円",  _pc,       "font-weight:600"),
            ("リターン率",   f"{_pretpct:+.2f}%",           _pc,       "font-weight:600"),
            ("保有株式数",   f"{_ppos:,}株",                "#e8eaf0", ""),
        ]
        _pf_tiles = "".join(
            f"<div style='text-align:center;padding:0 8px;border-right:1px solid #1a2236'>"
            f"<div style='font-size:0.58rem;color:#8892a4;margin-bottom:1px'>{lbl}</div>"
            f"<div style='font-size:0.68rem;color:{col};{xtra}'>{val}</div>"
            f"</div>"
            for lbl, val, col, xtra in _pf_items
        )
        _pf_html = (
            "<div style='flex-shrink:0;padding-right:16px;border-right:1px solid #1e2d40'>"
            "<div style='font-size:0.6rem;color:#f0c040;font-weight:600;margin-bottom:3px'>&#x1F4B0; ポートフォリオ</div>"
            "<div style='display:flex;align-items:center'>" + _pf_tiles + "</div>"
            "</div>"
        )

    # ── ファンダメンタルタイル ──
    _panel_rows = [
        ("PER",    "per",        "倍"),
        ("PBR",    "pbr",        "倍"),
        ("ROE",    "roe",        "%"),
        ("ROA",    "roa",        "%"),
        ("EPS成長", "eps_growth", "%"),
        ("売上成長", "rev_growth", "%"),
        ("営業利益率", "op_margin", "%"),
        ("負債比率",  "debt_ratio", ""),
    ]
    _fund_tiles = "".join(
        f"<div style='text-align:center;padding:0 8px;border-right:1px solid #1a2236'>"
        f"<div style='font-size:0.58rem;color:#8892a4;margin-bottom:1px'>{lbl}</div>"
        f"<div style='font-size:0.65rem;color:#e8eaf0'>{_sig_icon(key)} {_fmt(key, unit)}</div>"
        f"</div>"
        for lbl, key, unit in _panel_rows
    )
    _fund_html = (
        "<div style='flex-shrink:0"
        + (";padding-right:16px;border-right:1px solid #1e2d40" if _pedinet else "")
        + "'>"
        "<div style='font-size:0.6rem;color:#00d4ff;font-weight:600;margin-bottom:3px'>ファンダメンタル</div>"
        "<div style='display:flex;align-items:center'>" + _fund_tiles + "</div>"
        "</div>"
    )

    # ── EDINETタイル ──
    _edinet_html = ""
    if _pedinet:
        _edinet_html = (
            "<div style='flex-shrink:0;padding-left:4px'>"
            "<div style='font-size:0.6rem;color:#00d4ff;font-weight:600;margin-bottom:3px'>&#x1F4C4; EDINET</div>"
            f"<div style='font-size:0.62rem;color:#aaa;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{_pedinet.get('docDescription', '—')}</div>"
            f"<div style='font-size:0.58rem;color:#666;margin-top:1px'>提出: {str(_pedinet.get('submitDateTime', ''))[:10]}</div>"
            "</div>"
        )

    # ── 縦並びパネルのrows HTML ──
    _rows_html = "".join(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:3px 0;border-bottom:1px solid #1a2236'>"
        f"<span style='font-size:0.68rem;color:#8892a4'>{lbl}</span>"
        f"<span style='font-size:0.72rem;color:#e8eaf0'>{_sig_icon(key)} {_fmt(key, unit)}</span>"
        f"</div>"
        for lbl, key, unit in _panel_rows
    )

    # ── ポートフォリオ縦並びHTML ──
    _portfolio_html = ""
    if _pbts:
        _pf_items = [
            ("初期投資額",   f"{_pic:,.0f}円",              "#e8eaf0", ""),
            ("現在の評価額", f"{_pfv:,.0f}円",              "#e8eaf0", ""),
            ("損益合計",     f"{_psign}{_pprofit:,.0f}円",  _pc,       "font-weight:600"),
            ("リターン率",   f"{_pretpct:+.2f}%",           _pc,       "font-weight:600"),
            ("保有株式数",   f"{_ppos:,}株",                "#e8eaf0", ""),
        ]
        _pf_rows_html = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:2px 0;"
            f"border-bottom:{'1px solid #1a2236' if i < len(_pf_items) - 1 else 'none'}'>"
            f"<span style='font-size:0.65rem;color:#8892a4'>{lbl}</span>"
            f"<span style='font-size:0.68rem;color:{col};{xtra}'>{val}</span>"
            f"</div>"
            for i, (lbl, val, col, xtra) in enumerate(_pf_items)
        )
        _portfolio_html = (
            "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #1e2d40;margin-bottom:8px'>"
            "<div style='font-size:0.7rem;color:#f0c040;font-weight:600;margin-bottom:4px'>&#x1F4B0; ポートフォリオ</div>"
            + _pf_rows_html
            + "</div>"
        )

    _panel_html = (
        "<div id='fund-right-panel' style='"
        "position:fixed;right:12px;bottom:12px;"
        "width:200px;background:#0d1321;"
        "border:1px solid #1e2d40;border-radius:10px;"
        "padding:12px;z-index:9999;"
        "max-height:70vh;overflow-y:auto;"
        "box-shadow:0 4px 20px rgba(0,0,0,0.5);"
        "'>"
        "<div style='font-size:0.7rem;color:#00d4ff;font-weight:700;letter-spacing:0.04em;margin-bottom:6px'>&#x1F4CA; 基本情報</div>"
        f"<div style='font-size:0.78rem;color:#e8eaf0;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{_pname}</div>"
        f"<div style='font-size:0.65rem;color:#666;margin-bottom:6px'>{_panel_ticker}</div>"
        f"<div style='font-size:0.7rem;color:{_score_color};font-weight:600;margin-bottom:4px'>総合スコア: {_pscore:+d}</div>"
        + _portfolio_html
        + _rows_html
        + _edinet_html
        + "</div>"
        "<style>"
        "[data-testid='stPlotlyChart'] { width: calc(100% - 220px) !important; }"
        ".main .block-container { padding-right: 220px !important; }"
        "</style>"
    )
    st.markdown(_panel_html, unsafe_allow_html=True)

for ticker in tickers:
    if ticker != st.session_state.get("active_ticker"):
        continue
    with st.container():
        # 銘柄別設定を取得（診断トグルON時は推奨値で上書き）
        _s = _get_effective_settings(ticker, period)
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
            df_ext = st.session_state[ext_df_key].copy()  # コピーで mutation を防止
            for _col in df_ext.columns:
                if _col not in df.columns:
                    df[_col] = df_ext[_col]
            df, _ext_sig_cols = generate_ext_signals(df, extra_cols, params=_s)
            if _s.get("use_context_strategy", False):
                df = generate_context_signal(df, active, _ext_sig_cols,
                                             score_threshold=_s.get("context_score_threshold", 5),
                                             rsi_ob=_s.get("rsi_ob", 70),
                                             rsi_os=_s.get("rsi_os", 30))
            else:
                df = merge_all_signals(df, active, _ext_sig_cols)
        else:
            if _s.get("use_context_strategy", False):
                df = generate_context_signal(df, active,
                                             score_threshold=_s.get("context_score_threshold", 5),
                                             rsi_ob=_s.get("rsi_ob", 70),
                                             rsi_os=_s.get("rsi_os", 30))
            else:
                df = generate_composite_signal(df, active)

        # ── ファンダメンタルデータ取得（バックテスト前に実施）──
        _is_jp_stock = ticker.upper().endswith(".T")
        _fund_settings = _get_fund_settings(ticker)
        _fund_data = get_fundamental_data_cached(ticker)
        _fund_result = calculate_fundamental_signals(_fund_data, _fund_settings)
        _fund_score = _fund_result["score"]
        _fund_count = _fund_result["enabled_count"]
        _fund_signals = _fund_result["signals"]
        _fund_integrate = _s.get("fund_integrate", False)

        # ── ファンダメンタル統合 ──
        # テクニカルの閾値は据え置き、fund_score を vote_sum への定数加算として扱う
        if _fund_integrate and _fund_count > 0 and "vote_sum" in df.columns:
            _use_ctx = _s.get("use_context_strategy", False)
            if _use_ctx:
                _threshold = _s.get("context_score_threshold", 5)
            else:
                _tech_col_count = len(active) + len(extra_cols)
                _threshold = max(1, _tech_col_count / 2)
            df["vote_sum"] = df["vote_sum"] + _fund_score
            df["composite_signal"] = 0
            df.loc[df["vote_sum"] >= _threshold, "composite_signal"] = 1
            df.loc[df["vote_sum"] <= -_threshold, "composite_signal"] = -1
            df["order"] = df["composite_signal"].diff()

        # バックテスト実行（銘柄別リスク・投資設定を使用）
        initial_cash = int(_s.get("initial_cash", 1_000_000))
        _max_shares  = int(_s.get("max_shares", 0))
        risk = RiskManager(
            stop_loss_pct=_s.get("stop_loss", 5),
            take_profit_pct=_s.get("take_profit", 10),
            max_position_pct=_s.get("max_pos", 100),
            rebuy_dip_pct=_s.get("rebuy_dip", 0),
        )
        result = run_backtest(df, initial_cash=initial_cash, risk=risk, max_shares=_max_shares)
        # サイドバーサマリー用キャッシュ
        st.session_state[f"bt_summary_{ticker}"] = {
            "initial_cash": result["initial_cash"],
            "final_value":  result["final_value"],
            "total_return_pct": result["total_return_pct"],
            "current_position": result["current_position"],
        }

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
        _tech_vote_count = len(active) + len(extra_cols)
        total_vote_count = _tech_vote_count + (_fund_count if _fund_integrate else 0)
        _vote_label = f"投票: {vote:+d}/{total_vote_count}" + (" (F統合)" if _fund_integrate and _fund_count > 0 else "")
        col2.metric("シグナル", signal_label[current_signal], _vote_label)
        col3.metric("リターン", f"{ret_pct:.2f}%",
                    f"{result['final_value'] - result['initial_cash']:+,.0f}円")
        col4.metric("最大DD", f"{result['max_drawdown_pct']:.2f}%")
        col5.metric("勝率", f"{result['win_rate_pct']:.1f}%")
        col6.metric("取引回数", f"{trade_count}回")

        # ─── チャート ───
        # 推奨チェックボックスのsession_state値を先読みしてチャートに反映
        _chart_ext_overlays: list[str] = list(extra_overlays)
        _chart_ext_oscillators: list[str] = list(extra_oscillators)
        _cc_key = f"corr_cache_{ticker}_{period}"
        if _cc_key in st.session_state:
            for _r in st.session_state[_cc_key].get("corr", []):
                _c = _r["col"]
                if _r["type"] == "overlay" and st.session_state.get(f"ext_ov_{ticker}_{_c}") and _c not in _chart_ext_overlays:
                    _chart_ext_overlays.append(_c)
                    if df_ext is not None and _c in df_ext.columns and _c not in df.columns:
                        df[_c] = df_ext[_c]
                elif _r["type"] != "overlay" and st.session_state.get(f"ext_osc_{ticker}_{_c}") and _c not in _chart_ext_oscillators:
                    _chart_ext_oscillators.append(_c)
                    if df_ext is not None and _c in df_ext.columns and _c not in df.columns:
                        df[_c] = df_ext[_c]
        _chart_ext_overlays    = list(dict.fromkeys(_chart_ext_overlays))
        _chart_ext_oscillators = list(dict.fromkeys(_chart_ext_oscillators))
        _trades_df = result.get("trades") if result else None
        fig = create_chart(df, ticker, ic,
                           ext_overlays=_chart_ext_overlays,
                           ext_oscillators=_chart_ext_oscillators,
                           trades=_trades_df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

        # ─── ポートフォリオ推移 ───
        if not result["portfolio_curve"].empty:
            fig_pf = create_portfolio_chart(result["portfolio_curve"], initial_cash)
            st.plotly_chart(fig_pf, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

        # ─── 空売り・スクイーズ分析 ───
        _sp_danger  = _s.get("sell_pressure_danger",  0.50)
        _sp_caution = _s.get("sell_pressure_caution", 0.40)
        _sq_high    = _s.get("squeeze_high", 0.50)
        _sq_mid     = _s.get("squeeze_mid",  0.35)

        sell_pressure_now = float(df["SELL_PRESSURE"].iloc[-1]) if "SELL_PRESSURE" in df.columns else 0.0
        squeeze_now       = float(df["SQUEEZE_SCORE"].iloc[-1])  if "SQUEEZE_SCORE" in df.columns else 0.0
        has_short_alert   = sell_pressure_now > _sp_danger or squeeze_now > _sq_high

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
            pressure_label = "危険" if sell_pressure_now > _sp_danger else ("警戒" if sell_pressure_now > _sp_caution else "正常")
            squeeze_label  = "高" if squeeze_now > _sq_high else ("中" if squeeze_now > _sq_mid else "低")
            sc3.metric("空売り圧力", f"{sell_pressure_now:.2f}", pressure_label)
            sc4.metric("スクイーズスコア", f"{squeeze_now:.2f}", f"発生確率: {squeeze_label}")

            # アラートメッセージ
            if sell_pressure_now > _sp_danger:
                st.warning(
                    "⚠️ **空売り圧力が高い状態です。** "
                    "大量売りを伴う下落が続いており、機関投資家による空売り積み増しの可能性があります。"
                    " 新規買いは慎重に。既存保有株の損切りラインを確認してください。"
                )
            if squeeze_now > _sq_high:
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
                fig_ss.add_hline(y=_sp_danger, line=dict(color="#f44336", dash="dash", width=1),
                                 annotation_text=f"売り圧危険({_sp_danger:.2f})")
                fig_ss.add_hline(y=_sp_caution, line=dict(color="#ff9800", dash="dash", width=1),
                                 annotation_text=f"売り圧警戒({_sp_caution:.2f})")
                fig_ss.add_hline(y=_sq_high, line=dict(color="#4caf50", dash="dash", width=1),
                                 annotation_text=f"スクイーズ高({_sq_high:.2f})")
                fig_ss.add_hline(y=_sq_mid, line=dict(color="#8bc34a", dash="dot", width=1),
                                 annotation_text=f"スクイーズ中({_sq_mid:.2f})")
                fig_ss.update_layout(
                    height=220, template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=60),
                    title="空売り圧力 / スクイーズスコア（0〜1）",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_ss, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

            st.caption(
                "空売り圧力：大出来高を伴う下落の累積強度（0〜1）。"
                "　スクイーズスコア：下落後の急騰＋大出来高の強度（0〜1）。"
                "　空売りカバー日数・空売り比率はyfinanceから取得（主に米国株対応）。"
            )

        # ─── ファンダメンタル分析 ───
        _fund_sig_label = (
            "🟢 ファンダメンタル優良" if _fund_score >= 2 else
            "🔴 ファンダメンタル懸念" if _fund_score <= -2 else
            "⚪ ニュートラル"
        )

        with st.expander(f"📋 ファンダメンタル分析  {_fund_sig_label}", expanded=(_fund_score >= 3 or _fund_score <= -3)):

            # ── 統合トグル ──
            _fund_toggle_col1, _fund_toggle_col2 = st.columns([2, 3])
            _fund_toggle_col1.toggle(
                "テクニカルシグナルに統合する",
                key=f"cfg_{ticker}_fund_integrate",
                on_change=_auto_save_setting,
                args=(ticker, "fund_integrate"),
                help=(
                    "ONにすると、ファンダメンタルスコアをテクニカル投票に加算して"
                    "複合シグナルを決定します。設定は自動保存されます。"
                ),
            )
            if _fund_integrate and _fund_count > 0:
                _fund_toggle_col2.info(
                    f"統合中: ファンダメンタルスコア **{_fund_score:+d}** を"
                    f"テクニカル投票 ({_tech_vote_count}票) に加算しています。"
                )
            elif _fund_count > 0:
                _fund_toggle_col2.caption(
                    f"現在スコア: {_fund_score:+d}/{_fund_count}　"
                    "(トグルONでバックテストに反映)"
                )

            st.divider()

            # EDINET接続ステータス
            if _is_jp_stock:
                _edinet_ok = check_edinet_connection()
                _edinet_col1, _edinet_col2 = st.columns([1, 3])
                if _edinet_ok:
                    _edinet_col1.success("✅ EDINET接続済")
                else:
                    _edinet_col1.error("❌ EDINET接続失敗")

                # EDINET最新報告書の検索ボタン
                _code4 = ticker.replace(".T", "").replace(".t", "")[:4]
                _edinet_cache = db.load_edinet_cache(_code4)
                if _edinet_cache:
                    _edinet_col2.caption(
                        f"最新報告書: {_edinet_cache.get('docDescription', '—')}  "
                        f"（提出日: {_edinet_cache.get('submitDateTime', '—')[:10]}）"
                    )
                else:
                    _edinet_col2.caption("EDINET報告書未検索")

                if st.button("🔍 EDINET最新報告書を検索", key=f"edinet_search_{ticker}"):
                    _pb = st.progress(0.0, text="EDINET検索中...")
                    def _edinet_cb(v: float, msg: str):
                        _pb.progress(min(v, 0.99), text=f"EDINET {msg}")
                    _doc = find_latest_filing(_code4, progress_cb=_edinet_cb)
                    _pb.empty()
                    if _doc:
                        db.save_edinet_cache(_code4, _doc)
                        st.success(
                            f"📄 {_doc.get('docDescription', '—')}  "
                            f"提出日: {_doc.get('submitDateTime', '—')[:10]}"
                        )
                        st.rerun()
                    else:
                        st.warning("EDINET報告書が見つかりませんでした。")

                st.divider()

            # ファンダメンタルスコアサマリー
            _fs_col1, _fs_col2, _fs_col3 = st.columns(3)
            _fs_col1.metric(
                "ファンダメンタルスコア",
                f"{_fund_score:+d} / {_fund_count}",
                _fund_sig_label.split(" ", 1)[-1] if _fund_count > 0 else "—",
            )
            _fs_col2.metric("PER", f"{_fund_data.get('per', '—')}倍" if _fund_data.get('per') else "—")
            _fs_col3.metric("PBR", f"{_fund_data.get('pbr', '—')}倍" if _fund_data.get('pbr') else "—")

            # 各指標の詳細
            st.markdown("**指標詳細**")
            _FUND_DISPLAY = [
                ("roe",        "ROE",      "%",  "高いほど良い"),
                ("roa",        "ROA",      "%",  "高いほど良い"),
                ("eps_growth", "EPS成長率", "%",  "プラス成長が良い"),
                ("rev_growth", "売上成長率", "%", "プラス成長が良い"),
                ("per",        "PER",      "倍", "低いほど割安"),
                ("pbr",        "PBR",      "倍", "低いほど割安"),
                ("op_margin",  "営業利益率", "%", "高いほど良い"),
                ("debt_ratio", "負債比率",  "",  "低いほど健全"),
            ]
            _detail_cols = st.columns(4)
            for _di, (_dkey, _dlbl, _dunit, _dhint) in enumerate(_FUND_DISPLAY):
                _dval = _fund_data.get(_dkey)
                _dsig = _fund_signals.get(_dkey, 0)
                _sig_icon = "🟢" if _dsig > 0 else ("🔴" if _dsig < 0 else "⚪")
                _val_str = (
                    f"{_dval:.1f}{_dunit}" if _dval is not None else "データなし"
                )
                _detail_cols[_di % 4].metric(
                    f"{_sig_icon} {_dlbl}",
                    _val_str,
                    help=_dhint,
                )

            st.caption(
                "データソース: yfinance（Yahoo Finance）　更新: 24時間ごと　"
                "※ 日本株は一部データが取得できない場合があります。"
            )

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
                st.dataframe(trades_display, use_container_width=True, hide_index=True)

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
            st.dataframe(tail_df.round(2), use_container_width=True)


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
                st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

            # 取引履歴（拡張）
            if not er["trades"].empty:
                with st.expander(f"拡張バックテスト取引履歴（{len(er['trades'])}件）"):
                    td = er["trades"].copy()
                    td["date"]   = pd.to_datetime(td["date"]).dt.strftime("%Y-%m-%d")
                    td["price"]  = td["price"].map("{:,.1f}円".format)
                    td["profit"] = td["profit"].apply(lambda x: f"{x:+,.0f}円" if pd.notna(x) else "-")
                    td.columns   = ["日付", "種別", "株価", "株数", "損益"]
                    st.dataframe(td, use_container_width=True, hide_index=True)

        # ─── 時間軸適合診断（アクティブ銘柄のみ）───
        if _s.get("show_timeframe_diagnosis", False):
            st.divider()
            _tf_key = f"tf_result_{ticker}_{period}"
            _tf_cname = _get_company_name(ticker)
            _tf_cname_str = f"  {_tf_cname}" if _tf_cname != ticker else ""
            st.subheader(f"🕐 時間軸適合診断: {ticker}{_tf_cname_str}")
            st.caption(
                "4種類のMA設定（超短期〜長期）でバックテストを比較し、"
                "この銘柄に適した時間軸とレジームを診断します。"
            )
            # キャッシュ自動ロード（当日更新済みかつbaseline収録済みならDBから即表示）
            if st.session_state.get(_tf_key) is None:
                _cached_tf = db.load_diagnosis_cache(ticker, "timeframe", period)
                if _cached_tf is not None and _cached_tf.get("baseline_return_pct") is not None:
                    st.session_state[_tf_key] = _cached_tf
                    if st.session_state.pop(f"_diag_auto_apply_{ticker}_timeframe", False):
                        _ab = _cached_tf.get("best_combined", {})
                        if _ab.get("short") and _ab.get("long"):
                            _apply_diag_and_clear_caches(ticker, {
                                "ma_short": _ab["short"], "ma_long": _ab["long"], "use_ma": True,
                            })
                            st.rerun()
                else:
                    _tf_df = st.session_state.get(f"df_{ticker}", None)
                    if _tf_df is None:
                        try:
                            _tf_df = get_stock_data(ticker, period=period)
                        except Exception as _e:
                            st.error(f"データ取得エラー: {_e}")
                            _tf_df = None
                    if _tf_df is not None:
                        with st.spinner("各MA設定でバックテスト実行中..."):
                            _tf_ic = _s.get("initial_cash", 1_000_000)
                            _tf_result = analyze_timeframe(_tf_df, initial_cash=_tf_ic, indicator_config=_build_diag_indicator_config(_s))
                            db.save_diagnosis_cache(ticker, "timeframe", period, _tf_result)
                            st.session_state[_tf_key] = _tf_result
                            if st.session_state.pop(f"_diag_auto_apply_{ticker}_timeframe", False):
                                _ab = _tf_result.get("best_combined", {})
                                if _ab.get("short") and _ab.get("long"):
                                    _apply_diag_and_clear_caches(ticker, {
                                        "ma_short": _ab["short"], "ma_long": _ab["long"], "use_ma": True,
                                    })
                                    st.rerun()

            if st.button("🔄 再診断", key=f"run_tf_{ticker}", help="最新データで再診断してDBに保存します"):
                _tf_df = st.session_state.get(f"df_{ticker}", None)
                if _tf_df is None:
                    try:
                        _tf_df = get_stock_data(ticker, period=period)
                    except Exception as _e:
                        st.error(f"データ取得エラー: {_e}")
                        _tf_df = None
                if _tf_df is not None:
                    with st.spinner("各MA設定でバックテスト実行中..."):
                        _tf_ic = _s.get("initial_cash", 1_000_000)
                        _tf_result = analyze_timeframe(_tf_df, initial_cash=_tf_ic, indicator_config=_build_diag_indicator_config(_s))
                        db.save_diagnosis_cache(ticker, "timeframe", period, _tf_result)
                        st.session_state[_tf_key] = _tf_result

            tf_res = st.session_state.get(_tf_key)
            # インライン確認UI（初回診断が無効だった場合）
            if st.session_state.get(f"_diag_inline_confirm_{ticker}_timeframe") and tf_res:
                st.warning(_diag_confirm_message(tf_res))
                _ic1, _ic2 = st.columns(2)
                if _ic1.button("はい", key=f"inline_ok_tf_{ticker}"):
                    _ib = (tf_res or {}).get("best_combined", {})
                    _iupd = {"show_timeframe_diagnosis": True}
                    if _ib.get("short") and _ib.get("long"):
                        _iupd["ma_short"] = _ib["short"]; _iupd["ma_long"] = _ib["long"]; _iupd["use_ma"] = True
                    _apply_diag_and_clear_caches(ticker, _iupd)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_timeframe", None)
                    st.rerun()
                if _ic2.button("いいえ", key=f"inline_ng_tf_{ticker}"):
                    _irv = {**db.DEFAULT_SETTINGS, **db.load_settings(ticker)}
                    _irv["show_timeframe_diagnosis"] = False
                    _irv["snap_tf_ma_short"] = None
                    _irv["snap_tf_ma_long"]  = None
                    db.save_settings(ticker, _irv)
                    st.session_state.pop(f"_cfg_init_{ticker}", None)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_timeframe", None)
                    st.rerun()
            if tf_res:
                if tf_res.get("_cached_at"):
                    st.caption(f"キャッシュ使用（本日 {tf_res['_cached_at']} 更新）")
                if "error" in tf_res:
                    st.error(tf_res["error"])
                else:
                    # ── 総合判定 ──
                    regime = tf_res["regime"]
                    best = tf_res["best_combined"]
                    rlabel = tf_res["regime_label"]
                    dom = regime["dominant"]
                    badge = "📈" if dom == "SHORT_TERM" else ("📉" if dom == "LONG_TERM" else "↔️")

                    _rc1, _rc2 = st.columns([1, 2])
                    _rc1.metric("レジーム判定", f"{badge} {rlabel}")
                    _rc2.metric(
                        "推奨MA設定",
                        best["label"],
                        f"リターン {best['return_pct']:+.2f}% / PF {best['profit_factor']:.2f}",
                    )

                    # ── MA設定別バックテスト比較表 ──
                    st.markdown("**MA設定別バックテスト比較**")
                    _cfg_rows = []
                    for c in tf_res["configs"]:
                        is_best = c["label"] == best["label"]
                        _cfg_rows.append({
                            "MA設定":     ("★ " if is_best else "") + c["label"],
                            "リターン(%)": c["return_pct"],
                            "勝率(%)":    c["win_rate_pct"],
                            "取引回数":   c["trade_count"],
                            "PF":         c["profit_factor"],
                            "最大DD(%)":  c["max_dd_pct"],
                        })
                    _tf_baseline_row = {
                        "MA設定":     "📍 " + tf_res.get("baseline_label", "現在の設定"),
                        "リターン(%)": tf_res.get("baseline_return_pct", 0.0),
                        "勝率(%)":    "-", "取引回数": "-", "PF": "-", "最大DD(%)": "-",
                    }
                    _cfg_df = pd.DataFrame([_tf_baseline_row] + _cfg_rows)

                    def _color_return(val):
                        if isinstance(val, float):
                            return f"color: {'#00cc66' if val > 0 else '#ff4444'}"
                        return ""

                    st.dataframe(
                        _cfg_df.style.map(_color_return, subset=["リターン(%)", "最大DD(%)"]),
                        hide_index=True,
                        use_container_width=True,
                    )

                    # ── レジームスコア詳細 ──
                    st.markdown("**レジームスコア詳細**")
                    _score_rows = []
                    _label_map = {
                        "ma_cross":      "MAクロス頻度",
                        "bb_width":      "BB幅変動",
                        "rsi_behavior":  "RSI挙動",
                        "macd_behavior": "MACDゼロライン",
                    }
                    for k, v in regime["details"].items():
                        _score_rows.append({
                            "指標":       _label_map.get(k, k),
                            "短期スコア": v["short"],
                            "長期スコア": v["long"],
                            "詳細":       v["desc"],
                        })
                    _score_rows.append({
                        "指標": "合計",
                        "短期スコア": regime["short_score"],
                        "長期スコア": regime["long_score"],
                        "詳細": f"最大{regime['max_score']}点",
                    })
                    st.dataframe(pd.DataFrame(_score_rows), hide_index=True, use_container_width=True)

                    st.caption(
                        "⚠️ この診断は過去データに基づく参考値です。"
                        " 相場の状態は変化するため、定期的に再診断してください。"
                    )

        # ─── RSI閾値適合診断（アクティブ銘柄のみ）───
        if _s.get("show_rsi_diagnosis", False):
            st.divider()
            _rsi_key = f"rsi_result_{ticker}_{period}"
            _rsi_cname = _get_company_name(ticker)
            _rsi_cname_str = f"  {_rsi_cname}" if _rsi_cname != ticker else ""
            st.subheader(f"📊 RSI閾値適合診断: {ticker}{_rsi_cname_str}")
            st.caption(
                "4種類のRSI閾値設定でバックテストを比較し、"
                "この銘柄が早期反転型か遅延反転型かを診断します。"
            )
            # キャッシュ自動ロード（当日更新済みかつbaseline収録済みならDBから即表示）
            if st.session_state.get(_rsi_key) is None:
                _cached_rsi = db.load_diagnosis_cache(ticker, "rsi", period)
                if (_cached_rsi is not None
                        and _cached_rsi.get("baseline_return_pct") is not None
                        and _cached_rsi.get("diag_version", 1) >= 2):
                    st.session_state[_rsi_key] = _cached_rsi
                    if st.session_state.pop(f"_diag_auto_apply_{ticker}_rsi", False):
                        _ab = _cached_rsi.get("best_combined", {})
                        if _ab.get("ob") and _ab.get("os"):
                            _apply_diag_and_clear_caches(ticker, {
                                "rsi_ob": _ab["ob"], "rsi_os": _ab["os"],
                            })
                            st.rerun()
                else:
                    _rsi_df = st.session_state.get(f"df_{ticker}", None)
                    if _rsi_df is None:
                        try:
                            _rsi_df = get_stock_data(ticker, period=period)
                        except Exception as _e:
                            st.error(f"データ取得エラー: {_e}")
                            _rsi_df = None
                    if _rsi_df is not None:
                        with st.spinner("各RSI閾値でバックテスト実行中..."):
                            _rsi_ic = _s.get("initial_cash", 1_000_000)
                            _rsi_result = analyze_rsi(_rsi_df, initial_cash=_rsi_ic, indicator_config=_build_diag_indicator_config(_s))
                            db.save_diagnosis_cache(ticker, "rsi", period, _rsi_result)
                            st.session_state[_rsi_key] = _rsi_result
                            if st.session_state.pop(f"_diag_auto_apply_{ticker}_rsi", False):
                                _ab = _rsi_result.get("best_combined", {})
                                if _ab.get("ob") and _ab.get("os"):
                                    _apply_diag_and_clear_caches(ticker, {
                                        "rsi_ob": _ab["ob"], "rsi_os": _ab["os"],
                                    })
                                    st.rerun()

            if st.button("🔄 再診断", key=f"run_rsi_{ticker}", help="最新データで再診断してDBに保存します"):
                _rsi_df = st.session_state.get(f"df_{ticker}", None)
                if _rsi_df is None:
                    try:
                        _rsi_df = get_stock_data(ticker, period=period)
                    except Exception as _e:
                        st.error(f"データ取得エラー: {_e}")
                        _rsi_df = None
                if _rsi_df is not None:
                    with st.spinner("各RSI閾値でバックテスト実行中..."):
                        _rsi_ic = _s.get("initial_cash", 1_000_000)
                        _rsi_result = analyze_rsi(_rsi_df, initial_cash=_rsi_ic, indicator_config=_build_diag_indicator_config(_s))
                        db.save_diagnosis_cache(ticker, "rsi", period, _rsi_result)
                        st.session_state[_rsi_key] = _rsi_result

            rsi_res = st.session_state.get(_rsi_key)
            # インライン確認UI（初回診断が無効だった場合）
            if st.session_state.get(f"_diag_inline_confirm_{ticker}_rsi") and rsi_res:
                st.warning(_diag_confirm_message(rsi_res))
                _ic1, _ic2 = st.columns(2)
                if _ic1.button("はい", key=f"inline_ok_rsi_{ticker}"):
                    _ib = (rsi_res or {}).get("best_combined", {})
                    _iupd = {"show_rsi_diagnosis": True}
                    if _ib.get("ob") and _ib.get("os"):
                        _iupd["rsi_ob"] = _ib["ob"]; _iupd["rsi_os"] = _ib["os"]
                    _apply_diag_and_clear_caches(ticker, _iupd)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_rsi", None)
                    st.rerun()
                if _ic2.button("いいえ", key=f"inline_ng_rsi_{ticker}"):
                    _irv = {**db.DEFAULT_SETTINGS, **db.load_settings(ticker)}
                    _irv["show_rsi_diagnosis"] = False
                    _irv["snap_rsi_ob"] = None
                    _irv["snap_rsi_os"] = None
                    db.save_settings(ticker, _irv)
                    st.session_state.pop(f"_cfg_init_{ticker}", None)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_rsi", None)
                    st.rerun()
            if rsi_res:
                if rsi_res.get("_cached_at"):
                    st.caption(f"キャッシュ使用（本日 {rsi_res['_cached_at']} 更新）")
                if "error" in rsi_res:
                    st.error(rsi_res["error"])
                else:
                    reversal = rsi_res["reversal"]
                    best = rsi_res["best_combined"]
                    rlabel = rsi_res["reversal_label"]
                    dom = reversal["dominant"]
                    badge = "⚡" if dom == "EARLY_REVERSAL" else ("🐢" if dom == "LATE_REVERSAL" else "↔️")

                    _rc1, _rc2 = st.columns([1, 2])
                    _rc1.metric("銘柄タイプ", f"{badge} {rlabel}")
                    _rc2.metric(
                        "推奨RSI設定",
                        best["label"],
                        f"リターン {best['return_pct']:+.2f}% / PF {best['profit_factor']:.2f}",
                    )

                    # ── 全プリセットがマイナスリターンの場合の警告 ──
                    _all_negative_rsi = all(c["return_pct"] <= 0 for c in rsi_res["configs"])
                    if _all_negative_rsi:
                        st.warning(
                            "⚠️ 全プリセットでリターンがマイナスです。"
                            " 診断トグルをONにして「はい」を選択すると、診断値を適用できます"
                            "（パフォーマンスは保証されません）。"
                        )

                    # ── RSI設定別バックテスト比較表 ──
                    def _color_rsi_val(val):
                        if isinstance(val, float):
                            return f"color: {'#00cc66' if val > 0 else '#ff4444'}"
                        return ""

                    st.markdown("**RSI閾値別バックテスト比較**")
                    _rsi_rows = []
                    for c in rsi_res["configs"]:
                        is_best = c["label"] == best["label"]
                        _rsi_rows.append({
                            "RSI設定":    ("★ " if is_best else "") + c["label"],
                            "リターン(%)": c["return_pct"],
                            "勝率(%)":    c["win_rate_pct"],
                            "取引回数":   c["trade_count"],
                            "PF":         c["profit_factor"],
                            "最大DD(%)":  c["max_dd_pct"],
                        })
                    _rsi_baseline_row = {
                        "RSI設定":    "📍 " + rsi_res.get("baseline_label", "現在の設定"),
                        "リターン(%)": rsi_res.get("baseline_return_pct", 0.0),
                        "勝率(%)":    "-", "取引回数": "-", "PF": "-", "最大DD(%)": "-",
                    }
                    _rsi_df2 = pd.DataFrame([_rsi_baseline_row] + _rsi_rows)
                    st.dataframe(
                        _rsi_df2.style.map(_color_rsi_val, subset=["リターン(%)", "最大DD(%)"]),
                        hide_index=True,
                        use_container_width=True,
                    )

                    # ── 反応速度スコア詳細 ──
                    st.markdown("**反応速度スコア詳細**")
                    _rev_rows = []
                    for k, v in reversal["details"].items():
                        _rev_rows.append({
                            "指標":  {"zone_ratio": "60/40ゾーン比率", "continuation": "70/30への進展率",
                                      "touch_frequency": "タッチ頻度"}.get(k, k),
                            "値":    v["value"],
                            "詳細":  v["desc"],
                        })
                    st.dataframe(pd.DataFrame(_rev_rows), hide_index=True, use_container_width=True)

                    st.caption(
                        "⚠️ この診断は過去データに基づく参考値です。"
                        " 適用後はコンテキスト方式をONにすると選択した閾値が使用されます。"
                    )

        # ─── MACDパラメータ適合診断（アクティブ銘柄のみ）───
        if _s.get("show_macd_diagnosis", False):
            st.divider()
            _macd_key = f"macd_result_{ticker}_{period}"
            _macd_cname = _get_company_name(ticker)
            _macd_cname_str = f"  {_macd_cname}" if _macd_cname != ticker else ""
            st.subheader(f"📊 MACDパラメータ診断: {ticker}{_macd_cname_str}")
            st.caption(
                "4種類のMACDパラメータ設定でバックテストを比較し、"
                "この銘柄が短期クロス型か長期トレンド型かを診断します。"
            )
            # キャッシュ自動ロード（当日更新済みかつbaseline収録済みならDBから即表示）
            if st.session_state.get(_macd_key) is None:
                _cached_macd = db.load_diagnosis_cache(ticker, "macd", period)
                if _cached_macd is not None and _cached_macd.get("baseline_return_pct") is not None:
                    st.session_state[_macd_key] = _cached_macd
                    if st.session_state.pop(f"_diag_auto_apply_{ticker}_macd", False):
                        _ab = _cached_macd.get("best_combined", {})
                        if _ab.get("fast") and _ab.get("slow") and _ab.get("sig"):
                            _apply_diag_and_clear_caches(ticker, {
                                "macd_fast": _ab["fast"], "macd_slow": _ab["slow"], "macd_sig": _ab["sig"],
                            })
                            st.rerun()
                else:
                    _macd_df = st.session_state.get(f"df_{ticker}", None)
                    if _macd_df is None:
                        try:
                            _macd_df = get_stock_data(ticker, period=period)
                        except Exception as _e:
                            st.error(f"データ取得エラー: {_e}")
                            _macd_df = None
                    if _macd_df is not None:
                        with st.spinner("各MACDパラメータでバックテスト実行中..."):
                            _macd_ic  = _s.get("initial_cash", 1_000_000)
                            _macd_cfg = _build_diag_indicator_config(_s)
                            _macd_result = analyze_macd(_macd_df, initial_cash=_macd_ic, indicator_config=_macd_cfg)
                            db.save_diagnosis_cache(ticker, "macd", period, _macd_result)
                            st.session_state[_macd_key] = _macd_result
                            if st.session_state.pop(f"_diag_auto_apply_{ticker}_macd", False):
                                _ab = _macd_result.get("best_combined", {})
                                if _ab.get("fast") and _ab.get("slow") and _ab.get("sig"):
                                    _apply_diag_and_clear_caches(ticker, {
                                        "macd_fast": _ab["fast"], "macd_slow": _ab["slow"], "macd_sig": _ab["sig"],
                                    })
                                    st.rerun()

            if st.button("🔄 再診断", key=f"run_macd_{ticker}", help="最新データで再診断してDBに保存します"):
                _macd_df = st.session_state.get(f"df_{ticker}", None)
                if _macd_df is None:
                    try:
                        _macd_df = get_stock_data(ticker, period=period)
                    except Exception as _e:
                        st.error(f"データ取得エラー: {_e}")
                        _macd_df = None
                if _macd_df is not None:
                    with st.spinner("各MACDパラメータでバックテスト実行中..."):
                        _macd_ic  = _s.get("initial_cash", 1_000_000)
                        _macd_cfg = _build_diag_indicator_config(_s)
                        _macd_result = analyze_macd(_macd_df, initial_cash=_macd_ic, indicator_config=_macd_cfg)
                        db.save_diagnosis_cache(ticker, "macd", period, _macd_result)
                        st.session_state[_macd_key] = _macd_result

            macd_res = st.session_state.get(_macd_key)
            # インライン確認UI（初回診断が無効だった場合）
            if st.session_state.get(f"_diag_inline_confirm_{ticker}_macd") and macd_res:
                st.warning(_diag_confirm_message(macd_res))
                _ic1, _ic2 = st.columns(2)
                if _ic1.button("はい", key=f"inline_ok_macd_{ticker}"):
                    _ib = (macd_res or {}).get("best_combined", {})
                    _iupd = {"show_macd_diagnosis": True}
                    if _ib.get("fast") and _ib.get("slow") and _ib.get("sig"):
                        _iupd["macd_fast"] = _ib["fast"]; _iupd["macd_slow"] = _ib["slow"]; _iupd["macd_sig"] = _ib["sig"]
                    _apply_diag_and_clear_caches(ticker, _iupd)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_macd", None)
                    st.rerun()
                if _ic2.button("いいえ", key=f"inline_ng_macd_{ticker}"):
                    _irv = {**db.DEFAULT_SETTINGS, **db.load_settings(ticker)}
                    _irv["show_macd_diagnosis"] = False
                    _irv["snap_macd_fast"] = None
                    _irv["snap_macd_slow"] = None
                    _irv["snap_macd_sig"]  = None
                    db.save_settings(ticker, _irv)
                    st.session_state.pop(f"_cfg_init_{ticker}", None)
                    st.session_state.pop(f"_diag_inline_confirm_{ticker}_macd", None)
                    st.rerun()
            if macd_res:
                if macd_res.get("_cached_at"):
                    st.caption(f"キャッシュ使用（本日 {macd_res['_cached_at']} 更新）")
                if "error" in macd_res:
                    st.error(macd_res["error"])
                else:
                    cross = macd_res["cross"]
                    best = macd_res["best_combined"]
                    clabel = macd_res["cross_label"]
                    dom = cross["dominant"]
                    badge = "⚡" if dom == "SHORT_TERM" else ("🐢" if dom == "LONG_TERM" else "↔️")

                    _mc1, _mc2 = st.columns([1, 2])
                    _mc1.metric("銘柄タイプ", f"{badge} {clabel}")
                    _mc2.metric(
                        "推奨MACD設定",
                        best["label"],
                        f"リターン {best['return_pct']:+.2f}% / PF {best['profit_factor']:.2f}",
                    )

                    # ── ATRベースSL情報 ──
                    _macd_atr = macd_res.get("atr_pct")
                    _macd_sl  = macd_res.get("stop_loss_pct")
                    if _macd_atr is not None and _macd_sl is not None:
                        st.caption(
                            f"診断用SL: {_macd_sl:.1f}%（ATR {_macd_atr:.1f}% × 2倍、最小5%・最大15%）"
                        )

                    # ── 全プリセットがマイナスリターンの場合の警告 ──
                    _all_negative = all(c["return_pct"] <= 0 for c in macd_res["configs"])
                    if _all_negative:
                        st.warning(
                            "⚠️ 全プリセットでリターンがマイナスです。"
                            " 診断トグルをONにして「はい」を選択すると、診断値を適用できます"
                            "（パフォーマンスは保証されません）。"
                        )

                    # ── MACDパラメータ別バックテスト比較表 ──
                    def _color_macd_val(val):
                        if isinstance(val, float):
                            return f"color: {'#00cc66' if val > 0 else '#ff4444'}"
                        return ""

                    st.markdown("**MACDパラメータ別バックテスト比較**")
                    _macd_rows = []
                    for c in macd_res["configs"]:
                        is_best = c["label"] == best["label"]
                        _macd_rows.append({
                            "MACD設定":   ("★ " if is_best else "") + c["label"],
                            "リターン(%)": c["return_pct"],
                            "勝率(%)":    c["win_rate_pct"],
                            "取引回数":   c["trade_count"],
                            "PF":         c["profit_factor"],
                            "最大DD(%)":  c["max_dd_pct"],
                        })
                    _macd_baseline_row = {
                        "MACD設定":   "📍 " + macd_res.get("baseline_label", "現在の設定"),
                        "リターン(%)": macd_res.get("baseline_return_pct", 0.0),
                        "勝率(%)":    "-", "取引回数": "-", "PF": "-", "最大DD(%)": "-",
                    }
                    _macd_df2 = pd.DataFrame([_macd_baseline_row] + _macd_rows)
                    st.dataframe(
                        _macd_df2.style.map(_color_macd_val, subset=["リターン(%)", "最大DD(%)"]),
                        hide_index=True,
                        use_container_width=True,
                    )

                    # ── クロス傾向スコア詳細 ──
                    st.markdown("**クロス傾向スコア詳細**")
                    _cross_rows = []
                    for k, v in cross["details"].items():
                        _cross_rows.append({
                            "指標":  {"cross_freq": "クロス頻度", "zero_run": "ゼロライン滞在"}.get(k, k),
                            "値":    v["value"],
                            "詳細":  v["desc"],
                        })
                    st.dataframe(pd.DataFrame(_cross_rows), hide_index=True, use_container_width=True)

                    st.caption(
                        "⚠️ この診断は過去データに基づく参考値です。"
                        " 診断トグルがONの間は推奨設定が自動で有効設定に反映されます。"
                    )

        st.divider()

        # ─── テクニカル指標おすすめ（相関アラート）───
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
                        df_bt, sig_cols = generate_ext_signals(df_ext, all_checked, params=_s)
                        df_bt = build_ext_composite(df_bt, sig_cols)
                        ext_result = run_backtest(
                            df_bt, initial_cash, risk,
                            signal_col="ext_composite_signal",
                            order_col="ext_order",
                            max_shares=_max_shares,
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

# ─────────────────────────────────────────────
# フッター
# ─────────────────────────────────────────────
st.divider()
if auto_refresh:
    from datetime import datetime
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {refresh_sec}秒ごとに自動更新中")
else:
    st.caption("自動更新はオフです。サイドバーの「リアルタイム更新」をオンにすると定期更新されます。")
