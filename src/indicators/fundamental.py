"""ファンダメンタル指標 — yfinanceから取得し、売買シグナルを生成する。"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ── デフォルト閾値設定 ──────────────────────────────────────────────────────
DEFAULT_FUND_SETTINGS: dict = {
    # ROE（自己資本利益率）
    "use_roe": True,
    "roe_buy": 15.0,    # ROE >= この値なら買い (%)
    "roe_sell": 5.0,    # ROE <= この値なら売り (%)

    # ROA（総資産利益率）
    "use_roa": True,
    "roa_buy": 8.0,
    "roa_sell": 2.0,

    # EPS成長率
    "use_eps_growth": True,
    "eps_growth_buy": 10.0,     # 前年比成長率 >= この値なら買い (%)
    "eps_growth_sell": -5.0,    # 前年比成長率 <= この値なら売り (%)

    # 売上高成長率
    "use_rev_growth": True,
    "rev_growth_buy": 5.0,
    "rev_growth_sell": -3.0,

    # PER（株価収益率）
    "use_per": True,
    "per_buy": 15.0,    # PER <= この値なら買い（割安）
    "per_sell": 35.0,   # PER >= この値なら売り（割高）

    # PBR（株価純資産倍率）
    "use_pbr": True,
    "pbr_buy": 1.0,
    "pbr_sell": 3.0,

    # 営業利益率
    "use_op_margin": True,
    "op_margin_buy": 10.0,
    "op_margin_sell": 3.0,

    # 負債比率（D/E比 × 100、yfinance形式）
    "use_debt_ratio": True,
    "debt_ratio_buy": 30.0,     # D/E < 0.3 → 買い（財務健全）
    "debt_ratio_sell": 150.0,   # D/E > 1.5 → 売り（高負債）
}

# 指標名の日本語ラベル
FUND_LABELS: dict[str, str] = {
    "roe":        "ROE（自己資本利益率）",
    "roa":        "ROA（総資産利益率）",
    "eps_growth": "EPS成長率",
    "rev_growth": "売上高成長率",
    "per":        "PER（株価収益率）",
    "pbr":        "PBR（株価純資産倍率）",
    "op_margin":  "営業利益率",
    "debt_ratio": "負債比率（D/E）",
}

# 各指標の単位
FUND_UNITS: dict[str, str] = {
    "roe": "%", "roa": "%", "eps_growth": "%", "rev_growth": "%",
    "per": "倍", "pbr": "倍", "op_margin": "%", "debt_ratio": "",
}


def _to_pct(val) -> Optional[float]:
    """小数値（0.15）をパーセント（15.0）に変換。Noneはそのまま。"""
    if val is None:
        return None
    try:
        return round(float(val) * 100, 2)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


def get_fundamental_data(ticker: str) -> dict:
    """yfinanceからファンダメンタルデータを取得する。

    Returns:
        roe, roa, eps_growth, rev_growth, per, pbr, op_margin, debt_ratio,
        eps, market_cap, sector, currency, fetched_at, source
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning(f"yfinance info fetch failed for {ticker}: {e}")
        return {"error": str(e)}

    return {
        "roe":        _to_pct(info.get("returnOnEquity")),
        "roa":        _to_pct(info.get("returnOnAssets")),
        "eps_growth": _to_pct(info.get("earningsGrowth")),
        "rev_growth": _to_pct(info.get("revenueGrowth")),
        "per":        _safe_float(info.get("trailingPE")),
        "pbr":        _safe_float(info.get("priceToBook")),
        "op_margin":  _to_pct(info.get("operatingMargins")),
        # yfinanceのdebtToEquityは既に×100の値（D/E=1.5 → 150.0）
        "debt_ratio": _safe_float(info.get("debtToEquity")),
        "eps":        _safe_float(info.get("trailingEps")),
        "market_cap": info.get("marketCap"),
        "sector":     info.get("sector") or info.get("industryDisp", ""),
        "currency":   info.get("currency", "JPY"),
        "fetched_at": datetime.now().isoformat(),
        "source":     "yfinance",
    }


def calculate_fundamental_signals(fund_data: dict, settings: dict) -> dict:
    """ファンダメンタルデータからシグナルを計算する。

    各指標が閾値を満たすと +1（買い）/ -1（売り）/ 0（中立）を返す。

    Returns:
        {
            "signals":       {"roe": 1, "roa": -1, ...},
            "score":         合計スコア（正=買い優位、負=売り優位）,
            "enabled_count": 有効指標数,
        }
    """
    s = {**DEFAULT_FUND_SETTINGS, **settings}
    d = fund_data
    signals: dict[str, int] = {}

    def _sig_high(key: str, buy_k: str, sell_k: str) -> int:
        """高いほど良い指標（ROE, ROA等）のシグナル。"""
        val = d.get(key)
        if val is None:
            return 0
        if val >= s[buy_k]:
            return 1
        if val <= s[sell_k]:
            return -1
        return 0

    def _sig_low(key: str, buy_k: str, sell_k: str) -> int:
        """低いほど良い指標（PER, PBR, 負債比率）のシグナル。"""
        val = d.get(key)
        if val is None:
            return 0
        if val is not None and val > 0 and val <= s[buy_k]:
            return 1
        if val is not None and val >= s[sell_k]:
            return -1
        return 0

    if s.get("use_roe"):
        signals["roe"] = _sig_high("roe", "roe_buy", "roe_sell")
    if s.get("use_roa"):
        signals["roa"] = _sig_high("roa", "roa_buy", "roa_sell")
    if s.get("use_eps_growth"):
        signals["eps_growth"] = _sig_high("eps_growth", "eps_growth_buy", "eps_growth_sell")
    if s.get("use_rev_growth"):
        signals["rev_growth"] = _sig_high("rev_growth", "rev_growth_buy", "rev_growth_sell")
    if s.get("use_per"):
        signals["per"] = _sig_low("per", "per_buy", "per_sell")
    if s.get("use_pbr"):
        signals["pbr"] = _sig_low("pbr", "pbr_buy", "pbr_sell")
    if s.get("use_op_margin"):
        signals["op_margin"] = _sig_high("op_margin", "op_margin_buy", "op_margin_sell")
    if s.get("use_debt_ratio"):
        signals["debt_ratio"] = _sig_low("debt_ratio", "debt_ratio_buy", "debt_ratio_sell")

    score = sum(signals.values())

    return {
        "signals":       signals,
        "score":         score,
        "enabled_count": len(signals),
    }
