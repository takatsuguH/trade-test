"""
7通り組み合わせ診断（グリッドサーチ）

3指標（MA / RSI / MACD）の任意の組み合わせに対して、各指標のプリセット
すべての組み合わせをグリッドサーチでバックテストし、最良パラメータを返す。

7通りの組み合わせ:
    {MA}, {RSI}, {MACD}, {MA,RSI}, {MA,MACD}, {RSI,MACD}, {MA,RSI,MACD}

BB（ボリンジャーバンド）は含まない（サイドバーから独立した「資料」としての位置づけ）。
"""
from itertools import product
from typing import Iterable
import pandas as pd

from src.indicators.calculator import calculate_all
from src.strategies.composite import generate_composite_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager

# プリセット（既存の analyze_* と同一値）
MA_PRESETS: list[dict] = [
    {"label": "(5/25)",   "ma_short": 5,  "ma_long": 25},
    {"label": "(10/50)",  "ma_short": 10, "ma_long": 50},
    {"label": "(25/75)",  "ma_short": 25, "ma_long": 75},
    {"label": "(50/200)", "ma_short": 50, "ma_long": 200},
]

RSI_PRESETS: list[dict] = [
    {"label": "(70/30)", "rsi_ob": 70, "rsi_os": 30},
    {"label": "(65/32)", "rsi_ob": 65, "rsi_os": 32},
    {"label": "(60/35)", "rsi_ob": 60, "rsi_os": 35},
    {"label": "(55/40)", "rsi_ob": 55, "rsi_os": 40},
]

MACD_PRESETS: list[dict] = [
    {"label": "(5/20/5)",  "macd_fast": 5,  "macd_slow": 20, "macd_sig": 5},
    {"label": "(12/26/9)", "macd_fast": 12, "macd_slow": 26, "macd_sig": 9},
    {"label": "(20/50/9)", "macd_fast": 20, "macd_slow": 50, "macd_sig": 9},
    {"label": "(25/75/9)", "macd_fast": 25, "macd_slow": 75, "macd_sig": 9},
]

# 7通り定義: combo_key と必要指標セット
SEVEN_COMBINATIONS: list[tuple[str, frozenset[str], str]] = [
    ("ma",            frozenset({"MA"}),                 "MA単独"),
    ("rsi",           frozenset({"RSI"}),                "RSI単独"),
    ("macd",          frozenset({"MACD"}),               "MACD単独"),
    ("ma_rsi",        frozenset({"MA", "RSI"}),          "MA+RSI"),
    ("ma_macd",       frozenset({"MA", "MACD"}),         "MA+MACD"),
    ("rsi_macd",      frozenset({"RSI", "MACD"}),        "RSI+MACD"),
    ("ma_rsi_macd",   frozenset({"MA", "RSI", "MACD"}),  "MA+RSI+MACD"),
]

# RSI/MACD/BB計算用のデフォルト固定値（試験対象外の指標で使う）
_FIXED_RSI_PERIOD = 14
_FIXED_BB_PERIOD  = 20
_FIXED_BB_STD     = 2.0


def _profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    closed = trades_df[trades_df["profit"].notna()]
    if closed.empty:
        return 0.0
    wins = closed[closed["profit"] > 0]["profit"].sum()
    losses = closed[closed["profit"] < 0]["profit"].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / abs(losses), 2)


def _build_cfg(indicators: frozenset[str], params: dict) -> dict:
    """指標セットと試験パラメータから calculate_all 用の cfg を構築。"""
    cfg: dict = {
        "use_ma":   "MA"   in indicators,
        "use_rsi":  "RSI"  in indicators,
        "use_macd": "MACD" in indicators,
        "use_bb":   False,
        "rsi_period": _FIXED_RSI_PERIOD,
        "bb_period":  _FIXED_BB_PERIOD,
        "bb_std":     _FIXED_BB_STD,
    }
    if "MA" in indicators:
        cfg["ma_short"] = params["ma_short"]
        cfg["ma_long"]  = params["ma_long"]
    if "RSI" in indicators:
        cfg["rsi_ob"] = params["rsi_ob"]
        cfg["rsi_os"] = params["rsi_os"]
    if "MACD" in indicators:
        cfg["macd_fast"] = params["macd_fast"]
        cfg["macd_slow"] = params["macd_slow"]
        cfg["macd_sig"]  = params["macd_sig"]
    return cfg


def _run_combo_backtest(
    df: pd.DataFrame, indicators: frozenset[str], params: dict, initial_cash: float,
) -> dict:
    """指定パラメータの組み合わせで複合シグナルBTを実行。"""
    cfg = _build_cfg(indicators, params)
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    d = calculate_all(d, cfg)
    active = [ind for ind in ("MA", "RSI", "MACD") if ind in indicators]
    d = generate_composite_signal(d, active)
    risk = RiskManager(stop_loss_pct=5.0, take_profit_pct=10.0, max_position_pct=100.0)
    result = run_backtest(d, initial_cash=initial_cash, risk=risk)
    pf = _profit_factor(result["trades"])
    return {
        "return_pct":    round(result["total_return_pct"], 2),
        "win_rate_pct":  round(result["win_rate_pct"], 1),
        "trade_count":   len(result["trades"]),
        "profit_factor": pf,
        "max_dd_pct":    round(result["max_drawdown_pct"], 2),
    }


def _iter_param_combinations(indicators: frozenset[str]) -> Iterable[dict]:
    """指標セットに含まれる指標のプリセットの直積を生成する。"""
    preset_lists: list[list[dict]] = []
    if "MA" in indicators:
        preset_lists.append(MA_PRESETS)
    if "RSI" in indicators:
        preset_lists.append(RSI_PRESETS)
    if "MACD" in indicators:
        preset_lists.append(MACD_PRESETS)
    for combo in product(*preset_lists):
        merged: dict = {}
        for p in combo:
            merged.update(p)
        yield merged


def analyze_combination(
    df: pd.DataFrame,
    indicators: frozenset[str],
    initial_cash: float = 1_000_000,
) -> dict:
    """
    任意の指標セット（MA/RSI/MACDの部分集合）でグリッドサーチBT。

    Args:
        df:           OHLCVデータ（最低60日）
        indicators:   {"MA"}, {"RSI"}, ..., {"MA","RSI","MACD"} のいずれか
        initial_cash: 初期資金

    Returns:
        configs:        全プリセット組み合わせのBT結果リスト（各要素にlabel/return_pct/...）
        best_combined:  最高リターンの組み合わせ（採用パラメータ含む）
        n_configs:      試験した組み合わせ数
    """
    base_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(base_df) < 60:
        return {"error": "データが不足しています（最低60日必要）"}

    configs: list[dict] = []
    for params in _iter_param_combinations(indicators):
        label_parts = []
        if "MA" in indicators:
            label_parts.append(f"MA{params['ma_short']}/{params['ma_long']}")
        if "RSI" in indicators:
            label_parts.append(f"RSI{params['rsi_ob']}/{params['rsi_os']}")
        if "MACD" in indicators:
            label_parts.append(f"MACD{params['macd_fast']}/{params['macd_slow']}/{params['macd_sig']}")
        label = " + ".join(label_parts)

        bt = _run_combo_backtest(base_df, indicators, params, initial_cash)
        configs.append({
            "label":  label,
            "params": params,
            **bt,
        })

    # 最低3トレード以上を有効とする（少なすぎる場合は除外）
    valid = [c for c in configs if c["trade_count"] >= 3]
    if valid:
        best_combined = max(valid, key=lambda c: c["return_pct"])
    else:
        best_combined = max(configs, key=lambda c: c["return_pct"]) if configs else None

    return {
        "configs":       configs,
        "best_combined": best_combined,
        "n_configs":     len(configs),
    }
