"""
時間軸適合性診断 — 銘柄が短期・長期どちらのトレンド構造を持つか診断する。
複数のMA設定でバックテストを比較し、レジームスコアと合わせて最適MA設定を提案する。
"""
import pandas as pd
import numpy as np
from src.indicators.calculator import add_ma, add_rsi, add_macd, add_bollinger
from src.strategies.composite import generate_composite_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager

# 試験するMA設定プリセット
MA_PRESETS: list[dict] = [
    {"label": "超短期 (5/25)",   "short": 5,  "long": 25},
    {"label": "短中期 (10/50)",  "short": 10, "long": 50},
    {"label": "中期 (25/75)",    "short": 25, "long": 75},
    {"label": "長期 (50/200)",   "short": 50, "long": 200},
]

# レジームスコアの各項目の最大スコア
_REGIME_MAX = 2


def _profit_factor(trades_df: pd.DataFrame) -> float:
    """プロフィットファクター = 利益合計 / 損失合計の絶対値。"""
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


def _run_ma_backtest(df: pd.DataFrame, short: int, long: int, initial_cash: float) -> dict:
    """指定MA設定のみ（RSI/MACD/BBなし）でバックテストを実行する。"""
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    d = add_ma(d, short, long)
    d = generate_composite_signal(d, ["MA"])
    risk = RiskManager(stop_loss_pct=5.0, take_profit_pct=10.0, max_position_pct=100.0)
    result = run_backtest(d, initial_cash=initial_cash, risk=risk)
    pf = _profit_factor(result["trades"])
    return {
        "return_pct":   round(result["total_return_pct"], 2),
        "win_rate_pct": round(result["win_rate_pct"], 1),
        "trade_count":  len(result["trades"]),
        "profit_factor": pf,
        "max_dd_pct":   round(result["max_drawdown_pct"], 2),
    }


# ── レジームスコア計算 ─────────────────────────────────────

def _score_ma_cross(df: pd.DataFrame, short: int = 5, long: int = 25) -> tuple[int, int, str]:
    """MAクロス頻度: 高 → 短期寄り、MA乖離持続 → 長期寄り。"""
    d = add_ma(df.copy(), short, long)
    cross = (d["MA_short"] > d["MA_long"]).astype(int).diff().abs().fillna(0)
    freq = cross.sum()                                          # クロス回数
    months = max(len(d) / 21, 1)
    cross_per_month = freq / months

    # 乖離持続: MAクロスせずに同じ側に滞在する最長期間
    same_side = (d["MA_short"] > d["MA_long"]).astype(int)
    runs = []
    current = 0
    for v in same_side:
        if v == same_side.iloc[0] if len(runs) == 0 else (v == 1 or current > 0):
            current += 1
        else:
            runs.append(current)
            current = 1
    if current > 0:
        runs.append(current)
    max_run = max(runs) if runs else 0
    avg_run = sum(runs) / len(runs) if runs else 0

    short_s = _REGIME_MAX if cross_per_month >= 3 else (1 if cross_per_month >= 1.5 else 0)
    long_s  = _REGIME_MAX if avg_run >= 40 else (1 if avg_run >= 20 else 0)
    desc = f"月平均 {cross_per_month:.1f}回クロス / 平均連続 {avg_run:.0f}日同側"
    return short_s, long_s, desc


def _score_bb_width(df: pd.DataFrame) -> tuple[int, int, str]:
    """BB幅の変動: 激しい → 短期、安定拡大 → 長期。"""
    d = add_bollinger(df.copy(), 20, 2.0)
    bb_width = d["BB_upper"] - d["BB_lower"]
    width_cv = bb_width.std() / bb_width.mean() if bb_width.mean() > 0 else 0  # 変動係数

    # 安定拡大: 後半がまえ半より広い
    half = len(bb_width) // 2
    expanding_ratio = bb_width.iloc[half:].mean() / bb_width.iloc[:half].mean() if bb_width.iloc[:half].mean() > 0 else 1

    short_s = _REGIME_MAX if width_cv >= 0.4 else (1 if width_cv >= 0.2 else 0)
    long_s  = _REGIME_MAX if expanding_ratio >= 1.3 else (1 if expanding_ratio >= 1.1 else 0)
    desc = f"BB幅変動係数 {width_cv:.2f} / 後半拡大比 {expanding_ratio:.2f}x"
    return short_s, long_s, desc


def _score_rsi_behavior(df: pd.DataFrame) -> tuple[int, int, str]:
    """RSI挙動: 30/70往復多 → 短期、50上下滞在 → 長期。"""
    d = add_rsi(df.copy(), 14)
    rsi = d["RSI"].dropna()
    months = max(len(rsi) / 21, 1)

    # 30/70タッチ回数
    touch_30 = (rsi < 30).astype(int).diff().abs().fillna(0).sum()
    touch_70 = (rsi > 70).astype(int).diff().abs().fillna(0).sum()
    cross_per_month = (touch_30 + touch_70) / months

    # 50ライン上/下滞在比率
    above_50_ratio = (rsi > 50).mean()
    extreme_ratio = max(above_50_ratio, 1 - above_50_ratio)  # 片側への偏り

    short_s = _REGIME_MAX if cross_per_month >= 2 else (1 if cross_per_month >= 1 else 0)
    long_s  = _REGIME_MAX if extreme_ratio >= 0.7 else (1 if extreme_ratio >= 0.6 else 0)
    desc = f"RSI 30/70タッチ 月{cross_per_month:.1f}回 / 50片側滞在 {extreme_ratio:.0%}"
    return short_s, long_s, desc


def _score_macd_behavior(df: pd.DataFrame) -> tuple[int, int, str]:
    """MACDクロス頻度: 多 → 短期、ゼロライン滞在長 → 長期。"""
    d = add_macd(df.copy(), 12, 26, 9)
    months = max(len(d) / 21, 1)

    cross = (d["MACD"] > d["MACD_sig"]).astype(int).diff().abs().fillna(0)
    cross_per_month = cross.sum() / months

    # ゼロライン上/下の連続滞在
    above_zero = (d["MACD"] > 0).astype(int)
    run_len = above_zero.groupby((above_zero != above_zero.shift()).cumsum()).transform("count")
    avg_run = run_len.mean() if len(run_len) > 0 else 0

    short_s = _REGIME_MAX if cross_per_month >= 3 else (1 if cross_per_month >= 1.5 else 0)
    long_s  = _REGIME_MAX if avg_run >= 40 else (1 if avg_run >= 20 else 0)
    desc = f"MACDクロス 月{cross_per_month:.1f}回 / ゼロライン平均滞在 {avg_run:.0f}日"
    return short_s, long_s, desc


def _compute_regime_score(df: pd.DataFrame) -> dict:
    """4指標のレジームスコアを計算して返す。"""
    ms, ml, md = _score_ma_cross(df)
    bs, bl, bd = _score_bb_width(df)
    rs, rl, rd = _score_rsi_behavior(df)
    cs, cl, cd = _score_macd_behavior(df)

    short_total = ms + bs + rs + cs
    long_total  = ml + bl + rl + cl
    max_possible = _REGIME_MAX * 4

    if short_total > long_total:
        dominant = "SHORT_TERM"
    elif long_total > short_total:
        dominant = "LONG_TERM"
    else:
        dominant = "NEUTRAL"

    return {
        "short_score": short_total,
        "long_score":  long_total,
        "max_score":   max_possible,
        "dominant":    dominant,
        "details": {
            "ma_cross":      {"short": ms, "long": ml, "desc": md},
            "bb_width":      {"short": bs, "long": bl, "desc": bd},
            "rsi_behavior":  {"short": rs, "long": rl, "desc": rd},
            "macd_behavior": {"short": cs, "long": cl, "desc": cd},
        },
    }


# ── メイン公開関数 ─────────────────────────────────────────

def analyze_timeframe(
    df: pd.DataFrame,
    initial_cash: float = 1_000_000,
    indicator_config: dict | None = None,
) -> dict:
    """
    銘柄の時間軸適合性を診断する。

    Returns:
        configs   : 各MAプリセットのバックテスト結果リスト
        best      : 最高リターンのMA設定
        regime    : レジームスコア詳細
        recommendation: 推奨MA設定（バックテスト＋レジームスコアの総合）
        baseline_return_pct: 現在の手動MA設定でのリターン（比較用）
        baseline_label     : 現在の設定ラベル
    """
    base_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(base_df) < 60:
        return {"error": "データが不足しています（最低60日必要）"}

    # ベースライン: 現在の手動MA設定でBT（有効性比較用）
    _ic = indicator_config or {}
    _cur_short = int(_ic.get("ma_short", 5))
    _cur_long  = int(_ic.get("ma_long",  25))
    _baseline  = _run_ma_backtest(base_df, _cur_short, _cur_long, initial_cash)
    baseline_return_pct = _baseline["return_pct"]
    baseline_label = f"現在の設定 ({_cur_short}/{_cur_long})"

    # 各MAプリセットでバックテスト
    configs = []
    for preset in MA_PRESETS:
        bt = _run_ma_backtest(base_df, preset["short"], preset["long"], initial_cash)
        configs.append({**preset, **bt})

    # バックテスト最優秀（リターン基準）
    valid = [c for c in configs if c["trade_count"] >= 3]
    best_bt = max(valid, key=lambda c: c["return_pct"]) if valid else configs[0]

    # レジームスコア
    regime = _compute_regime_score(base_df)

    # 総合推奨: レジーム判定とバックテスト結果が一致するものを優先
    if regime["dominant"] == "SHORT_TERM":
        regime_preferred = [c for c in configs if c["short"] <= 10]
    elif regime["dominant"] == "LONG_TERM":
        regime_preferred = [c for c in configs if c["short"] >= 25]
    else:
        regime_preferred = configs

    valid_pref = [c for c in regime_preferred if c["trade_count"] >= 3]
    best_combined = (
        max(valid_pref, key=lambda c: c["return_pct"])
        if valid_pref else best_bt
    )

    # レジーム判定ラベル（日本語）
    regime_label = {
        "SHORT_TERM": "短期トレンド型",
        "LONG_TERM":  "長期トレンド型",
        "NEUTRAL":    "混合型（状況依存）",
    }.get(regime["dominant"], "不明")

    return {
        "configs":             configs,
        "best_backtest":       best_bt,
        "best_combined":       best_combined,
        "regime":              regime,
        "regime_label":        regime_label,
        "baseline_return_pct": baseline_return_pct,
        "baseline_label":      baseline_label,
    }
