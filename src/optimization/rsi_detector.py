"""
RSI閾値適合性診断 — 銘柄が早期反転型か遅延反転型かを診断する。
複数のRSI閾値設定でバックテストを比較し、反応速度スコアと合わせて最適RSI閾値を提案する。
MACDパラメータ診断と同様に、MA+RSI+MACD+BBの複合シグナルでバックテストする。
"""
import pandas as pd
import numpy as np
from src.indicators.calculator import calculate_all, add_rsi
from src.strategies.composite import generate_composite_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager

# 試験するRSI閾値プリセット
RSI_PRESETS: list[dict] = [
    {"label": "標準 (70/30)",   "ob": 70, "os": 30},
    {"label": "緩感応 (65/32)", "ob": 65, "os": 32},
    {"label": "感応 (60/35)",   "ob": 60, "os": 35},
    {"label": "超感応 (55/40)", "ob": 55, "os": 40},
]

# indicator_config が渡されない場合のデフォルト設定
_DEFAULT_CONFIG: dict = {
    "use_ma":   True,  "ma_short":   5,   "ma_long":    25,
    "use_rsi":  True,  "rsi_period": 14,  "rsi_ob":     70,  "rsi_os":  30,
    "use_macd": True,  "macd_fast":  12,  "macd_slow":  26,  "macd_sig": 9,
    "use_bb":   True,  "bb_period":  20,  "bb_std":     2.0,
}


def _profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    closed = trades_df[trades_df["profit"].notna()]
    if closed.empty:
        return 0.0
    wins   = closed[closed["profit"] > 0]["profit"].sum()
    losses = closed[closed["profit"] < 0]["profit"].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / abs(losses), 2)


def _build_active(cfg: dict) -> list[str]:
    active = []
    if cfg.get("use_ma"):   active.append("MA")
    if cfg.get("use_rsi"):  active.append("RSI")
    if cfg.get("use_macd"): active.append("MACD")
    if cfg.get("use_bb"):   active.append("BB")
    return active or ["RSI"]


def _run_rsi_backtest(
    df: pd.DataFrame,
    ob: int,
    os: int,
    initial_cash: float,
    indicator_config: dict | None = None,
) -> dict:
    """指定RSI閾値での複合シグナルバックテストを実行する。
    MA/MACD/BBなど他指標はindicator_configの設定を維持し、RSI閾値のみ差し替える。
    """
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    base_cfg = {**_DEFAULT_CONFIG, **(indicator_config or {})}
    cfg = {**base_cfg, "use_rsi": True, "rsi_ob": ob, "rsi_os": os}
    d = calculate_all(d, cfg)
    active = _build_active(cfg)
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


def _compute_reversal_score(df: pd.DataFrame) -> dict:
    """
    反応速度スコアを計算する。

    zone_ratio = (RSIが60/40ゾーン滞在率) / (RSIが70/30ゾーン滞在率)
      - 高い(>3): 60ゾーンで頻繁に反転、70まで届きにくい → 早期反転型(EARLY)
      - 低い(<2): 60を超えて70まで伸びることが多い → 遅延反転型(LATE)
    """
    d = add_rsi(df.copy(), 14)
    rsi = d["RSI"].dropna()
    months = max(len(rsi) / 21, 1)

    # 各ゾーン滞在比率
    in_zone_60 = ((rsi > 60) | (rsi < 40)).mean()
    in_zone_70 = ((rsi > 70) | (rsi < 30)).mean()
    zone_ratio = in_zone_60 / max(in_zone_70, 0.01)

    # 60ゾーンから70ゾーンへの進展率（60超えた後、さらに70超えた割合）
    above_60 = rsi > 60
    above_70 = rsi > 70
    continuation = above_70.sum() / max(above_60.sum(), 1)

    # RSI 30/70タッチ回数（月当たり）
    touch_70 = ((rsi > 70).astype(int).diff().abs().fillna(0).sum() +
                (rsi < 30).astype(int).diff().abs().fillna(0).sum())
    touch_60 = ((rsi > 60).astype(int).diff().abs().fillna(0).sum() +
                (rsi < 40).astype(int).diff().abs().fillna(0).sum())
    touch_70_per_month = touch_70 / months
    touch_60_per_month = touch_60 / months

    # 判定
    if zone_ratio > 3 and continuation < 0.4:
        dominant = "EARLY_REVERSAL"
    elif zone_ratio < 2 or continuation > 0.6:
        dominant = "LATE_REVERSAL"
    else:
        dominant = "MIXED"

    return {
        "dominant": dominant,
        "details": {
            "zone_ratio": {
                "value": round(zone_ratio, 2),
                "desc":  f"60/40ゾーン比率 {zone_ratio:.1f}（高い→早期反転型）",
            },
            "continuation": {
                "value": round(continuation, 2),
                "desc":  f"70/30への進展率 {continuation:.0%}（低い→早期反転型）",
            },
            "touch_frequency": {
                "value": round(touch_70_per_month, 1),
                "desc":  f"月平均タッチ {touch_70_per_month:.1f}回（70/30）/ {touch_60_per_month:.1f}回（60/40）",
            },
        },
    }


def analyze_rsi(
    df: pd.DataFrame,
    initial_cash: float = 1_000_000,
    indicator_config: dict | None = None,
) -> dict:
    """
    銘柄のRSI閾値適合性を診断する。
    RSI閾値4種を、ユーザーの現在の指標設定（MA/MACD/BB）と組み合わせた
    複合シグナルでバックテストし、最も効果的な閾値を推奨する。

    Args:
        indicator_config: 現在の銘柄の指標設定（_build_indicator_config()の結果）。
                          rsi_ob/rsi_os には現在のアクティブ閾値（rsi_diag_ob/os）を渡すこと。

    Returns:
        configs            : 各プリセットのバックテスト結果リスト
        best_backtest      : バックテスト最優秀プリセット
        best_combined      : バックテスト＋反応速度スコアの総合推奨プリセット
        reversal           : 反応速度スコア詳細
        reversal_label     : 銘柄タイプ（日本語ラベル）
        baseline_return_pct: 現在のアクティブRSI閾値での複合BTリターン（比較用）
        baseline_label     : 現在の設定ラベル
    """
    base_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(base_df) < 30:
        return {"error": "データが不足しています（最低30日必要）"}

    base_cfg = {**_DEFAULT_CONFIG, **(indicator_config or {})}

    # ベースライン: 現在のアクティブRSI閾値（rsi_ob/os）で複合BT（有効性比較用）
    _cur_ob = int(base_cfg.get("rsi_ob", 70))
    _cur_os = int(base_cfg.get("rsi_os", 30))
    _baseline = _run_rsi_backtest(base_df, _cur_ob, _cur_os, initial_cash, base_cfg)
    baseline_return_pct = _baseline["return_pct"]
    baseline_label = f"現在の設定 ({_cur_ob}/{_cur_os})"

    # 各プリセットで複合バックテスト（RSI閾値のみ差し替え）
    configs = []
    for preset in RSI_PRESETS:
        bt = _run_rsi_backtest(base_df, preset["ob"], preset["os"], initial_cash, base_cfg)
        configs.append({**preset, **bt})

    # バックテスト最優秀（リターン基準、最低3トレード以上）
    valid = [c for c in configs if c["trade_count"] >= 3]
    best_bt = max(valid, key=lambda c: c["return_pct"]) if valid else configs[0]

    # 反応速度スコア
    reversal = _compute_reversal_score(base_df)

    # 総合推奨: 反応速度判定とバックテスト結果の総合
    if reversal["dominant"] == "EARLY_REVERSAL":
        preferred = [c for c in configs if c["ob"] <= 65]
    elif reversal["dominant"] == "LATE_REVERSAL":
        preferred = [c for c in configs if c["ob"] >= 65]
    else:
        preferred = configs

    valid_pref = [c for c in preferred if c["trade_count"] >= 3]
    best_combined = (
        max(valid_pref, key=lambda c: c["return_pct"])
        if valid_pref else best_bt
    )

    reversal_label = {
        "EARLY_REVERSAL": "早期反転型（レンジ寄り）",
        "LATE_REVERSAL":  "遅延反転型（トレンド寄り）",
        "MIXED":          "中間型（状況依存）",
    }.get(reversal["dominant"], "不明")

    return {
        "configs":             configs,
        "best_backtest":       best_bt,
        "best_combined":       best_combined,
        "reversal":            reversal,
        "reversal_label":      reversal_label,
        "baseline_return_pct": baseline_return_pct,
        "baseline_label":      baseline_label,
        "diag_version":        2,   # 複合シグナルベース（v1: RSI単独）
    }
