"""
MACDパラメータ適合性診断 — 銘柄が短期・長期どちらのMACD設定に向くかを診断する。
MACDパラメータ4種を、ユーザーの現在の指標設定（MA/RSI/BB）と組み合わせた
複合シグナルでバックテストし、最も効果的な設定を推奨する。
"""
import pandas as pd
from src.indicators.calculator import add_macd, calculate_all
from src.strategies.composite import generate_composite_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager

# 試験するMACDパラメータプリセット
MACD_PRESETS: list[dict] = [
    {"label": "短期 (5/20/5)",   "fast": 5,  "slow": 20, "sig": 5},
    {"label": "標準 (12/26/9)",  "fast": 12, "slow": 26, "sig": 9},
    {"label": "中期 (20/50/9)",  "fast": 20, "slow": 50, "sig": 9},
    {"label": "長期 (25/75/9)",  "fast": 25, "slow": 75, "sig": 9},
]

# indicator_config が渡されない場合のデフォルト設定
_DEFAULT_CONFIG: dict = {
    "use_ma":   True,  "ma_short":   25,  "ma_long":    75,
    "use_rsi":  True,  "rsi_period": 14,  "rsi_ob":     60,  "rsi_os":  35,
    "use_macd": True,  "macd_fast":  12,  "macd_slow":  26,  "macd_sig": 9,
    "use_bb":   True,  "bb_period":  20,  "bb_std":     2.0,
}


def _calc_atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """14日ATRを株価の%で返す（ボラティリティ推定用）。"""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return round(float(atr / df["Close"].iloc[-1] * 100), 2)


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
    return active or ["MACD"]


def _run_composite_backtest(
    df: pd.DataFrame,
    indicator_config: dict,
    initial_cash: float = 1_000_000,
    stop_loss_pct: float = 5.0,
) -> dict:
    """指定indicator_configで複合シグナルバックテストを実行する。"""
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    d = calculate_all(d, indicator_config)
    active = _build_active(indicator_config)
    d = generate_composite_signal(d, active)
    take_profit_pct = round(stop_loss_pct * 2.0, 1)
    risk = RiskManager(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, max_position_pct=100.0)
    result = run_backtest(d, initial_cash=initial_cash, risk=risk)
    pf = _profit_factor(result["trades"])
    return {
        "return_pct":    round(result["total_return_pct"], 2),
        "win_rate_pct":  round(result["win_rate_pct"], 1),
        "trade_count":   len(result["trades"]),
        "profit_factor": pf,
        "max_dd_pct":    round(result["max_drawdown_pct"], 2),
    }


def _score_cross_tendency(df: pd.DataFrame) -> dict:
    """MACDクロス頻度・ゼロライン滞在長 → 短期向き / 長期向き判定。"""
    d = add_macd(df.copy(), 12, 26, 9)
    months = max(len(d) / 21, 1)

    # シグナル線クロス頻度（月当たり）
    cross = (d["MACD"] > d["MACD_sig"]).astype(int).diff().abs().fillna(0)
    cross_per_month = cross.sum() / months

    # ゼロライン上下の平均連続滞在日数
    above_zero = (d["MACD"] > 0).astype(int)
    groups = (above_zero != above_zero.shift()).cumsum()
    avg_run = above_zero.groupby(groups).transform("count").mean()

    if cross_per_month >= 3:
        dominant = "SHORT_TERM"
    elif avg_run >= 40:
        dominant = "LONG_TERM"
    else:
        dominant = "NEUTRAL"

    return {
        "dominant":         dominant,
        "cross_per_month":  round(cross_per_month, 1),
        "avg_run":          round(avg_run, 1),
        "details": {
            "cross_freq": {
                "value": round(cross_per_month, 1),
                "desc":  f"月平均 {cross_per_month:.1f}回クロス（多い→短期向き）",
            },
            "zero_run": {
                "value": round(avg_run, 1),
                "desc":  f"ゼロライン平均滞在 {avg_run:.0f}日（長い→長期向き）",
            },
        },
    }


def analyze_macd(
    df: pd.DataFrame,
    initial_cash: float = 1_000_000,
    indicator_config: dict | None = None,
) -> dict:
    """
    銘柄のMACDパラメータ適合性を診断する。
    MACDパラメータ4種を、ユーザーの現在の指標設定（MA/RSI/BB）と組み合わせた
    複合シグナルでバックテストし、最も効果的な設定を推奨する。

    Args:
        indicator_config: 現在の銘柄の指標設定（_build_indicator_config()の結果）。
                          Noneの場合はデフォルト設定（全指標ON）を使用。

    Returns:
        configs       : 各プリセットのバックテスト結果リスト
        best_backtest : バックテスト最優秀プリセット
        best_combined : バックテスト＋クロス傾向スコアの総合推奨プリセット
        cross         : クロス傾向スコア詳細
        cross_label   : 銘柄タイプ（日本語ラベル）
        atr_pct       : 14日ATR（%）
        stop_loss_pct : 診断に使用したストップロス（%）
    """
    base_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(base_df) < 60:
        return {"error": "データが不足しています（最低60日必要）"}

    # ATRベースでストップロスを動的設定（高ボラ銘柄でSL頻発を防ぐ）
    atr_pct = _calc_atr_pct(base_df)
    stop_loss_pct = min(max(5.0, atr_pct * 2.0), 15.0)

    # ベース設定（ユーザー設定があれば優先、なければデフォルト）
    base_cfg = {**_DEFAULT_CONFIG, **(indicator_config or {})}

    # ベースライン: 現在のMACD設定（上書きなし）で複合BT（有効性比較用）
    _baseline = _run_composite_backtest(base_df, base_cfg, initial_cash, stop_loss_pct)
    baseline_return_pct = _baseline["return_pct"]
    baseline_label = (
        f"現在の設定 ({base_cfg.get('macd_fast', 12)}"
        f"/{base_cfg.get('macd_slow', 26)}/{base_cfg.get('macd_sig', 9)})"
    )

    # 各MACDプリセットで複合バックテスト（MACD以外の設定は現在の銘柄設定を維持）
    configs = []
    for preset in MACD_PRESETS:
        cfg = {
            **base_cfg,
            "use_macd":  True,
            "macd_fast": preset["fast"],
            "macd_slow": preset["slow"],
            "macd_sig":  preset["sig"],
        }
        bt = _run_composite_backtest(base_df, cfg, initial_cash, stop_loss_pct)
        configs.append({**preset, **bt})

    # バックテスト最優秀（リターン基準、最低3トレード以上）
    valid = [c for c in configs if c["trade_count"] >= 3]
    best_bt = max(valid, key=lambda c: c["return_pct"]) if valid else configs[0]

    # クロス傾向スコア
    cross = _score_cross_tendency(base_df)

    # 総合推奨: クロス傾向とバックテスト結果の総合
    if cross["dominant"] == "SHORT_TERM":
        preferred = [c for c in configs if c["fast"] <= 8]
    elif cross["dominant"] == "LONG_TERM":
        preferred = [c for c in configs if c["fast"] >= 20]
    else:
        preferred = configs

    valid_pref = [c for c in preferred if c["trade_count"] >= 3]
    best_combined = (
        max(valid_pref, key=lambda c: c["return_pct"])
        if valid_pref else best_bt
    )

    cross_label = {
        "SHORT_TERM": "短期クロス型（値動き速め）",
        "LONG_TERM":  "長期トレンド型（値動き緩やか）",
        "NEUTRAL":    "中間型（状況依存）",
    }.get(cross["dominant"], "不明")

    return {
        "configs":             configs,
        "best_backtest":       best_bt,
        "best_combined":       best_combined,
        "cross":               cross,
        "cross_label":         cross_label,
        "atr_pct":             atr_pct,
        "stop_loss_pct":       stop_loss_pct,
        "baseline_return_pct": baseline_return_pct,
        "baseline_label":      baseline_label,
    }
