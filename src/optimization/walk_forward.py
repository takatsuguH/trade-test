"""
ウォークフォワード方式バックテスト — 各時点で過去データから診断 → そのパラメータで売買判断。
ルックアヘッドバイアスを排除した、より実運用に近いバックテストを提供する。

通常診断: 期間全体のデータでパラメータを最適化 → 全期間に適用（バイアスあり）
ウォークフォワード: t時点では data[:t] のみで診断 → [t, t+freq) の判断に適用（バイアスなし）

キャッシュ:
  end_date 単位で診断結果を DB (wf_diagnosis_cache) に保存する。
  同じ過去データ × 同じパラメータなら結果は決定的なので再計算不要。
  頻度違い (日次/週次/月次) も end_date 集合で表現でき、共有可能。
"""
from typing import Callable, Optional
import pandas as pd
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended, calculate_short_signals
from src.indicators.signal_generator import generate_ext_signals
from src.strategies.composite import generate_composite_signal, merge_all_signals
from src.strategies.context_strategy import generate_context_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager
from src.optimization.timeframe_detector import analyze_timeframe
from src.optimization.rsi_detector import analyze_rsi
from src.optimization.macd_detector import analyze_macd
from src.db.storage import (
    compute_wf_params_hash, load_wf_cache_batch, save_wf_cache,
)

# キャッシュバージョン。analyze_timeframe/rsi/macd のロジックを変更したら bump する。
WF_ALGO_VERSION = "v1"


def _is_diag_effective(diag_result: dict) -> bool:
    """診断結果が現在設定より優秀か（best_combined.return_pct > baseline_return_pct）。"""
    if "error" in diag_result:
        return False
    baseline = diag_result.get("baseline_return_pct")
    if baseline is None:
        return False
    return diag_result.get("best_combined", {}).get("return_pct", 0) > baseline


def _build_ic(s: dict) -> dict:
    return {
        "use_ma":   s["use_ma"],   "ma_short":   s["ma_short"],  "ma_long":   s["ma_long"],
        "use_rsi":  s["use_rsi"],  "rsi_period": s["rsi_period"], "rsi_ob":   s["rsi_ob"],   "rsi_os": s["rsi_os"],
        "use_macd": s["use_macd"], "macd_fast":  s["macd_fast"],  "macd_slow": s["macd_slow"], "macd_sig": s["macd_sig"],
        "use_bb":   s["use_bb"],   "bb_period":  s["bb_period"],  "bb_std":   s["bb_std"],
    }


def _snap_resolved_settings(s: dict) -> dict:
    """snap_* があれば元の手動値に戻した設定を返す。
    診断トグルONで現在値が診断値に置き換わっている場合、ユーザー本来の手動値が snap に退避されている。
    WFのベースラインには手動値を使うべきなので、snap があればそちらを優先する。
    """
    r = dict(s)
    if s.get("snap_tf_ma_short") is not None:
        r["ma_short"] = int(s["snap_tf_ma_short"])
        r["ma_long"]  = int(s["snap_tf_ma_long"])
    if s.get("snap_rsi_ob") is not None:
        r["rsi_ob"] = int(s["snap_rsi_ob"])
        r["rsi_os"] = int(s["snap_rsi_os"])
    if s.get("snap_macd_fast") is not None:
        r["macd_fast"] = int(s["snap_macd_fast"])
        r["macd_slow"] = int(s["snap_macd_slow"])
        r["macd_sig"]  = int(s["snap_macd_sig"])
    return r


def _build_active(s: dict) -> list[str]:
    a = []
    if s.get("use_ma"):   a.append("MA")
    if s.get("use_rsi"):  a.append("RSI")
    if s.get("use_macd"): a.append("MACD")
    if s.get("use_bb"):   a.append("BB")
    return a


def _generate_signals(
    raw_df: pd.DataFrame, settings: dict, extra_cols: list[str], ext_params: dict,
) -> pd.DataFrame:
    """指定設定で指標計算とシグナル生成（拡張指標/context_strategy対応）。"""
    ic = _build_ic(settings)
    active = _build_active(settings)
    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = calculate_all(df, ic)
    df = calculate_short_signals(df)

    if extra_cols:
        df_ext = calculate_extended(df, ext_params)
        for _c in df_ext.columns:
            if _c not in df.columns:
                df[_c] = df_ext[_c]
        df, _ext_sig_cols = generate_ext_signals(df, extra_cols, params=settings)
        if settings.get("use_context_strategy", False):
            df = generate_context_signal(
                df, active, _ext_sig_cols,
                score_threshold=settings.get("context_score_threshold", 5),
                rsi_ob=settings.get("rsi_ob", 70),
                rsi_os=settings.get("rsi_os", 30),
            )
        else:
            df = merge_all_signals(df, active, _ext_sig_cols)
    else:
        if settings.get("use_context_strategy", False):
            df = generate_context_signal(
                df, active,
                score_threshold=settings.get("context_score_threshold", 5),
                rsi_ob=settings.get("rsi_ob", 70),
                rsi_os=settings.get("rsi_os", 30),
            )
        else:
            df = generate_composite_signal(df, active)
    return df


def run_walk_forward_backtest(
    raw_df: pd.DataFrame,
    base_settings: dict,
    frequency_days: int,
    initial_cash: float,
    extra_cols: list[str],
    ext_params: dict,
    fund_score: int = 0,
    fund_count: int = 0,
    fund_integrate: bool = False,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
    max_position_pct: float = 100.0,
    rebuy_dip_pct: float = 0.0,
    max_shares: int = 0,
    diag_types: tuple[str, ...] = ("timeframe", "rsi", "macd"),
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stock_code: Optional[str] = None,
) -> dict:
    """
    ウォークフォワード方式でバックテストを実行する。

    各 frequency_days 日ごとに「過去データのみ」で3診断を実行し、
    有効ならその推奨値で、無効なら base_settings でその区間のシグナルを生成。
    最初の60日は診断不能のため base_settings を使用。

    Args:
        raw_df:          OHLCV データ（指標未計算）
        base_settings:   ユーザーの現在設定。WF対象外の項目（BBや拡張指標、context等）はそのまま使用
        frequency_days:  再診断間隔（取引日単位、例: 5=週次、21=月次、1=毎日）
        diag_types:      ウォークフォワード対象の診断 ("timeframe", "rsi", "macd")
        progress_callback: 進捗報告 (current_step, total_steps, date_str)

    Returns:
        run_backtest の結果に以下を追加:
            'segments':         各セグメントの設定履歴（list of dict）
            'frequency_days':   再診断間隔
            'rebalance_count':  再診断回数
    """
    n = len(raw_df)
    if n < 60:
        return {"error": "データが不足しています（最低60日必要）"}

    # snap_* がある場合は手動値に戻したベースライン設定を使用（診断値ではなく）
    baseline_settings = _snap_resolved_settings(base_settings)
    base_ic = _build_ic(baseline_settings)
    rebalance_points = list(range(60, n, frequency_days))
    if not rebalance_points:
        return {"error": "再診断ポイントがありません"}

    # キャッシュキー生成: stock_code が無ければキャッシュ無効
    params_hash = compute_wf_params_hash(base_ic, initial_cash) if stock_code else None

    # 各 t に対応する end_date（YYYY-MM-DD）を事前に算出
    end_dates: list[str] = [str(raw_df.index[t - 1].date()) for t in rebalance_points]

    # 有効な診断種別を確定
    active_diags = [
        d for d in diag_types
        if (
            (d == "timeframe" and base_settings.get("show_timeframe_diagnosis"))
            or (d == "rsi" and base_settings.get("show_rsi_diagnosis"))
            or (d == "macd" and base_settings.get("show_macd_diagnosis"))
        )
    ]

    # DBから既存キャッシュをまとめて取得 (diag_type ごとに end_date -> result)
    cache_by_diag: dict[str, dict[str, dict]] = {d: {} for d in active_diags}
    if stock_code and params_hash:
        for d in active_diags:
            cache_by_diag[d] = load_wf_cache_batch(
                stock_code, d, end_dates, params_hash, WF_ALGO_VERSION,
            )

    # キャッシュ統計（UI表示用）
    total_diag_calls = len(rebalance_points) * len(active_diags)
    cache_hits = sum(len(cache_by_diag[d]) for d in active_diags)
    cache_misses = total_diag_calls - cache_hits

    segments: list[tuple[int, int, dict, list[str]]] = []
    total_steps = len(rebalance_points)

    def _apply_diag(seg_settings: dict, applied: list[str], diag: str, res: dict) -> None:
        """診断結果が有効なら seg_settings に反映、applied にラベル追加。"""
        if not _is_diag_effective(res):
            return
        b = res["best_combined"]
        if diag == "timeframe":
            seg_settings["ma_short"] = b["short"]
            seg_settings["ma_long"]  = b["long"]
            applied.append("MA")
        elif diag == "rsi":
            seg_settings["rsi_ob"] = b["ob"]
            seg_settings["rsi_os"] = b["os"]
            applied.append("RSI")
        elif diag == "macd":
            seg_settings["macd_fast"] = b["fast"]
            seg_settings["macd_slow"] = b["slow"]
            seg_settings["macd_sig"]  = b["sig"]
            applied.append("MACD")

    _analyze_fn = {
        "timeframe": analyze_timeframe,
        "rsi":       analyze_rsi,
        "macd":      analyze_macd,
    }

    for i, t in enumerate(rebalance_points):
        end_t = rebalance_points[i + 1] if i + 1 < total_steps else n
        end_date_str = end_dates[i]
        past_df = raw_df.iloc[:t]
        seg_settings = dict(baseline_settings)
        applied: list[str] = []

        for diag in active_diags:
            cached = cache_by_diag[diag].get(end_date_str)
            if cached is not None:
                # is_effective は load 時に注入済み。_apply_diag は内部で再判定するため
                # baseline_return_pct と best_combined が入っていればそのまま使える。
                _apply_diag(seg_settings, applied, diag, cached)
                continue
            try:
                res = _analyze_fn[diag](past_df, initial_cash=initial_cash, indicator_config=base_ic)
            except Exception:
                continue
            _apply_diag(seg_settings, applied, diag, res)
            if stock_code and params_hash and "error" not in res:
                save_wf_cache(
                    stock_code, diag, end_date_str,
                    params_hash, WF_ALGO_VERSION,
                    res, _is_diag_effective(res),
                )

        segments.append((t, end_t, seg_settings, applied))

        if progress_callback:
            progress_callback(i + 1, total_steps, str(raw_df.index[t].date()))

    # 最初の60日は baseline_settings（ユーザー手動値）でシグナル生成
    full_df = _generate_signals(raw_df, baseline_settings, extra_cols, ext_params)

    # 各セグメント [t, end_t) を seg_settings で再計算 → composite_signal/vote_sum/order を上書き
    for t, end_t, seg_settings, _applied in segments:
        seg_df = _generate_signals(raw_df.iloc[:end_t], seg_settings, extra_cols, ext_params)
        for col in ("composite_signal", "vote_sum"):
            if col in seg_df.columns and col in full_df.columns:
                full_df.iloc[t:end_t, full_df.columns.get_loc(col)] = seg_df.iloc[t:end_t][col].values

    # order を再計算（composite_signal の差分）
    if "composite_signal" in full_df.columns:
        full_df["order"] = full_df["composite_signal"].diff()

    # ファンダメンタル統合（通常BTと同じロジック）
    if fund_integrate and fund_count > 0 and "vote_sum" in full_df.columns:
        if baseline_settings.get("use_context_strategy", False):
            threshold = baseline_settings.get("context_score_threshold", 5)
        else:
            tech_col_count = len(_build_active(baseline_settings)) + len(extra_cols)
            threshold = max(1, tech_col_count / 2)
        full_df["vote_sum"] = full_df["vote_sum"] + fund_score
        full_df["composite_signal"] = 0
        full_df.loc[full_df["vote_sum"] >= threshold, "composite_signal"] = 1
        full_df.loc[full_df["vote_sum"] <= -threshold, "composite_signal"] = -1
        full_df["order"] = full_df["composite_signal"].diff()

    risk = RiskManager(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_position_pct=max_position_pct,
        rebuy_dip_pct=rebuy_dip_pct,
    )
    result = run_backtest(full_df, initial_cash=initial_cash, risk=risk, max_shares=max_shares)

    result["segments"] = [
        {
            "start_date":     str(raw_df.index[t].date()),
            "end_date":       str(raw_df.index[min(end_t, n) - 1].date()),
            "applied_diags":  applied,
            "ma_short":       seg_s["ma_short"],
            "ma_long":        seg_s["ma_long"],
            "rsi_ob":         seg_s["rsi_ob"],
            "rsi_os":         seg_s["rsi_os"],
            "macd_fast":      seg_s["macd_fast"],
            "macd_slow":      seg_s["macd_slow"],
            "macd_sig":       seg_s["macd_sig"],
        }
        for t, end_t, seg_s, applied in segments
    ]
    result["frequency_days"]  = frequency_days
    result["rebalance_count"] = len(segments)
    result["cache_stats"] = {
        "total":  total_diag_calls,
        "hits":   cache_hits,
        "misses": cache_misses,
        "active_diags": active_diags,
    }
    return result
