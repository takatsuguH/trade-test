"""
7通りディープ調査（ウォークフォワード方式・グリッドサーチ）

3指標（MA / RSI / MACD）の7つの組み合わせそれぞれについて、
WFループ × グリッドサーチで「過去どのパラメータが最良だったか」を
時系列で追跡し、各組み合わせ独立のBT結果を出力する。

サイドバー設定からは完全独立（指標設定はグリッドサーチで動的に選ぶ）。
キャッシュキーは (stock_code, combo_key, end_date, initial_cash) のみ。
"""
from typing import Callable, Optional
import pandas as pd

from src.indicators.calculator import calculate_all
from src.strategies.composite import generate_composite_signal
from src.backtest import run_backtest
from src.risk.manager import RiskManager
from src.optimization.combination_analyzer import (
    analyze_combination, SEVEN_COMBINATIONS, _build_cfg,
)
from src.db.storage import (
    compute_seven_way_params_hash,
    load_seven_way_cache_batch,
    save_seven_way_cache,
)

# 7通り側のキャッシュバージョン。グリッドサーチロジック変更時にbump
SEVEN_WAY_ALGO_VERSION = "sw_v1"


def _generate_signals_for_combo(
    raw_df: pd.DataFrame, combo_indicators: frozenset[str], params: dict,
) -> pd.DataFrame:
    """指定パラメータで指標計算と複合シグナル生成。"""
    cfg = _build_cfg(combo_indicators, params)
    d = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    d = calculate_all(d, cfg)
    active = [ind for ind in ("MA", "RSI", "MACD") if ind in combo_indicators]
    d = generate_composite_signals_safe(d, active)
    return d


def generate_composite_signals_safe(df: pd.DataFrame, active: list[str]) -> pd.DataFrame:
    """active が空の場合のフォールバック対応版。"""
    if not active:
        df = df.copy()
        df["composite_signal"] = 0
        df["vote_sum"] = 0
        df["order"] = 0
        return df
    return generate_composite_signal(df, active)


def run_seven_way_wf_backtest(
    raw_df: pd.DataFrame,
    frequency_days: int,
    initial_cash: float,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
    max_position_pct: float = 100.0,
    rebuy_dip_pct: float = 0.0,
    max_shares: int = 0,
    stock_code: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    7通り×WFでグリッドサーチBTを実行する。

    Args:
        raw_df:          OHLCVデータ（指標未計算、最低60日）
        frequency_days:  再診断間隔（5=週次, 21=月次, 1=毎日）
        initial_cash:    初期資金（キャッシュキーに含まれる）
        stop_loss_pct..max_shares: 最終BTで使うリスク管理パラメータ
        stock_code:      銘柄コード（無ければキャッシュ無効）
        progress_callback: 進捗報告 (current, total, message)

    Returns:
        {
            "results": {combo_key: {portfolio_curve, trades, total_return_pct, ...}},
            "segments_by_combo": {combo_key: [segment_dict, ...]},
            "cache_stats": {total, hits, misses},
            "frequency_days": int,
            "rebalance_count": int,
        }
    """
    n = len(raw_df)
    if n < 60:
        return {"error": "データが不足しています（最低60日必要）"}

    rebalance_points = list(range(60, n, frequency_days))
    if not rebalance_points:
        return {"error": "再診断ポイントがありません"}

    end_dates: list[str] = [str(raw_df.index[t - 1].date()) for t in rebalance_points]
    params_hash = compute_seven_way_params_hash(initial_cash) if stock_code else None

    # 全7組み合わせのキャッシュを事前ロード
    cache_by_combo: dict[str, dict[str, dict]] = {}
    if stock_code and params_hash:
        for combo_key, _, _ in SEVEN_COMBINATIONS:
            cache_by_combo[combo_key] = load_seven_way_cache_batch(
                stock_code, combo_key, end_dates,
                params_hash, SEVEN_WAY_ALGO_VERSION,
            )
    else:
        cache_by_combo = {ck: {} for ck, _, _ in SEVEN_COMBINATIONS}

    total_calls = len(rebalance_points) * len(SEVEN_COMBINATIONS)
    cache_hits = sum(len(cache_by_combo[ck]) for ck, _, _ in SEVEN_COMBINATIONS)
    cache_misses = total_calls - cache_hits

    # 各組み合わせごとに、時点ごとの採用パラメータを記録
    # segments_by_combo[combo_key] = [(t, end_t, params, label), ...]
    segments_by_combo: dict[str, list[tuple[int, int, dict, str]]] = {
        ck: [] for ck, _, _ in SEVEN_COMBINATIONS
    }

    # 進捗計算用
    total_steps = len(rebalance_points) * len(SEVEN_COMBINATIONS)
    current_step = 0

    for i, t in enumerate(rebalance_points):
        end_t = rebalance_points[i + 1] if i + 1 < len(rebalance_points) else n
        end_date_str = end_dates[i]
        past_df = raw_df.iloc[:t]

        for combo_key, indicators, _label in SEVEN_COMBINATIONS:
            current_step += 1
            cached = cache_by_combo[combo_key].get(end_date_str)
            if cached is not None:
                best = cached.get("best_combined") or {}
                segments_by_combo[combo_key].append(
                    (t, end_t, best.get("params", {}), best.get("label", ""))
                )
            else:
                try:
                    res = analyze_combination(past_df, indicators, initial_cash=initial_cash)
                except Exception:
                    res = {"error": "analyze_combination failed"}

                if "error" not in res and res.get("best_combined"):
                    best = res["best_combined"]
                    segments_by_combo[combo_key].append(
                        (t, end_t, best["params"], best["label"])
                    )
                    if stock_code and params_hash:
                        save_seven_way_cache(
                            stock_code, combo_key, end_date_str,
                            params_hash, SEVEN_WAY_ALGO_VERSION,
                            {"best_combined": best, "n_configs": res.get("n_configs", 0)},
                        )
                else:
                    # 失敗時は空セグメント（シグナル0で進む）
                    segments_by_combo[combo_key].append((t, end_t, {}, ""))

            if progress_callback and current_step % max(1, total_steps // 100) == 0:
                progress_callback(current_step, total_steps, end_date_str)

    if progress_callback:
        progress_callback(total_steps, total_steps, "完了")

    # 各組み合わせで run_backtest を実行
    results: dict[str, dict] = {}
    segments_export: dict[str, list[dict]] = {}

    for combo_key, indicators, label in SEVEN_COMBINATIONS:
        segments = segments_by_combo[combo_key]
        if not segments:
            results[combo_key] = {"error": "セグメントなし"}
            continue

        # 最初の60日は最初のセグメントのパラメータでフォールバック（params が空ならシグナル0）
        first_params = segments[0][2] if segments[0][2] else None
        if first_params:
            full_df = _generate_signals_for_combo(raw_df, indicators, first_params)
        else:
            full_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
            full_df["composite_signal"] = 0
            full_df["vote_sum"] = 0

        # 各セグメント [t, end_t) を採用パラメータで再計算して上書き
        for seg_t, seg_end_t, seg_params, _seg_label in segments:
            if not seg_params:
                continue
            seg_df = _generate_signals_for_combo(
                raw_df.iloc[:seg_end_t], indicators, seg_params,
            )
            for col in ("composite_signal", "vote_sum"):
                if col in seg_df.columns and col in full_df.columns:
                    full_df.iloc[seg_t:seg_end_t, full_df.columns.get_loc(col)] = (
                        seg_df.iloc[seg_t:seg_end_t][col].values
                    )

        # order を再計算
        if "composite_signal" in full_df.columns:
            full_df["order"] = full_df["composite_signal"].diff()

        risk = RiskManager(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_pct=max_position_pct,
            rebuy_dip_pct=rebuy_dip_pct,
        )
        bt = run_backtest(full_df, initial_cash=initial_cash, risk=risk, max_shares=max_shares)
        bt["label"] = label
        results[combo_key] = bt

        segments_export[combo_key] = [
            {
                "start_date":  str(raw_df.index[seg_t].date()),
                "end_date":    str(raw_df.index[min(seg_end_t, n) - 1].date()),
                "label":       seg_label,
                "params":      seg_params,
            }
            for seg_t, seg_end_t, seg_params, seg_label in segments
        ]

    return {
        "results":           results,
        "segments_by_combo": segments_export,
        "cache_stats": {
            "total":  total_calls,
            "hits":   cache_hits,
            "misses": cache_misses,
        },
        "frequency_days":  frequency_days,
        "rebalance_count": len(rebalance_points),
    }
