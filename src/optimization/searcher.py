"""最良テクニカル指標組み合わせ探索 — numpyベクトル化で高速化"""
import itertools
import time
import numpy as np
import pandas as pd
from src.indicators.signal_generator import _RULES, _sign

# N ≤ EXHAUSTIVE_THRESHOLD → 全探索、それ以上 → グリーディー
EXHAUSTIVE_THRESHOLD = 18


def _precompute(df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    各指標のシグナルを numpy 配列に変換して返す。
    Returns: (signals_np, daily_returns_np)
    """
    close = df["Close"].values.astype(float)
    # 日次リターン（1日シフト済みで位置合わせ）
    # position[t] で翌日 daily_ret[t+1] を得る
    daily_ret = np.empty(len(close))
    daily_ret[0] = 0.0
    daily_ret[1:] = np.diff(close) / np.where(close[:-1] != 0, close[:-1], np.nan)
    daily_ret = np.nan_to_num(daily_ret, nan=0.0)

    signals: dict[str, np.ndarray] = {}
    for col in cols:
        if col not in df.columns:
            continue
        rule = _RULES.get(col, lambda df_, c=col: _sign(df_[c]))
        try:
            s = rule(df).astype(float).fillna(0).values
            signals[col] = s
        except Exception:
            pass

    return signals, daily_ret


def _fast_return(signals_np: dict, combo: tuple, daily_ret: np.ndarray) -> float:
    """
    指定した組み合わせの多数決シグナルで簡易バックテストを実行し、
    累積リターン(%) を返す。（手数料・リスク管理なしの高速近似）
    """
    available = [c for c in combo if c in signals_np]
    if not available:
        return -999.0

    n = len(daily_ret)
    vote = np.zeros(n)
    for c in available:
        vote += signals_np[c]

    threshold = max(1, len(available) / 2)
    position = (vote >= threshold).astype(float)

    # 1日後に反映（position[t] → daily_ret[t+1]）
    strat_ret = daily_ret[1:] * position[:-1]
    strat_ret = strat_ret[~np.isnan(strat_ret)]
    if len(strat_ret) == 0:
        return -999.0
    return float((np.prod(1.0 + strat_ret) - 1.0) * 100)


def _greedy_search(
    signals_np: dict,
    valid_cols: list[str],
    daily_ret: np.ndarray,
    progress_cb=None,
) -> tuple[list[str], float]:
    """グリーディー前向き選択 + 後ろ向き除去で最良組み合わせを探索する。"""
    remaining = list(valid_cols)
    selected: list[str] = []
    best_ret = -999.0
    total_steps = len(valid_cols) * 2

    step = 0
    improved = True
    while improved and remaining:
        improved = False
        best_col, best_new_ret = None, best_ret

        # ── 前向き選択 ──
        for col in remaining:
            ret = _fast_return(signals_np, tuple(selected + [col]), daily_ret)
            if ret > best_new_ret:
                best_new_ret, best_col = ret, col

        if best_col and best_new_ret > best_ret:
            selected.append(best_col)
            remaining.remove(best_col)
            best_ret = best_new_ret
            improved = True

            # ── 後ろ向き除去（追加後、各指標を外して改善するか確認） ──
            backward = True
            while backward and len(selected) > 1:
                backward = False
                worst_col, best_without = None, best_ret
                for col in selected:
                    trial = [c for c in selected if c != col]
                    ret = _fast_return(signals_np, tuple(trial), daily_ret)
                    if ret > best_without:
                        best_without, worst_col = ret, col
                if worst_col:
                    selected.remove(worst_col)
                    remaining.append(worst_col)
                    best_ret = best_without
                    backward = True

        step += 1
        if progress_cb:
            progress_cb(min(step / total_steps, 0.99))

    return selected, best_ret


def _exhaustive_search(
    signals_np: dict,
    valid_cols: list[str],
    daily_ret: np.ndarray,
    progress_cb=None,
) -> tuple[list[str], float]:
    """全組み合わせを網羅探索する（N ≤ EXHAUSTIVE_THRESHOLD 専用）。"""
    n = len(valid_cols)
    total = 2 ** n - 1
    best_cols: list[str] = []
    best_ret = -999.0
    count = 0

    for size in range(1, n + 1):
        for combo in itertools.combinations(valid_cols, size):
            ret = _fast_return(signals_np, combo, daily_ret)
            if ret > best_ret:
                best_ret, best_cols = ret, list(combo)
            count += 1
            if progress_cb and count % max(1, total // 200) == 0:
                progress_cb(count / total)

    return best_cols, best_ret


def find_best_combination(
    df: pd.DataFrame,
    candidate_cols: list[str],
    progress_cb=None,
) -> tuple[list[str], float, str]:
    """
    最良の指標組み合わせを探索する。

    Returns:
        best_cols   : 選択された指標の列名リスト
        best_return : 推定リターン(%)
        method      : 使用した探索手法の説明文
    """
    t0 = time.perf_counter()
    signals_np, daily_ret = _precompute(df, candidate_cols)
    valid_cols = [c for c in candidate_cols if c in signals_np]
    n = len(valid_cols)

    if n == 0:
        return [], -999.0, "候補なし"

    if n <= EXHAUSTIVE_THRESHOLD:
        best_cols, best_ret = _exhaustive_search(signals_np, valid_cols, daily_ret, progress_cb)
        method = f"全探索（{2**n - 1:,}通り）"
    else:
        best_cols, best_ret = _greedy_search(signals_np, valid_cols, daily_ret, progress_cb)
        method = f"グリーディー探索（{n}候補）"

    elapsed = time.perf_counter() - t0
    method += f" / {elapsed:.1f}秒"
    return best_cols, best_ret, method
