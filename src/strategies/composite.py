"""複合シグナル生成 — 複数指標の多数決で最終売買シグナルを決定する"""
import pandas as pd

_SIGNAL_COLS = {
    "MA": "MA_signal",
    "RSI": "RSI_signal",
    "MACD": "MACD_vote",
    "BB": "BB_signal",
}


def generate_composite_signal(df: pd.DataFrame, active: list[str]) -> pd.DataFrame:
    """
    active: 有効化する指標名リスト（例: ["MA", "RSI", "MACD"]）
    各指標の票を合計し、過半数以上で買い/売りシグナルを立てる。
    """
    df = df.copy()
    cols = [_SIGNAL_COLS[ind] for ind in active if _SIGNAL_COLS.get(ind) in df.columns]

    if not cols:
        df["composite_signal"] = 0
        df["vote_sum"] = 0
        df["order"] = 0
        return df

    df["vote_sum"] = df[cols].sum(axis=1)
    threshold = max(1, len(cols) / 2)

    df["composite_signal"] = 0
    df.loc[df["vote_sum"] >= threshold, "composite_signal"] = 1
    df.loc[df["vote_sum"] <= -threshold, "composite_signal"] = -1

    df["order"] = df["composite_signal"].diff()
    return df


def merge_all_signals(
    df: pd.DataFrame,
    active_main: list[str],
    extra_sig_cols: list[str],
) -> pd.DataFrame:
    """
    メイン指標（MA/RSI/MACD/BB）と追加指標のシグナルを統合する。
    active_main : ["MA","RSI",...] — 有効なメイン指標名
    extra_sig_cols : ["_sig_STOCH_K",...] — generate_ext_signals で生成された列名
    """
    df = df.copy()
    main_cols = [_SIGNAL_COLS[k] for k in active_main if _SIGNAL_COLS.get(k) in df.columns]
    all_cols = [c for c in main_cols + extra_sig_cols if c in df.columns]

    if not all_cols:
        df["composite_signal"] = 0
        df["vote_sum"] = 0
        df["order"] = 0
        return df

    df["vote_sum"] = df[all_cols].sum(axis=1)
    threshold = max(1, len(all_cols) / 2)

    df["composite_signal"] = 0
    df.loc[df["vote_sum"] >= threshold, "composite_signal"] = 1
    df.loc[df["vote_sum"] <= -threshold, "composite_signal"] = -1

    df["order"] = df["composite_signal"].diff()
    return df
