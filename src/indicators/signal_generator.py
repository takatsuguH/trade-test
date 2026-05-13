"""拡張テクニカル指標 → 売買シグナル変換（-1/0/+1）"""
import pandas as pd
import numpy as np


def _sign(s: pd.Series) -> pd.Series:
    """正→+1、負→-1、ゼロ→0 に変換する。"""
    return s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0) if pd.notna(x) else 0)


def _price_vs(df: pd.DataFrame, col: str) -> pd.Series:
    """株価がcolより上→+1、下→-1。"""
    return (df["Close"] > df[col]).astype(int) * 2 - 1


# ── 指標別シグナル生成ルール ─────────────────────────────
_RULES: dict[str, callable] = {
    # ── モメンタム系（閾値） ──
    "STOCH_K":    lambda df: df["STOCH_K"].apply(lambda x: 1 if x < 20 else (-1 if x > 80 else 0)),
    "STOCH_D":    lambda df: df["STOCH_D"].apply(lambda x: 1 if x < 20 else (-1 if x > 80 else 0)),
    "CCI":        lambda df: df["CCI"].apply(lambda x: 1 if x < -100 else (-1 if x > 100 else 0)),
    "WILLIAMS_R": lambda df: df["WILLIAMS_R"].apply(lambda x: 1 if x < -80 else (-1 if x > -20 else 0)),
    "MFI":        lambda df: df["MFI"].apply(lambda x: 1 if x < 20 else (-1 if x > 80 else 0)),
    "UO":         lambda df: df["UO"].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0)),
    "RSI":        lambda df: df["RSI"].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0)),

    # ── ゼロクロス系 ──
    "ROC":          lambda df: _sign(df["ROC"]),
    "MOM":          lambda df: _sign(df["MOM"]),
    "CMF":          lambda df: _sign(df["CMF"]),
    "CMO":          lambda df: _sign(df["CMO"]),
    "TRIX":         lambda df: _sign(df["TRIX"]),
    "DPO":          lambda df: _sign(df["DPO"]),
    "FORCE_INDEX":  lambda df: _sign(df["FORCE_INDEX"]),
    "EOM":          lambda df: _sign(df["EOM"]),
    "AROON_OSC":    lambda df: _sign(df["AROON_OSC"]),
    "COPPOCK":      lambda df: _sign(df["COPPOCK"]),
    "MACD":         lambda df: _sign(df["MACD"] - df["MACD_sig"]) if "MACD_sig" in df.columns else _sign(df["MACD"]),

    # ── OBV：変化方向 ──
    "OBV": lambda df: _sign(df["OBV"].diff()),

    # ── 価格 vs 指標線 ──
    "DEMA":      lambda df: _price_vs(df, "DEMA"),
    "TEMA":      lambda df: _price_vs(df, "TEMA"),
    "HMA":       lambda df: _price_vs(df, "HMA"),
    "VWAP":      lambda df: _price_vs(df, "VWAP"),
    "PSAR":      lambda df: _price_vs(df, "PSAR"),
    "KC_MID":    lambda df: _price_vs(df, "KC_MID"),
    "KC_UPPER":  lambda df: _price_vs(df, "KC_MID") if "KC_MID" in df.columns else pd.Series(0, index=df.index),
    "KC_LOWER":  lambda df: _price_vs(df, "KC_MID") if "KC_MID" in df.columns else pd.Series(0, index=df.index),
    "DC_MID":    lambda df: _price_vs(df, "DC_MID"),
    "DC_UPPER":  lambda df: _price_vs(df, "DC_MID") if "DC_MID" in df.columns else pd.Series(0, index=df.index),
    "DC_LOWER":  lambda df: _price_vs(df, "DC_MID") if "DC_MID" in df.columns else pd.Series(0, index=df.index),
    "BB_mid":    lambda df: _price_vs(df, "BB_mid"),
    "BB_upper":  lambda df: _price_vs(df, "BB_mid") if "BB_mid" in df.columns else pd.Series(0, index=df.index),
    "BB_lower":  lambda df: _price_vs(df, "BB_mid") if "BB_mid" in df.columns else pd.Series(0, index=df.index),
    "MA_short":  lambda df: (df["MA_short"] > df["MA_long"]).astype(int) * 2 - 1 if "MA_long" in df.columns else pd.Series(0, index=df.index),
    "MA_long":   lambda df: (df["MA_short"] > df["MA_long"]).astype(int) * 2 - 1 if "MA_short" in df.columns else pd.Series(0, index=df.index),

    # ── ADX：+DI vs -DI ──
    "ADX":      lambda df: (df["PLUS_DI"] > df["MINUS_DI"]).astype(int) * 2 - 1 if "PLUS_DI" in df.columns else pd.Series(0, index=df.index),
    "PLUS_DI":  lambda df: (df["PLUS_DI"] > df["MINUS_DI"]).astype(int) * 2 - 1 if "MINUS_DI" in df.columns else pd.Series(0, index=df.index),
    "MINUS_DI": lambda df: (df["PLUS_DI"] > df["MINUS_DI"]).astype(int) * 2 - 1 if "PLUS_DI" in df.columns else pd.Series(0, index=df.index),

    # ── アルーン ──
    "AROON_UP":   lambda df: _sign(df["AROON_UP"] - df["AROON_DOWN"]) if "AROON_DOWN" in df.columns else _sign(df["AROON_UP"] - 50),
    "AROON_DOWN": lambda df: _sign(df["AROON_UP"] - df["AROON_DOWN"]) if "AROON_UP" in df.columns else _sign(50 - df["AROON_DOWN"]),

    # ── 一目均衡表 ──
    "ICHI_TENKAN": lambda df: (df["ICHI_TENKAN"] > df["ICHI_KIJUN"]).astype(int) * 2 - 1 if "ICHI_KIJUN" in df.columns else _price_vs(df, "ICHI_TENKAN"),
    "ICHI_KIJUN":  lambda df: (df["ICHI_TENKAN"] > df["ICHI_KIJUN"]).astype(int) * 2 - 1 if "ICHI_TENKAN" in df.columns else _price_vs(df, "ICHI_KIJUN"),
    "ICHI_SPAN_A": lambda df: _price_vs(df, "ICHI_SPAN_A"),
    "ICHI_SPAN_B": lambda df: _price_vs(df, "ICHI_SPAN_B"),

    # ── 空売り圧力・スクイーズ ──
    "SELL_PRESSURE": lambda df: df["SELL_PRESSURE"].apply(lambda x: -1 if x > 0.65 else 0),
    "SQUEEZE_SCORE": lambda df: df["SQUEEZE_SCORE"].apply(lambda x: 1 if x > 0.55 else 0),

    # ── ATR：単独ではシグナル生成不可（ボラティリティ指標）──
    "ATR": lambda df: pd.Series(0, index=df.index),

    # ── マスインデックス：転換シグナル ──
    "MASS_INDEX": lambda df: _sign(
        pd.Series(
            [1 if (df["MASS_INDEX"].iloc[i] < 26.5 and df["MASS_INDEX"].iloc[i - 1] > 27
                   and df["Close"].iloc[i] > df["Close"].iloc[i - 1])
             else (-1 if (df["MASS_INDEX"].iloc[i] < 26.5 and df["MASS_INDEX"].iloc[i - 1] > 27
                          and df["Close"].iloc[i] < df["Close"].iloc[i - 1])
                   else 0)
             for i in range(len(df))],
            index=df.index,
        )
    ),
}


def generate_ext_signals(df: pd.DataFrame, checked_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    チェックされた拡張指標列ごとにシグナル列を生成する。
    戻り値: (シグナル列が追加されたdf, 生成されたシグナル列名リスト)
    """
    df = df.copy()
    signal_cols: list[str] = []

    for col in checked_cols:
        if col not in df.columns:
            continue
        rule = _RULES.get(col)
        if rule is None:
            # デフォルト：ゼロクロス
            rule = lambda df, c=col: _sign(df[c])
        try:
            sig = rule(df).astype(float)
            sig_col = f"_sig_{col}"
            df[sig_col] = sig
            signal_cols.append(sig_col)
        except Exception:
            pass

    return df, signal_cols


def build_ext_composite(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """
    任意のシグナル列リストの多数決で ext_composite_signal / ext_order を生成する。
    """
    df = df.copy()
    valid = [c for c in signal_cols if c in df.columns]

    if not valid:
        df["ext_composite_signal"] = 0
        df["ext_vote_sum"] = 0
        df["ext_order"] = 0
        return df

    df["ext_vote_sum"] = df[valid].sum(axis=1)
    threshold = max(1, len(valid) / 2)

    df["ext_composite_signal"] = 0
    df.loc[df["ext_vote_sum"] >= threshold, "ext_composite_signal"] = 1
    df.loc[df["ext_vote_sum"] <= -threshold, "ext_composite_signal"] = -1
    df["ext_order"] = df["ext_composite_signal"].diff()
    return df
