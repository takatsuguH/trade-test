"""テクニカル指標の計算モジュール"""
import pandas as pd
import numpy as np


def add_ma(df: pd.DataFrame, short: int = 5, long: int = 25) -> pd.DataFrame:
    df["MA_short"] = df["Close"].rolling(short).mean()
    df["MA_long"] = df["Close"].rolling(long).mean()
    df["MA_signal"] = 0
    df.loc[df["MA_short"] > df["MA_long"], "MA_signal"] = 1
    df.loc[df["MA_short"] < df["MA_long"], "MA_signal"] = -1
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, overbought: int = 70, oversold: int = 30) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_ob"] = overbought
    df["RSI_os"] = oversold
    df["RSI_signal"] = 0
    df.loc[df["RSI"] < oversold, "RSI_signal"] = 1
    df.loc[df["RSI"] > overbought, "RSI_signal"] = -1
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_sig"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["MACD_vote"] = 0
    df.loc[df["MACD"] > df["MACD_sig"], "MACD_vote"] = 1
    df.loc[df["MACD"] < df["MACD_sig"], "MACD_vote"] = -1
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df["BB_mid"] = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_upper"] = df["BB_mid"] + std_dev * std
    df["BB_lower"] = df["BB_mid"] - std_dev * std
    df["BB_signal"] = 0
    df.loc[df["Close"] < df["BB_lower"], "BB_signal"] = 1
    df.loc[df["Close"] > df["BB_upper"], "BB_signal"] = -1
    return df


def calculate_all(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """設定に基づいて全指標を計算する。"""
    df = df.copy()
    if config.get("use_ma"):
        df = add_ma(df, config.get("ma_short", 5), config.get("ma_long", 25))
    if config.get("use_rsi"):
        df = add_rsi(df, config.get("rsi_period", 14), config.get("rsi_ob", 70), config.get("rsi_os", 30))
    if config.get("use_macd"):
        df = add_macd(df, config.get("macd_fast", 12), config.get("macd_slow", 26), config.get("macd_sig", 9))
    if config.get("use_bb"):
        df = add_bollinger(df, config.get("bb_period", 20), config.get("bb_std", 2.0))
    return df
