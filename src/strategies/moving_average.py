"""移動平均クロス戦略 — 最もシンプルなトレード戦略の例"""
import pandas as pd


def add_signals(df: pd.DataFrame, short_window: int = 5, long_window: int = 25) -> pd.DataFrame:
    """
    短期・長期移動平均のゴールデンクロス/デッドクロスでシグナルを生成する。
    signal=1: 買いシグナル（ゴールデンクロス）
    signal=-1: 売りシグナル（デッドクロス）
    signal=0: ポジションなし
    """
    df = df.copy()
    df["MA_short"] = df["Close"].rolling(window=short_window).mean()
    df["MA_long"] = df["Close"].rolling(window=long_window).mean()

    df["signal"] = 0
    df.loc[df["MA_short"] > df["MA_long"], "signal"] = 1   # 買い
    df.loc[df["MA_short"] < df["MA_long"], "signal"] = -1  # 売り

    # シグナルが変化した日だけ注文（持ち続けるのではなく、クロス時だけ）
    df["order"] = df["signal"].diff()
    return df


def get_latest_signal(df: pd.DataFrame) -> str:
    """最新のシグナルを返す。"""
    latest = df.dropna().iloc[-1]
    if latest["order"] > 0:
        return "BUY"
    elif latest["order"] < 0:
        return "SELL"
    return "HOLD"


if __name__ == "__main__":
    from src.data.fetcher import get_stock_data

    df = get_stock_data("7203.T", period="6mo")
    df = add_signals(df, short_window=5, long_window=25)

    print("=== 最新5日 ===")
    print(df.tail(5)[["Close", "MA_short", "MA_long", "signal", "order"]])
    print(f"\n現在のシグナル: {get_latest_signal(df)}")
