"""株価データ取得モジュール（yfinanceを使用）"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    株価データを取得する。
    ticker: 銘柄コード（例: "7203.T" はトヨタ、"^N225" は日経平均）
    period: 取得期間 ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y")
    interval: 足の種類 ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"データが取得できませんでした: {ticker}")
    df.index = pd.to_datetime(df.index)
    return df


def get_multiple_stocks(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """複数銘柄のデータを一括取得する。"""
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = get_stock_data(ticker, period)
            print(f"取得成功: {ticker}")
        except Exception as e:
            print(f"取得失敗: {ticker} - {e}")
    return result


if __name__ == "__main__":
    # 動作確認
    print("=== 日経平均 直近1ヶ月 ===")
    df = get_stock_data("^N225", period="1mo")
    print(df.tail(5)[["Open", "High", "Low", "Close", "Volume"]])

    print("\n=== トヨタ(7203) 直近1ヶ月 ===")
    df = get_stock_data("7203.T", period="1mo")
    print(df.tail(5)[["Open", "High", "Low", "Close", "Volume"]])
