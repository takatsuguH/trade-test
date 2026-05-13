"""バックテスト実行スクリプト（チャート表示なし版）"""
import sys
sys.path.insert(0, ".")

from src.data.fetcher import get_stock_data
from src.strategies.moving_average import add_signals
from src.backtest import run_backtest

ticker = "7203.T"
df = get_stock_data(ticker, period="1y")
df = add_signals(df, short_window=5, long_window=25)
result = run_backtest(df, initial_cash=1_000_000)

initial = result["initial_cash"]
final = result["final_value"]
ret = result["total_return_pct"]
print(f"=== バックテスト結果: {ticker} ===")
print(f"初期資金: {initial:,.0f}円")
print(f"最終資産: {final:,.0f}円")
print(f"リターン: {ret:.2f}%")

trades = result["trades"]
if not trades.empty:
    print(f"\n取引回数: {len(trades)}回")
    print(trades.to_string(index=False))
