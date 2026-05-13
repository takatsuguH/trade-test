import sys
sys.path.insert(0, ".")

from src.data.fetcher import get_stock_data
from src.indicators.calculator import calculate_all
from src.strategies.composite import generate_composite_signal
from src.risk.manager import RiskManager
from src.backtest import run_backtest

cfg = {
    "use_ma": True, "ma_short": 5, "ma_long": 25,
    "use_rsi": True, "rsi_period": 14, "rsi_ob": 70, "rsi_os": 30,
    "use_macd": True, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "use_bb": True, "bb_period": 20, "bb_std": 2.0,
}
active = ["MA", "RSI", "MACD", "BB"]
risk = RiskManager(5.0, 10.0, 100.0)

labels = {1: "BUY", -1: "SELL", 0: "HOLD"}

for ticker in ["7203.T", "6758.T", "9984.T"]:
    df = get_stock_data(ticker, period="1y")
    df = calculate_all(df, cfg)
    df = generate_composite_signal(df, active)
    result = run_backtest(df, 1_000_000, risk)
    sig = int(df.iloc[-1]["composite_signal"])
    r = result["total_return_pct"]
    dd = result["max_drawdown_pct"]
    w = result["win_rate_pct"]
    n = len(result["trades"])
    print(f"{ticker}: {labels[sig]} | return={r:.1f}% | DD={dd:.1f}% | win={w:.0f}% | trades={n}")

print("\n全モジュール正常動作確認完了")
