import sys
sys.path.insert(0, ".")

from src.data.fetcher import get_stock_data
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended
from src.strategies.composite import generate_composite_signal
from src.risk.manager import RiskManager
from src.backtest import run_backtest
from src.analysis.correlation import analyze_correlations

cfg = {
    "use_ma": True, "ma_short": 5, "ma_long": 25,
    "use_rsi": True, "rsi_period": 14, "rsi_ob": 70, "rsi_os": 30,
    "use_macd": True, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "use_bb": True, "bb_period": 20, "bb_std": 2.0,
}
ticker = "9984.T"
df = get_stock_data(ticker, period="1y")
df = calculate_all(df, cfg)
df = generate_composite_signal(df, ["MA", "RSI", "MACD", "BB"])
result = run_backtest(df, 1_000_000, RiskManager(5, 10, 100))

n_trades = len(result["trades"])
ret = result["total_return_pct"]
print(f"Trades={n_trades}, Return={ret:.1f}%")
print(f"Alert condition (>10 trades & negative return): {n_trades > 10 and ret < 0}")

df_ext = calculate_extended(df)
ext_cols = [c for c in df_ext.columns if c not in df.columns]
print(f"Extended cols added: {len(ext_cols)}")

corr_list = analyze_correlations(df_ext)
print(f"Correlated indicators (|r|>=0.15): {len(corr_list)}")
print("\nTop 10:")
for r in corr_list[:10]:
    direction = "上昇連動" if r["corr"] > 0 else "下落連動"
    print(f"  [{r['type']:12s}] {r['name']:25s} r={r['corr']:+.3f}  {direction}")
