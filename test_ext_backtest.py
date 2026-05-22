"""拡張バックテストの動作確認"""
import sys
sys.path.insert(0, ".")

from src.data.fetcher import get_stock_data
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended
from src.indicators.signal_generator import generate_ext_signals, build_ext_composite
from src.strategies.composite import generate_composite_signal
from src.risk.manager import RiskManager
from src.backtest import run_backtest
from src.analysis.correlation import analyze_correlations

cfg = {
    "use_ma": True, "ma_short": 25, "ma_long": 75,
    "use_rsi": True, "rsi_period": 14, "rsi_ob": 60, "rsi_os": 35,
    "use_macd": True, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "use_bb": True, "bb_period": 20, "bb_std": 2.0,
}
ticker = "6758.T"  # リターンがマイナスの銘柄
risk = RiskManager(5, 10, 100)

df = get_stock_data(ticker, period="1y")
df = calculate_all(df, cfg)
df = generate_composite_signal(df, ["MA", "RSI", "MACD", "BB"])
orig_result = run_backtest(df, 1_000_000, risk)

print(f"=== 元のバックテスト ({ticker}) ===")
print(f"リターン: {orig_result['total_return_pct']:.2f}%")
print(f"取引回数: {len(orig_result['trades'])}")
print(f"最大DD:   {orig_result['max_drawdown_pct']:.2f}%")
print(f"勝率:     {orig_result['win_rate_pct']:.1f}%")

# アラート条件チェック
if len(orig_result["trades"]) > 10 and orig_result["total_return_pct"] < 0:
    print("\n✅ アラート条件満たす → 相関分析実行")

    df_ext = calculate_extended(df)
    corr_list = analyze_correlations(df_ext, forward_days=5, min_corr=0.15)
    print(f"相関指標数: {len(corr_list)}")

    # 上位5指標をチェックしたと仮定
    top5 = [r["col"] for r in corr_list[:5]]
    print(f"使用指標: {top5}")

    df_bt, sig_cols = generate_ext_signals(df_ext, top5)
    df_bt = build_ext_composite(df_bt, sig_cols)
    print(f"シグナル列数: {len(sig_cols)}")

    ext_result = run_backtest(df_bt, 1_000_000, risk,
                              signal_col="ext_composite_signal",
                              order_col="ext_order")

    print(f"\n=== 拡張バックテスト ===")
    print(f"リターン: {ext_result['total_return_pct']:.2f}%  (元: {orig_result['total_return_pct']:.2f}%)")
    print(f"取引回数: {len(ext_result['trades'])}  (元: {len(orig_result['trades'])})")
    print(f"最大DD:   {ext_result['max_drawdown_pct']:.2f}%  (元: {orig_result['max_drawdown_pct']:.2f}%)")
    print(f"勝率:     {ext_result['win_rate_pct']:.1f}%  (元: {orig_result['win_rate_pct']:.1f}%)")
else:
    print(f"\n⚠️ アラート条件未達（取引{len(orig_result['trades'])}回, リターン{orig_result['total_return_pct']:.1f}%）")
    print("7203.T で試してみます")
    ticker2 = "7203.T"
    df2 = get_stock_data(ticker2, period="1y")
    df2 = calculate_all(df2, cfg)
    df2 = generate_composite_signal(df2, ["MA", "RSI", "MACD", "BB"])
    r2 = run_backtest(df2, 1_000_000, risk)
    print(f"{ticker2}: 取引{len(r2['trades'])}回, リターン{r2['total_return_pct']:.1f}%")
