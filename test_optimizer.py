"""最適化エンジンの単体テスト"""
import sys, time
sys.path.insert(0, ".")

from src.data.fetcher import get_stock_data
from src.indicators.calculator import calculate_all
from src.indicators.extended import calculate_extended
from src.strategies.composite import generate_composite_signal
from src.risk.manager import RiskManager
from src.backtest import run_backtest
from src.analysis.correlation import analyze_correlations, INDICATOR_META
from src.optimization.searcher import find_best_combination

cfg = {
    "use_ma": True, "ma_short": 25, "ma_long": 75,
    "use_rsi": True, "rsi_period": 14, "rsi_ob": 60, "rsi_os": 35,
    "use_macd": True, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "use_bb": True, "bb_period": 20, "bb_std": 2.0,
}

for ticker in ["6758.T", "7203.T"]:
    print(f"\n{'='*50}")
    print(f"銘柄: {ticker}")

    df = get_stock_data(ticker, period="1y")
    df = calculate_all(df, cfg)
    df = generate_composite_signal(df, ["MA", "RSI", "MACD", "BB"])
    risk = RiskManager(5, 10, 100)
    orig = run_backtest(df, 1_000_000, risk)
    print(f"元: リターン={orig['total_return_pct']:.2f}%  取引={len(orig['trades'])}")

    if len(orig["trades"]) <= 10 or orig["total_return_pct"] >= 0:
        print("  → アラート条件未達、スキップ")
        continue

    df_ext = calculate_extended(df)
    corr_list = analyze_correlations(df_ext, forward_days=5, min_corr=0.15)
    candidates = [r["col"] for r in corr_list]
    print(f"相関指標数: {len(candidates)}")

    t0 = time.perf_counter()
    best_cols, best_ret, method = find_best_combination(df_ext, candidates)
    elapsed = time.perf_counter() - t0

    names = [INDICATOR_META.get(c, {}).get("name", c) for c in best_cols]
    print(f"探索手法: {method}")
    print(f"推奨: {' + '.join(names)}")
    print(f"推定リターン: {best_ret:.2f}%  (改善: {best_ret - orig['total_return_pct']:+.2f}%)")
    print(f"処理時間: {elapsed:.2f}秒")
