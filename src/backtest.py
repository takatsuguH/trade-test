"""バックテストエンジン — 複合シグナル + リスク管理対応"""
import pandas as pd
from src.risk.manager import RiskManager


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 1_000_000,
    risk: RiskManager | None = None,
    signal_col: str = "composite_signal",
    order_col: str = "order",
    max_shares: int = 0,
) -> dict:
    if risk is None:
        risk = RiskManager()

    df = df.dropna(subset=[signal_col]).copy()

    cash = initial_cash
    position = 0
    buy_price = 0.0
    last_sell_price = 0.0  # 直近の売却価格（買い戻し条件に使用）
    peak_value = initial_cash
    max_drawdown = 0.0
    trades = []
    portfolio_curve = []

    for idx, row in df.iterrows():
        price = float(row["Close"])
        order = float(row.get(order_col, 0))

        # ポートフォリオ時価を記録
        current_value = cash + position * price
        portfolio_curve.append({"date": idx, "value": current_value})

        # 最大ドローダウンを追跡
        peak_value = max(peak_value, current_value)
        drawdown = (current_value - peak_value) / peak_value
        max_drawdown = min(max_drawdown, drawdown)

        # 保有中のリスク管理チェック（損切り・利確）
        if position > 0:
            if price <= buy_price * (1 - risk.stop_loss):
                cash += position * price
                profit = (price - buy_price) * position
                trades.append({
                    "date": idx, "type": "STOP_LOSS",
                    "price": price, "shares": position, "profit": profit,
                })
                position = 0
                last_sell_price = price  # 売却価格を記録
                continue
            if price >= buy_price * (1 + risk.take_profit):
                cash += position * price
                profit = (price - buy_price) * position
                trades.append({
                    "date": idx, "type": "TAKE_PROFIT",
                    "price": price, "shares": position, "profit": profit,
                })
                position = 0
                last_sell_price = price  # 売却価格を記録
                continue

        # シグナルによる売買
        if order > 0 and position == 0:
            # 買い戻し条件チェック:
            #   rebuy_dip_pct == 0 → 無効（即時再エントリー）
            #   last_sell_price == 0 → 初回買いなので条件なし
            #   それ以外 → 売値より rebuy_dip_pct% 以上下落していることを確認
            can_rebuy = (
                risk.rebuy_dip_pct == 0
                or last_sell_price == 0
                or price <= last_sell_price * (1 - risk.rebuy_dip)
            )
            if can_rebuy:
                invest_cash = cash * risk.max_position
                shares = int(invest_cash // price // 100) * 100
                if max_shares > 0:
                    shares = min(shares, int(max_shares // 100) * 100)
                if shares > 0:
                    position = shares
                    buy_price = price
                    cash -= shares * price
                    last_sell_price = 0.0  # 買い戻し後はリセット
                    trades.append({
                        "date": idx, "type": "BUY",
                        "price": price, "shares": shares, "profit": None,
                    })

        elif order < 0 and position > 0:
            cash += position * price
            profit = (price - buy_price) * position
            trades.append({
                "date": idx, "type": "SELL",
                "price": price, "shares": position, "profit": profit,
            })
            position = 0
            last_sell_price = price  # 売却価格を記録

    # 最終日に保有株があれば時価清算
    final_price = float(df.iloc[-1]["Close"])
    total_value = cash + position * final_price
    total_return = (total_value - initial_cash) / initial_cash * 100

    trades_df = pd.DataFrame(trades)
    win_rate = 0.0
    if not trades_df.empty:
        closed = trades_df[trades_df["profit"].notna()]
        if len(closed) > 0:
            win_rate = (closed["profit"] > 0).sum() / len(closed) * 100

    return {
        "initial_cash": initial_cash,
        "final_value": total_value,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate,
        "trades": trades_df,
        "portfolio_curve": pd.DataFrame(portfolio_curve),
        "current_position": position,
    }
