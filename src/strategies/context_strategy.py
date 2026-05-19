"""
コンテキストベース売買シグナル生成
相場状態（TREND / RANGE）を先に判定し、指標の役割に基づいた重み付きスコアリングで判断する。
既存の単純投票方式（composite.py）の代替として設計。
"""
import pandas as pd
import numpy as np

_DEFAULT_THRESHOLD = 5   # BUY/SELLに必要な最低スコア（最大±10）


class MarketAnalyzer:
    """相場状態（TREND_UP / TREND_DOWN / RANGE）を判定する。"""

    def __init__(self, ma_slope_window: int = 3, bb_expand_window: int = 10):
        self.ma_slope_window = ma_slope_window
        self.bb_expand_window = bb_expand_window

    def detect_market_state(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series("RANGE", index=df.index, dtype=str)

        if "MA_short" not in df.columns or "MA_long" not in df.columns:
            return result

        ma_slope = df["MA_short"].diff(self.ma_slope_window).fillna(0)

        bb_expanding = pd.Series(False, index=df.index)
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            bb_width = df["BB_upper"] - df["BB_lower"]
            bb_avg = bb_width.rolling(self.bb_expand_window).mean()
            bb_expanding = bb_width > bb_avg.fillna(0)

        uptrend = (
            (df["MA_short"] > df["MA_long"])
            & (df["Close"] > df["MA_short"])
            & (ma_slope > 0)
            & bb_expanding
        )
        downtrend = (
            (df["MA_short"] < df["MA_long"])
            & (df["Close"] < df["MA_short"])
            & (ma_slope < 0)
            & bb_expanding
        )

        result[uptrend] = "TREND_UP"
        result[downtrend] = "TREND_DOWN"
        return result


class IndicatorEvaluator:
    """各指標を役割に応じて評価しスコアを返す（合計最大±10）。"""

    @staticmethod
    def evaluate_MA(df: pd.DataFrame) -> pd.Series:
        """トレンド判定役割：MA整列 → ±3"""
        score = pd.Series(0, index=df.index, dtype=float)
        if "MA_short" not in df.columns or "MA_long" not in df.columns:
            return score
        score[df["MA_short"] > df["MA_long"]] = 3.0
        score[df["MA_short"] < df["MA_long"]] = -3.0
        return score

    @staticmethod
    def evaluate_MACD(df: pd.DataFrame) -> pd.Series:
        """トレンド判定役割：MACD方向 → ±3"""
        score = pd.Series(0, index=df.index, dtype=float)
        if "MACD" not in df.columns or "MACD_sig" not in df.columns:
            return score
        macd_above = df["MACD"] > df["MACD_sig"]
        if "MACD_hist" in df.columns:
            hist_rising = df["MACD_hist"].diff().fillna(0) > 0
        else:
            hist_rising = macd_above
        score[macd_above & hist_rising] = 3.0
        score[~macd_above & ~hist_rising] = -3.0
        return score

    @staticmethod
    def evaluate_RSI(
        df: pd.DataFrame,
        market_state: pd.Series,
        rsi_ob: int = 70,
        rsi_os: int = 30,
    ) -> pd.Series:
        """エントリータイミング役割：相場状態別 RSI → ±2"""
        score = pd.Series(0, index=df.index, dtype=float)
        if "RSI" not in df.columns:
            return score

        rsi_diff = df["RSI"].diff().fillna(0)
        mid = (rsi_ob + rsi_os) / 2  # 50相当の中間値

        # TREND_UP：RSI mid-5 〜 mid+5 で反発 → BUY
        mask_up = (market_state == "TREND_UP") & df["RSI"].between(mid - 5, mid + 5) & (rsi_diff > 0)
        score[mask_up] = 2.0

        # TREND_DOWN：RSI mid 〜 mid+10 で反落 → SELL
        mask_down = (market_state == "TREND_DOWN") & df["RSI"].between(mid, mid + 10) & (rsi_diff < 0)
        score[mask_down] = -2.0

        # RANGE：過売 → BUY、過買 → SELL
        mask_range = market_state == "RANGE"
        score[mask_range & (df["RSI"] < rsi_os)] = 2.0
        score[mask_range & (df["RSI"] > rsi_ob)] = -2.0

        return score

    @staticmethod
    def evaluate_BB(df: pd.DataFrame, market_state: pd.Series) -> pd.Series:
        """エントリータイミング役割：相場状態別 BB → ±2"""
        score = pd.Series(0, index=df.index, dtype=float)
        if "BB_mid" not in df.columns:
            return score

        # TREND_UP：押し目（BB_mid 付近） → BUY
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            bb_range = (df["BB_upper"] - df["BB_lower"]).clip(lower=1e-9)
            near_mid = (df["Close"] - df["BB_mid"]).abs() / bb_range < 0.25
            score[(market_state == "TREND_UP") & near_mid] = 2.0

        # RANGE：BB下限タッチ → BUY、BB上限タッチ → SELL
        mask_range = market_state == "RANGE"
        if "BB_lower" in df.columns:
            score[mask_range & (df["Close"] <= df["BB_lower"])] = 2.0
        if "BB_upper" in df.columns:
            score[mask_range & (df["Close"] >= df["BB_upper"])] = -2.0

        return score


class StrategyEngine:
    """禁止ルール・フィルター・ダイバージェンスを適用する。"""

    @staticmethod
    def detect_divergence(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
        """簡易ダイバージェンス検出（bearish, bullish）。"""
        bearish = pd.Series(False, index=df.index)
        bullish = pd.Series(False, index=df.index)
        if "RSI" not in df.columns:
            return bearish, bullish

        price_hi = df["Close"].rolling(window).max()
        rsi_hi = df["RSI"].rolling(window).max()
        price_lo = df["Close"].rolling(window).min()
        rsi_lo = df["RSI"].rolling(window).min()

        # 価格高値更新 & RSI高値低下 → 弱気ダイバージェンス
        bearish = (price_hi > price_hi.shift(window).fillna(0)) & (rsi_hi < rsi_hi.shift(window).fillna(100))
        # 価格安値更新 & RSI安値上昇 → 強気ダイバージェンス
        bullish = (price_lo < price_lo.shift(window).fillna(float("inf"))) & (rsi_lo > rsi_lo.shift(window).fillna(0))

        return bearish.fillna(False), bullish.fillna(False)

    @staticmethod
    def apply_forbidden_rules(
        signal: pd.Series,
        df: pd.DataFrame,
        market_state: pd.Series,
        rsi_ob: int = 70,
        rsi_os: int = 30,
    ) -> pd.Series:
        """TREND相場でのRSI過熱・売られすぎエントリーを禁止。"""
        signal = signal.copy()
        if "RSI" not in df.columns:
            return signal
        trend_mask = market_state.str.startswith("TREND")
        signal[(trend_mask) & (df["RSI"] > rsi_ob) & (signal == -1)] = 0
        signal[(trend_mask) & (df["RSI"] < rsi_os) & (signal == 1)] = 0
        return signal

    @staticmethod
    def apply_filters(signal: pd.Series, df: pd.DataFrame) -> pd.Series:
        """低ボラ・弱トレンドフィルター（エントリースキップ）。"""
        signal = signal.copy()

        # 低ボラ回避：BB幅が極小（終値の1%未満）
        if all(c in df.columns for c in ("BB_upper", "BB_lower", "BB_mid")):
            bb_width_pct = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"].replace(0, np.nan)
            signal[bb_width_pct.fillna(1) < 0.01] = 0

        # 弱トレンド回避：MACD ≈ 0
        if "MACD" in df.columns and "BB_mid" in df.columns:
            threshold = (df["BB_mid"].abs() * 0.0001).fillna(0.01)
            signal[df["MACD"].abs() < threshold] = 0

        return signal


class SignalGenerator:
    """全処理を統合して最終シグナルを生成する。"""

    def __init__(
        self,
        score_threshold: int = _DEFAULT_THRESHOLD,
        rsi_ob: int = 70,
        rsi_os: int = 30,
    ):
        self.score_threshold = score_threshold
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os
        self._analyzer = MarketAnalyzer()
        self._evaluator = IndicatorEvaluator()
        self._engine = StrategyEngine()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. 相場状態判定
        market_state = self._analyzer.detect_market_state(df)
        df["market_state"] = market_state

        # 2. 重み付きスコア計算
        score = (
            self._evaluator.evaluate_MA(df)
            + self._evaluator.evaluate_MACD(df)
            + self._evaluator.evaluate_RSI(df, market_state, self.rsi_ob, self.rsi_os)
            + self._evaluator.evaluate_BB(df, market_state)
        )
        df["context_score"] = score

        # 3. ダイバージェンス検出
        bearish_div, bullish_div = self._engine.detect_divergence(df)
        df["divergence_bearish"] = bearish_div
        df["divergence_bullish"] = bullish_div

        # 4. スコア → 生シグナル
        raw = pd.Series(0, index=df.index, dtype=int)
        raw[score >= self.score_threshold] = 1
        raw[score <= -self.score_threshold] = -1

        # ダイバージェンス時は逆方向エントリーを禁止
        raw[bearish_div & (raw == 1)] = 0
        raw[bullish_div & (raw == -1)] = 0

        # 5. 禁止ルール適用
        raw = self._engine.apply_forbidden_rules(raw, df, market_state, self.rsi_ob, self.rsi_os)

        # 6. フィルター適用
        final = self._engine.apply_filters(raw, df)

        # 7. 既存インタフェース互換列（composite.py と同じ列名で出力）
        df["composite_signal"] = final
        df["vote_sum"] = score      # スコアを vote_sum として出力（ファンダ統合で加算される）
        df["order"] = final.diff()

        return df


def generate_context_signal(
    df: pd.DataFrame,
    active: list[str],
    extra_sig_cols: list[str] | None = None,
    score_threshold: int = _DEFAULT_THRESHOLD,
    rsi_ob: int = 70,
    rsi_os: int = 30,
) -> pd.DataFrame:
    """
    app.py の merge_all_signals / generate_composite_signal と互換の公開関数。
    active / extra_sig_cols は将来の拡張用（現バージョンでは参照しない）。
    """
    return SignalGenerator(
        score_threshold=score_threshold,
        rsi_ob=rsi_ob,
        rsi_os=rsi_os,
    ).generate(df)
