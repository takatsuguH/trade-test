"""テクニカル指標と将来リターンの相関分析"""
import pandas as pd
import numpy as np

# 分析対象外の列（シグナル列・生値など）
_EXCLUDE = {
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "composite_signal", "order", "vote_sum",
    "MA_signal", "RSI_signal", "MACD_vote", "BB_signal",
    "RSI_ob", "RSI_os", "ICHI_CHIKOU",  # 先行値は除外
}

# 各指標の日本語名・種別・説明
INDICATOR_META: dict[str, dict] = {
    "STOCH_K":     {"name": "ストキャスティクス %K",    "type": "oscillator", "range": (0, 100),   "desc": "過買い(80超)/過売り(20未満)を判定するモメンタム指標"},
    "STOCH_D":     {"name": "ストキャスティクス %D",    "type": "oscillator", "range": (0, 100),   "desc": "%Kの3日移動平均。シグナル線として利用"},
    "CCI":         {"name": "CCI（商品チャネル指数）", "type": "oscillator", "range": None,       "desc": "±100を超えるとトレンド発生のサイン"},
    "WILLIAMS_R":  {"name": "ウィリアムズ %R",         "type": "oscillator", "range": (-100, 0),  "desc": "-80未満で過売り、-20超で過買い"},
    "ROC":         {"name": "ROC（変化率）",           "type": "oscillator", "range": None,       "desc": "N日間の価格変化率(%)"},
    "MOM":         {"name": "モメンタム",              "type": "oscillator", "range": None,       "desc": "現在値とN日前の価格差。勢いを測定"},
    "ADX":         {"name": "ADX（方向性指数）",       "type": "oscillator", "range": (0, 100),   "desc": "25超でトレンド発生。方向は±DIで判定"},
    "PLUS_DI":     {"name": "+DI（上昇方向性）",       "type": "oscillator", "range": (0, 100),   "desc": "+DIが-DIを上回ると上昇トレンド"},
    "MINUS_DI":    {"name": "-DI（下降方向性）",       "type": "oscillator", "range": (0, 100),   "desc": "-DIが+DIを上回ると下降トレンド"},
    "ATR":         {"name": "ATR（真の値幅）",         "type": "oscillator", "range": None,       "desc": "市場のボラティリティを円単位で表示"},
    "OBV":         {"name": "OBV（出来高加重）",       "type": "volume",     "range": None,       "desc": "価格と出来高の乖離で需給を分析"},
    "MFI":         {"name": "MFI（資金流量指数）",     "type": "oscillator", "range": (0, 100),   "desc": "出来高を加味したRSI。80超で過買い、20未満で過売り"},
    "CMF":         {"name": "CMF（チャイキン資金流）", "type": "oscillator", "range": (-1, 1),    "desc": "正値で資金流入（強気）、負値で流出（弱気）"},
    "DEMA":        {"name": "DEMA（二重EMA）",        "type": "overlay",    "range": None,       "desc": "ノイズを低減した高速移動平均"},
    "TEMA":        {"name": "TEMA（三重EMA）",        "type": "overlay",    "range": None,       "desc": "さらに遅延を減らした移動平均"},
    "HMA":         {"name": "HMA（ハル移動平均）",    "type": "overlay",    "range": None,       "desc": "反応が最速な移動平均"},
    "VWAP":        {"name": "VWAP（出来高加重平均）", "type": "overlay",    "range": None,       "desc": "機関投資家が基準とする平均コスト"},
    "AROON_UP":    {"name": "アルーン上昇",           "type": "oscillator", "range": (0, 100),   "desc": "N日間で高値を付けた日からの経過度"},
    "AROON_DOWN":  {"name": "アルーン下降",           "type": "oscillator", "range": (0, 100),   "desc": "N日間で安値を付けた日からの経過度"},
    "AROON_OSC":   {"name": "アルーンオシレーター",   "type": "oscillator", "range": (-100, 100),"desc": "上昇−下降。正値で上昇優位、負値で下降優位"},
    "TRIX":        {"name": "TRIX",                  "type": "oscillator", "range": None,       "desc": "三重EMAの変化率。ゼロクロスで売買サイン"},
    "DPO":         {"name": "DPO（デトレンド）",      "type": "oscillator", "range": None,       "desc": "長期トレンドを除去した価格変動のサイクルを抽出"},
    "FORCE_INDEX": {"name": "フォースインデックス",   "type": "oscillator", "range": None,       "desc": "価格変動×出来高で買い圧/売り圧を定量化"},
    "EOM":         {"name": "EOM（移動容易性）",      "type": "oscillator", "range": None,       "desc": "少ない出来高で大きく動く＝トレンドが強い"},
    "KC_UPPER":    {"name": "ケルトナー上限",         "type": "overlay",    "range": None,       "desc": "ATRベースの上限バンド（突破で強い上昇）"},
    "KC_MID":      {"name": "ケルトナー中心線",       "type": "overlay",    "range": None,       "desc": "ケルトナーチャネルの中心EMA"},
    "KC_LOWER":    {"name": "ケルトナー下限",         "type": "overlay",    "range": None,       "desc": "ATRベースの下限バンド（割れで強い下落）"},
    "DC_UPPER":    {"name": "ドンチャン上限",         "type": "overlay",    "range": None,       "desc": "N日間の最高値（ブレイクアウト判定）"},
    "DC_MID":      {"name": "ドンチャン中心",         "type": "overlay",    "range": None,       "desc": "最高値と最安値の中間"},
    "DC_LOWER":    {"name": "ドンチャン下限",         "type": "overlay",    "range": None,       "desc": "N日間の最安値（ブレイクダウン判定）"},
    "PSAR":        {"name": "パラボリックSAR",        "type": "overlay",    "range": None,       "desc": "価格の上下に反転ドットを表示。トレンド転換シグナル"},
    "UO":          {"name": "アルティメットオシレーター","type": "oscillator","range": (0, 100),  "desc": "3つの期間の買い圧を統合。70超で過買い、30未満で過売り"},
    "CMO":         {"name": "CMO（チャンデモメンタム）","type": "oscillator","range": (-100, 100),"desc": "上昇幅と下降幅の比率。±50で売買サイン"},
    "ICHI_TENKAN": {"name": "一目均衡表 転換線",      "type": "overlay",    "range": None,       "desc": "9日間の高値+安値÷2。短期トレンド"},
    "ICHI_KIJUN":  {"name": "一目均衡表 基準線",      "type": "overlay",    "range": None,       "desc": "26日間の高値+安値÷2。中期トレンドの支持/抵抗"},
    "ICHI_SPAN_A": {"name": "一目均衡表 先行スパンA", "type": "overlay",    "range": None,       "desc": "転換線と基準線の平均（26日先行）。雲の上辺"},
    "ICHI_SPAN_B": {"name": "一目均衡表 先行スパンB", "type": "overlay",    "range": None,       "desc": "52日間の中間値（26日先行）。雲の下辺"},
    "MASS_INDEX":  {"name": "マスインデックス",       "type": "oscillator", "range": None,       "desc": "27超→26.5割れでトレンド転換の可能性"},
    "COPPOCK":     {"name": "コポックカーブ",         "type": "oscillator", "range": None,       "desc": "ゼロライン上向き突破で強い買いサイン"},
    "SELL_PRESSURE": {"name": "空売り圧力指数",        "type": "oscillator", "range": (0, 1),     "desc": "大出来高を伴う下落の累積強度（0〜1）。0.65超で空売り圧力が高い"},
    "SQUEEZE_SCORE": {"name": "スクイーズスコア",     "type": "oscillator", "range": (0, 1),     "desc": "下落後の大出来高急騰強度（0〜1）。0.55超でショートスクイーズ兆候"},
    "MA_short":    {"name": "短期移動平均",           "type": "overlay",    "range": None,       "desc": "設定された短期MA"},
    "MA_long":     {"name": "長期移動平均",           "type": "overlay",    "range": None,       "desc": "設定された長期MA"},
    "RSI":         {"name": "RSI",                   "type": "oscillator", "range": (0, 100),   "desc": "相対力指数"},
    "MACD":        {"name": "MACD",                  "type": "oscillator", "range": None,       "desc": "Moving Average Convergence Divergence"},
    "BB_upper":    {"name": "ボリンジャー上限",       "type": "overlay",    "range": None,       "desc": ""},
    "BB_lower":    {"name": "ボリンジャー下限",       "type": "overlay",    "range": None,       "desc": ""},
}


def analyze_correlations(
    df: pd.DataFrame,
    forward_days: int = 5,
    min_corr: float = 0.15,
) -> list[dict]:
    """
    各テクニカル指標とN日後リターンの相関を計算し、|相関|>=min_corrのものを返す。
    戻り値: [{'col', 'name', 'corr', 'type', 'desc', 'range'}, ...]（相関の絶対値の降順）
    """
    future_ret = df["Close"].pct_change(forward_days).shift(-forward_days)

    results = []
    for col in df.columns:
        if col in _EXCLUDE:
            continue
        series = df[col]
        if series.isna().all() or series.nunique() <= 1:
            continue
        try:
            corr = float(series.corr(future_ret))
            if np.isnan(corr) or abs(corr) < min_corr:
                continue
            meta = INDICATOR_META.get(col, {})
            results.append({
                "col":   col,
                "name":  meta.get("name", col),
                "corr":  round(corr, 3),
                "type":  meta.get("type", "oscillator"),
                "desc":  meta.get("desc", ""),
                "range": meta.get("range"),
            })
        except Exception:
            pass

    return sorted(results, key=lambda x: -abs(x["corr"]))
