"""拡張テクニカル指標 — 25種類以上を純粋なpandas/numpyで実装"""
import pandas as pd
import numpy as np


def add_stochastic(df: pd.DataFrame, k=14, d=3) -> pd.DataFrame:
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    df["STOCH_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["STOCH_D"] = df["STOCH_K"].rolling(d).mean()
    return df


def add_cci(df: pd.DataFrame, period=20) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - mean_tp) / (0.015 * mad.replace(0, np.nan))
    return df


def add_williams_r(df: pd.DataFrame, period=14) -> pd.DataFrame:
    h = df["High"].rolling(period).max()
    l = df["Low"].rolling(period).min()
    df["WILLIAMS_R"] = -100 * (h - df["Close"]) / (h - l).replace(0, np.nan)
    return df


def add_roc(df: pd.DataFrame, period=10) -> pd.DataFrame:
    df["ROC"] = df["Close"].pct_change(period) * 100
    return df


def add_momentum(df: pd.DataFrame, period=10) -> pd.DataFrame:
    df["MOM"] = df["Close"] - df["Close"].shift(period)
    return df


def add_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    plus_dm = h.diff().clip(lower=0).where(h.diff() > -l.diff(), 0)
    minus_dm = (-l.diff()).clip(lower=0).where(-l.diff() > h.diff(), 0)
    atr14 = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(span=period, adjust=False).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    return df


def add_atr(df: pd.DataFrame, period=14) -> pd.DataFrame:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    sign = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["OBV"] = (df["Volume"] * sign).cumsum()
    return df


def add_mfi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = tp * df["Volume"]
    pos = raw_mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = raw_mf.where(tp <= tp.shift(), 0).rolling(period).sum()
    df["MFI"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return df


def add_cmf(df: pd.DataFrame, period=20) -> pd.DataFrame:
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    df["CMF"] = (mfm * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)
    return df


def add_dema(df: pd.DataFrame, period=20) -> pd.DataFrame:
    ema = df["Close"].ewm(span=period, adjust=False).mean()
    df["DEMA"] = 2 * ema - ema.ewm(span=period, adjust=False).mean()
    return df


def add_tema(df: pd.DataFrame, period=20) -> pd.DataFrame:
    ema1 = df["Close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    df["TEMA"] = 3 * ema1 - 3 * ema2 + ema3
    return df


def add_hma(df: pd.DataFrame, period=20) -> pd.DataFrame:
    half = max(period // 2, 2)
    wma = lambda s, p: s.rolling(p).apply(
        lambda x: np.dot(x, np.arange(1, p + 1)) / np.arange(1, p + 1).sum(), raw=True
    )
    raw = 2 * wma(df["Close"], half) - wma(df["Close"], period)
    df["HMA"] = wma(raw, max(int(np.sqrt(period)), 2))
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def add_aroon(df: pd.DataFrame, period=25) -> pd.DataFrame:
    n = period + 1
    df["AROON_UP"] = df["High"].rolling(n).apply(lambda x: (np.argmax(x) / period) * 100, raw=True)
    df["AROON_DOWN"] = df["Low"].rolling(n).apply(lambda x: (np.argmin(x) / period) * 100, raw=True)
    df["AROON_OSC"] = df["AROON_UP"] - df["AROON_DOWN"]
    return df


def add_trix(df: pd.DataFrame, period=15) -> pd.DataFrame:
    e1 = df["Close"].ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    df["TRIX"] = e3.pct_change() * 100
    return df


def add_dpo(df: pd.DataFrame, period=20) -> pd.DataFrame:
    shift = period // 2 + 1
    df["DPO"] = df["Close"] - df["Close"].rolling(period).mean().shift(shift)
    return df


def add_force_index(df: pd.DataFrame, period=13) -> pd.DataFrame:
    fi = df["Close"].diff() * df["Volume"]
    df["FORCE_INDEX"] = fi.ewm(span=period, adjust=False).mean()
    return df


def add_eom(df: pd.DataFrame, period=14) -> pd.DataFrame:
    mid_diff = ((df["High"] + df["Low"]) / 2).diff()
    box = (df["Volume"] / 1e6) / (df["High"] - df["Low"]).replace(0, np.nan)
    df["EOM"] = (mid_diff / box.replace(0, np.nan)).rolling(period).mean()
    return df


def add_keltner(df: pd.DataFrame, period=20, mult=2.0) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mid = tp.ewm(span=period, adjust=False).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    df["KC_UPPER"] = mid + mult * atr
    df["KC_MID"] = mid
    df["KC_LOWER"] = mid - mult * atr
    return df


def add_donchian(df: pd.DataFrame, period=20) -> pd.DataFrame:
    df["DC_UPPER"] = df["High"].rolling(period).max()
    df["DC_LOWER"] = df["Low"].rolling(period).min()
    df["DC_MID"] = (df["DC_UPPER"] + df["DC_LOWER"]) / 2
    return df


def add_psar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2) -> pd.DataFrame:
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    psar = np.full(n, np.nan)
    bull = True
    af = af_start
    ep = lows[0]
    hp = highs[0]
    lp = lows[0]
    psar[0] = lows[0]
    psar[1] = lows[1]

    for i in range(2, n):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], lows[i - 1], lows[i - 2])
            if lows[i] < psar[i]:
                bull, psar[i], lp, af = False, hp, lows[i], af_start
            elif highs[i] > hp:
                hp = highs[i]
                af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], highs[i - 1], highs[i - 2])
            if highs[i] > psar[i]:
                bull, psar[i], hp, af = True, lp, highs[i], af_start
            elif lows[i] < lp:
                lp = lows[i]
                af = min(af + af_step, af_max)

    df["PSAR"] = psar
    return df


def add_ultimate_oscillator(df: pd.DataFrame, p1=7, p2=14, p3=28) -> pd.DataFrame:
    prev_close = df["Close"].shift()
    bp = df["Close"] - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    def avg(p):
        return bp.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)

    df["UO"] = 100 * (4 * avg(p1) + 2 * avg(p2) + avg(p3)) / 7
    return df


def add_cmo(df: pd.DataFrame, period=14) -> pd.DataFrame:
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(period).sum()
    dn = (-delta).clip(lower=0).rolling(period).sum()
    df["CMO"] = 100 * (up - dn) / (up + dn).replace(0, np.nan)
    return df


def add_ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52) -> pd.DataFrame:
    def mid(h, l, p):
        return (h.rolling(p).max() + l.rolling(p).min()) / 2

    df["ICHI_TENKAN"] = mid(df["High"], df["Low"], tenkan)
    df["ICHI_KIJUN"] = mid(df["High"], df["Low"], kijun)
    df["ICHI_SPAN_A"] = ((df["ICHI_TENKAN"] + df["ICHI_KIJUN"]) / 2).shift(kijun)
    df["ICHI_SPAN_B"] = mid(df["High"], df["Low"], senkou_b).shift(kijun)
    df["ICHI_CHIKOU"] = df["Close"].shift(-kijun)
    return df


def add_mass_index(df: pd.DataFrame, fast=9, slow=25) -> pd.DataFrame:
    ema_fast = (df["High"] - df["Low"]).ewm(span=fast, adjust=False).mean()
    ema_slow = ema_fast.ewm(span=fast, adjust=False).mean()
    df["MASS_INDEX"] = (ema_fast / ema_slow.replace(0, np.nan)).rolling(slow).sum()
    return df


def add_coppock(df: pd.DataFrame, roc1=11, roc2=14, wma_period=10) -> pd.DataFrame:
    r1 = df["Close"].pct_change(roc1) * 100
    r2 = df["Close"].pct_change(roc2) * 100
    combined = r1 + r2
    weights = np.arange(1, wma_period + 1)
    df["COPPOCK"] = combined.rolling(wma_period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df


def calculate_short_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    空売り圧力（SELL_PRESSURE）とスクイーズスコア（SQUEEZE_SCORE）を計算する。
    Close・Volume だけを使うため calculate_all() の直後から呼べる軽量関数。

    SELL_PRESSURE: 大出来高を伴う下落の累積強度。0.65超で空売り圧力が高い。
    SQUEEZE_SCORE: 下落トレンド後の大出来高急騰強度。0.55超でスクイーズ兆候。
    どちらも 0〜1 に正規化。
    """
    df = df.copy()
    price_chg = df["Close"].pct_change().fillna(0)
    vol_avg = df["Volume"].rolling(20).mean()
    vol_ratio = (df["Volume"] / vol_avg.where(vol_avg > 0, np.nan)).fillna(1.0).clip(0, 5)

    # 空売り圧力：大出来高×下落幅の5日累積
    down_move = (-price_chg).clip(lower=0)
    sell_raw = (down_move * vol_ratio).rolling(5).sum().fillna(0)
    sp_norm = sell_raw.rolling(60).max().replace(0, np.nan).ffill().fillna(1)
    df["SELL_PRESSURE"] = (sell_raw / sp_norm).clip(0, 1)

    # スクイーズスコア：下落トレンド後の大出来高上昇（踏み上げ特徴を強調）
    up_move = price_chg.clip(lower=0)
    prior_drop = (-df["Close"].pct_change(10)).clip(lower=0).fillna(0)
    squeeze_raw = (up_move * vol_ratio * (1 + prior_drop * 5)).rolling(3).sum().fillna(0)
    sq_norm = squeeze_raw.rolling(60).max().replace(0, np.nan).ffill().fillna(1)
    df["SQUEEZE_SCORE"] = (squeeze_raw / sq_norm).clip(0, 1)

    return df


def calculate_extended(df: pd.DataFrame, ext_params: dict | None = None) -> pd.DataFrame:
    """全拡張指標を一括計算する。ext_params で各指標のパラメータを上書き可能。"""
    p = ext_params or {}
    df = df.copy()
    func_map = [
        (add_stochastic,          p.get("stochastic", {})),
        (add_cci,                 p.get("cci", {})),
        (add_williams_r,          p.get("williams_r", {})),
        (add_roc,                 p.get("roc", {})),
        (add_momentum,            p.get("momentum", {})),
        (add_adx,                 p.get("adx", {})),
        (add_atr,                 p.get("atr", {})),
        (add_obv,                 {}),
        (add_mfi,                 p.get("mfi", {})),
        (add_cmf,                 p.get("cmf", {})),
        (add_dema,                p.get("dema", {})),
        (add_tema,                p.get("tema", {})),
        (add_hma,                 p.get("hma", {})),
        (add_vwap,                {}),
        (add_aroon,               p.get("aroon", {})),
        (add_trix,                p.get("trix", {})),
        (add_dpo,                 p.get("dpo", {})),
        (add_force_index,         p.get("force_index", {})),
        (add_eom,                 p.get("eom", {})),
        (add_keltner,             p.get("keltner", {})),
        (add_donchian,            p.get("donchian", {})),
        (add_psar,                p.get("psar", {})),
        (add_ultimate_oscillator, p.get("ultimate_oscillator", {})),
        (add_cmo,                 p.get("cmo", {})),
        (add_ichimoku,            p.get("ichimoku", {})),
        (add_mass_index,          p.get("mass_index", {})),
        (add_coppock,             p.get("coppock", {})),
    ]
    for fn, kwargs in func_map:
        try:
            df = fn(df, **kwargs)
        except Exception:
            pass
    return df
