"""SQLite-backed persistent storage for stocks and per-ticker indicator settings."""
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = Path("trade_settings.db")

DEFAULT_GLOBAL: dict = {
    "period": "1y",
}

DEFAULT_SETTINGS: dict = {
    # メイン4指標
    "use_ma": True,   "ma_short": 25,    "ma_long": 75,
    "use_rsi": True,  "rsi_period": 14,  "rsi_ob": 60,  "rsi_os": 35,
    "use_macd": True, "macd_fast": 12,   "macd_slow": 26, "macd_sig": 9,
    "use_bb": True,   "bb_period": 20,   "bb_std": 2.0,
    # 追加指標チェック状態
    "extra_checked": [],
    # Stochastic
    "stoch_k": 14, "stoch_d": 3,
    # CCI
    "cci_period": 20,
    # Williams %R
    "williams_period": 14,
    # ROC
    "roc_period": 10,
    # Momentum
    "mom_period": 10,
    # Ultimate Oscillator
    "uo_p1": 7, "uo_p2": 14, "uo_p3": 28,
    # CMO
    "cmo_period": 14,
    # ADX
    "adx_period": 14,
    # Aroon
    "aroon_period": 25,
    # TRIX
    "trix_period": 15,
    # DPO
    "dpo_period": 20,
    # Mass Index
    "mass_fast": 9, "mass_slow": 25,
    # Coppock
    "coppock_roc1": 11, "coppock_roc2": 14, "coppock_wma": 10,
    # DEMA
    "dema_period": 20,
    # TEMA
    "tema_period": 20,
    # HMA
    "hma_period": 20,
    # Parabolic SAR
    "psar_af": 0.02, "psar_step": 0.02, "psar_max": 0.2,
    # Ichimoku
    "ichi_tenkan": 9, "ichi_kijun": 26, "ichi_senkou_b": 52,
    # Keltner
    "keltner_period": 20, "keltner_mult": 2.0,
    # Donchian
    "donchian_period": 20,
    # MFI
    "mfi_period": 14,
    # CMF
    "cmf_period": 20,
    # Force Index
    "force_period": 13,
    # EOM
    "eom_period": 14,
    # ATR
    "atr_period": 14,
    # SELL_PRESSURE 閾値
    "sell_pressure_danger": 0.50,
    "sell_pressure_caution": 0.40,
    # SQUEEZE_SCORE 閾値
    "squeeze_high": 0.50,
    "squeeze_mid": 0.35,
    # リスク管理（銘柄別）
    "stop_loss": 5,
    "take_profit": 10,
    "max_pos": 100,
    "rebuy_dip": 0,
    # 投資設定（銘柄別）
    "initial_cash": 1_000_000,
    "max_shares": 0,
    # シグナル方式
    "use_context_strategy": False,
    "context_score_threshold": 5,
    # ファンダメンタル統合
    "fund_integrate": False,
    # 時間軸適合診断の表示
    "show_timeframe_diagnosis": True,
    # RSI閾値適合診断の表示
    "show_rsi_diagnosis": True,
    # RSI閾値診断で選択されたプリセット (ob/os)
    "rsi_diag_ob": 70,
    "rsi_diag_os": 30,
    # MACDパラメータ適合診断の表示
    "show_macd_diagnosis": True,
    # 診断適用前スナップショット（診断トグルOFFで元の値に戻す用）
    "snap_tf_ma_short": None, "snap_tf_ma_long": None,
    "snap_rsi_ob": None,      "snap_rsi_os": None,
    "snap_macd_fast": None,   "snap_macd_slow": None, "snap_macd_sig": None,
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS global_settings (
                id INTEGER PRIMARY KEY DEFAULT 1,
                settings_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS stocks (
                code TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS indicator_settings (
                stock_code TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (stock_code) REFERENCES stocks(code) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS fundamental_settings (
                stock_code TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS edinet_cache (
                sec_code TEXT PRIMARY KEY,
                doc_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS diagnosis_cache (
                stock_code  TEXT NOT NULL,
                diag_type   TEXT NOT NULL,
                data_period TEXT NOT NULL,
                result_json TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now','localtime')),
                PRIMARY KEY (stock_code, diag_type, data_period)
            );
            CREATE TABLE IF NOT EXISTS wf_diagnosis_cache (
                stock_code   TEXT NOT NULL,
                diag_type    TEXT NOT NULL,
                end_date     TEXT NOT NULL,
                params_hash  TEXT NOT NULL,
                algo_version TEXT NOT NULL,
                result_json  TEXT NOT NULL,
                is_effective INTEGER NOT NULL,
                updated_at   TEXT DEFAULT (datetime('now','localtime')),
                PRIMARY KEY (stock_code, diag_type, end_date, params_hash, algo_version)
            );
            CREATE TABLE IF NOT EXISTS wf_seven_way_cache (
                stock_code   TEXT NOT NULL,
                combo_key    TEXT NOT NULL,
                end_date     TEXT NOT NULL,
                params_hash  TEXT NOT NULL,
                algo_version TEXT NOT NULL,
                result_json  TEXT NOT NULL,
                updated_at   TEXT DEFAULT (datetime('now','localtime')),
                PRIMARY KEY (stock_code, combo_key, end_date, params_hash, algo_version)
            );
        """)


def load_global_settings() -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT settings_json FROM global_settings WHERE id = 1"
        ).fetchone()
        if row:
            return {**DEFAULT_GLOBAL, **json.loads(row["settings_json"])}
        return dict(DEFAULT_GLOBAL)


def save_global_settings(settings: dict) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO global_settings (id, settings_json, updated_at)
            VALUES (1, ?, datetime('now','localtime'))
        """, (json.dumps(settings, ensure_ascii=False),))


def load_stocks() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT code FROM stocks ORDER BY created_at").fetchall()
        return [row["code"] for row in rows]


def add_stock(code: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO stocks (code) VALUES (?)", (code,))


def remove_stock(code: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM stocks WHERE code = ?", (code,))


def load_settings(stock_code: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT settings_json FROM indicator_settings WHERE stock_code = ?",
            (stock_code,)
        ).fetchone()
        if row:
            return json.loads(row["settings_json"])
        return {}


def save_settings(stock_code: str, settings: dict) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO indicator_settings (stock_code, settings_json, updated_at)
            VALUES (?, ?, datetime('now','localtime'))
        """, (stock_code, json.dumps(settings, ensure_ascii=False)))


# ── ファンダメンタル設定 ──────────────────────────────────────────────────────

def load_fund_settings(stock_code: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT settings_json FROM fundamental_settings WHERE stock_code = ?",
            (stock_code,)
        ).fetchone()
        if row:
            return json.loads(row["settings_json"])
        return {}


def save_fund_settings(stock_code: str, settings: dict) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO fundamental_settings (stock_code, settings_json, updated_at)
            VALUES (?, ?, datetime('now','localtime'))
        """, (stock_code, json.dumps(settings, ensure_ascii=False)))


# ── EDINET検索キャッシュ ──────────────────────────────────────────────────────

def load_edinet_cache(sec_code: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT doc_json FROM edinet_cache WHERE sec_code = ?",
            (sec_code,)
        ).fetchone()
        if row:
            return json.loads(row["doc_json"])
        return None


def save_edinet_cache(sec_code: str, doc: dict) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO edinet_cache (sec_code, doc_json, updated_at)
            VALUES (?, ?, datetime('now','localtime'))
        """, (sec_code, json.dumps(doc, ensure_ascii=False)))


# ── 診断キャッシュ ──────────────────────────────────────────────────────────

def load_diagnosis_cache(stock_code: str, diag_type: str, data_period: str) -> dict | None:
    """当日更新済みの診断キャッシュを返す。なければ None。結果に _cached_at (HH:MM) を付与。"""
    with _conn() as conn:
        row = conn.execute(
            """SELECT result_json, updated_at FROM diagnosis_cache
               WHERE stock_code = ? AND diag_type = ? AND data_period = ?""",
            (stock_code, diag_type, data_period),
        ).fetchone()
    if row is None:
        return None
    if row["updated_at"][:10] != datetime.now().strftime("%Y-%m-%d"):
        return None
    result = json.loads(row["result_json"])
    result["_cached_at"] = row["updated_at"][11:16]  # "HH:MM"
    return result


def save_diagnosis_cache(stock_code: str, diag_type: str, data_period: str, result: dict) -> None:
    """診断結果をDBに保存する（_cached_at などのメタキーは除去して保存）。"""
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO diagnosis_cache
                (stock_code, diag_type, data_period, result_json, updated_at)
            VALUES (?, ?, ?, ?, datetime('now','localtime'))
        """, (stock_code, diag_type, data_period, json.dumps(clean, ensure_ascii=False)))


# ── ウォークフォワード診断キャッシュ ──────────────────────────────────────
# end_date 単位で診断結果を保存し、頻度違い (日次/週次/月次) で共有する。
# キャッシュキー: (stock_code, diag_type, end_date, params_hash, algo_version)

def compute_wf_params_hash(indicator_config: dict, initial_cash: float) -> str:
    """WFキャッシュキー用のパラメータハッシュ。
    indicator_config（ベースIC: ma/rsi/macd/bb の use_* と数値）+ initial_cash を
    正規化JSONにしてSHA-256。
    """
    payload = {
        "ic": {k: indicator_config[k] for k in sorted(indicator_config.keys())},
        "initial_cash": float(initial_cash),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_wf_cache_batch(
    stock_code: str, diag_type: str, end_dates: list[str],
    params_hash: str, algo_version: str,
) -> dict[str, dict]:
    """指定 end_date 一覧に対する既存キャッシュをまとめて取得。
    戻り値: {end_date: result_dict (is_effective を含む)}。ヒットしなかった日付はキーに含まれない。
    """
    if not end_dates:
        return {}
    placeholders = ",".join("?" * len(end_dates))
    sql = f"""
        SELECT end_date, result_json, is_effective FROM wf_diagnosis_cache
        WHERE stock_code = ? AND diag_type = ? AND params_hash = ? AND algo_version = ?
          AND end_date IN ({placeholders})
    """
    params = [stock_code, diag_type, params_hash, algo_version, *end_dates]
    out: dict[str, dict] = {}
    with _conn() as conn:
        for row in conn.execute(sql, params).fetchall():
            r = json.loads(row["result_json"])
            r["is_effective"] = bool(row["is_effective"])
            out[row["end_date"]] = r
    return out


def save_wf_cache(
    stock_code: str, diag_type: str, end_date: str,
    params_hash: str, algo_version: str,
    result: dict, is_effective: bool,
) -> None:
    """1件のWF診断結果を保存する（_* プレフィクスのメタキーは除去）。"""
    clean = {k: v for k, v in result.items() if not k.startswith("_") and k != "is_effective"}
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO wf_diagnosis_cache
                (stock_code, diag_type, end_date, params_hash, algo_version,
                 result_json, is_effective, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
        """, (
            stock_code, diag_type, end_date, params_hash, algo_version,
            json.dumps(clean, ensure_ascii=False),
            1 if is_effective else 0,
        ))


def count_wf_cache(stock_code: str, params_hash: str, algo_version: str) -> dict[str, int]:
    """銘柄ごとのキャッシュ件数を diag_type 別に返す（UI表示用）。"""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT diag_type, COUNT(*) AS n FROM wf_diagnosis_cache
            WHERE stock_code = ? AND params_hash = ? AND algo_version = ?
            GROUP BY diag_type
        """, (stock_code, params_hash, algo_version)).fetchall()
    return {row["diag_type"]: row["n"] for row in rows}


# ── 7通りディープ調査キャッシュ ────────────────────────────────────────
# サイドバー設定から完全独立。combo_key で7通りの組み合わせを区別。
# params_hash は initial_cash だけ（指標設定はグリッドサーチで動的に決まるため）。

def compute_seven_way_params_hash(initial_cash: float) -> str:
    """7通りWFキャッシュキー用のパラメータハッシュ。初期資金のみに依存。"""
    payload = {"initial_cash": float(initial_cash)}
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_seven_way_cache_batch(
    stock_code: str, combo_key: str, end_dates: list[str],
    params_hash: str, algo_version: str,
) -> dict[str, dict]:
    """指定 end_date 一覧に対する既存キャッシュをまとめて取得。
    戻り値: {end_date: result_dict}。ヒットしなかった日付はキーに含まれない。
    """
    if not end_dates:
        return {}
    placeholders = ",".join("?" * len(end_dates))
    sql = f"""
        SELECT end_date, result_json FROM wf_seven_way_cache
        WHERE stock_code = ? AND combo_key = ? AND params_hash = ? AND algo_version = ?
          AND end_date IN ({placeholders})
    """
    params = [stock_code, combo_key, params_hash, algo_version, *end_dates]
    out: dict[str, dict] = {}
    with _conn() as conn:
        for row in conn.execute(sql, params).fetchall():
            out[row["end_date"]] = json.loads(row["result_json"])
    return out


def save_seven_way_cache(
    stock_code: str, combo_key: str, end_date: str,
    params_hash: str, algo_version: str,
    result: dict,
) -> None:
    """1件の7通りWF診断結果を保存する（_* プレフィクスのメタキーは除去）。"""
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO wf_seven_way_cache
                (stock_code, combo_key, end_date, params_hash, algo_version,
                 result_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now','localtime'))
        """, (
            stock_code, combo_key, end_date, params_hash, algo_version,
            json.dumps(clean, ensure_ascii=False),
        ))


def count_seven_way_cache(stock_code: str, params_hash: str, algo_version: str) -> dict[str, int]:
    """銘柄ごとの7通りキャッシュ件数を combo_key 別に返す（UI表示用）。"""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT combo_key, COUNT(*) AS n FROM wf_seven_way_cache
            WHERE stock_code = ? AND params_hash = ? AND algo_version = ?
            GROUP BY combo_key
        """, (stock_code, params_hash, algo_version)).fetchall()
    return {row["combo_key"]: row["n"] for row in rows}
