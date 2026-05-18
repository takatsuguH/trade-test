"""SQLite-backed persistent storage for stocks and per-ticker indicator settings."""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path("trade_settings.db")

DEFAULT_SETTINGS: dict = {
    # メイン4指標
    "use_ma": True,   "ma_short": 5,     "ma_long": 25,
    "use_rsi": True,  "rsi_period": 14,  "rsi_ob": 70,  "rsi_os": 30,
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
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
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
        """)


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
