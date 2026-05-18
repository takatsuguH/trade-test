"""EDINET API v2 クライアント — 日本企業の有価証券報告書を検索・取得する。"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

import requests
from dotenv import load_dotenv

load_dotenv()

EDINET_BASE = "https://disclosure.edinet-fsa.go.jp/api/v2"
_API_KEY = os.getenv("EDINET_API_KEY", "")

logger = logging.getLogger(__name__)

# 書類種別コード
DOC_TYPE_LABELS: dict[str, str] = {
    "120": "有価証券報告書（年次）",
    "130": "訂正有価証券報告書",
    "140": "四半期報告書",
    "150": "訂正四半期報告書",
    "160": "半期報告書",
    "180": "自己株券買付状況報告書",
}

_DEFAULT_DOC_TYPES = ("120", "140", "160")


def _headers() -> dict:
    return {"Subscription-Key": _API_KEY}


def check_api_connection() -> bool:
    """APIキーが有効か確認する（直近1日の書類一覧を試す）。"""
    if not _API_KEY:
        return False
    date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"{EDINET_BASE}/documents.json",
            params={"date": date_str, "type": 1},
            headers=_headers(),
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def get_document_list(date_str: str) -> list[dict]:
    """指定日の書類一覧を取得する（YYYY-MM-DD）。"""
    r = requests.get(
        f"{EDINET_BASE}/documents.json",
        params={"date": date_str, "type": 2},
        headers=_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def find_latest_filing(
    sec_code: str,
    doc_types: tuple[str, ...] = _DEFAULT_DOC_TYPES,
    max_days: int = 400,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Optional[dict]:
    """証券コードで最新の有価証券報告書を検索する。

    sec_code: 4桁コード（"7203"）または "7203.T" 形式でも可
    doc_types: 検索対象の書類種別コード
    progress_cb: (0.0〜1.0, メッセージ) を受け取るコールバック
    Returns: 書類メタデータのdict、見つからない場合は None
    """
    # 末尾の ".T" を除去して4桁を取り出す
    code4 = sec_code.upper().replace(".T", "").strip()[:4]
    # EDINET の secCode は5桁（末尾に "0" を付ける）
    target_sec = code4.ljust(5, "0")

    today = datetime.now()
    steps = range(0, max_days, 3)  # 3日おきに検索（精度とAPI負荷のバランス）
    total = len(steps)

    for i, days_back in enumerate(steps):
        if progress_cb:
            progress_cb(i / total, f"検索中… {days_back}日前")

        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            docs = get_document_list(date_str)
            time.sleep(0.15)
        except requests.HTTPError as e:
            logger.warning(f"EDINET list error on {date_str}: {e}")
            continue
        except Exception as e:
            logger.debug(f"EDINET request failed for {date_str}: {e}")
            continue

        for doc in docs:
            if doc.get("secCode") == target_sec and doc.get("docTypeCode") in doc_types:
                if progress_cb:
                    progress_cb(1.0, "完了")
                return doc

    if progress_cb:
        progress_cb(1.0, "見つかりませんでした")
    return None
