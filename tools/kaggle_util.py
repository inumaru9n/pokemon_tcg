"""Kaggle CLI呼び出しの共通ヘルパー（tools/get_notebooks.py / get_episodes.py / get_discussions.py が使用）。

方針: Kaggleからの情報取得はすべて公式CLI（subprocess）経由に統一する。
取得系ツールは共通の形を持つ:
  - `--min-votes N` / `--force` のargparse
  - `data/<source>/index.json` に全件メタデータを保存し、前回値との差分で新規/更新を判定
  - 取得はリトライ付き、レート制限対策のスリープ入り
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def kaggle_run(args: list[str]) -> None:
    """出力をそのまま流す実行（ダウンロード進捗表示用）。"""
    subprocess.run(["kaggle", *args], check=True)


def kaggle_text(args: list[str]) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, check=True)
    return r.stdout


def kaggle_json(args: list[str]):
    return json.loads(kaggle_text(args))


def load_index(path: Path, key: str, force: bool = False) -> dict:
    """index.jsonを {key値: item} のdictで返す。--force時は空扱い。"""
    if force or not path.exists():
        return {}
    return {item[key]: item for item in json.loads(path.read_text())}


def save_index(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1))


def fetch_all(changed: list[tuple[dict, str]], fetch_one, describe, tries: int = 3) -> int:
    """変更分を順に取得する共通ループ。fetch_one(item)を呼び、失敗はリトライ。

    Returns: 成功件数
    """
    n_ok = 0
    for item, status in changed:
        print(f"{status}: {describe(item)}")
        for attempt in range(tries):
            try:
                fetch_one(item)
                n_ok += 1
                break
            except Exception as e:  # noqa: BLE001 - CLI/API失敗はリトライ、ダメなら次へ
                detail = getattr(e, "stderr", "") or str(e)
                print(f"  fetch failed (try {attempt + 1}): {str(detail)[:150]}")
                time.sleep(2.0 * (attempt + 1))
        time.sleep(0.3)  # レート制限対策
    return n_ok
