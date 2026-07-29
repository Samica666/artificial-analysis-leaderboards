#!/usr/bin/env python3
"""
Fetch LMArena (lmarena.ai) Text Arena category leaderboards.

抓取 Text Arena 的 Instruction Following 与 Coding 分类榜单（主观双盲 Elo），
供图灵坐标六维雷达图体系使用。

数据源（SSR 内嵌于页面 RSC 载荷，无需 API Key）：
  - https://lmarena.ai/leaderboard/text/instruction-following
  - https://lmarena.ai/leaderboard/text/coding

Usage:
  python3 scripts/fetch_lmarena.py
  python3 scripts/fetch_lmarena.py --date 2026-07-30   # 指定输出目录日期
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
PARSER_VERSION = "lmarena-rsc-v1"

CATEGORIES = {
    "instruction-following": "https://lmarena.ai/leaderboard/text/instruction-following",
    "coding": "https://lmarena.ai/leaderboard/text/coding",
}

CHUNK_PATTERN = re.compile(r'self\.__next_f\.push\(\[1,("(?:\\.|[^"\\])*")\]\)')


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8", "ignore")


def extract_entries(html: str) -> list[dict[str, Any]]:
    """从页面 RSC 数据块中提取榜单 entries 数组"""
    chunks: list[str] = []
    for m in CHUNK_PATTERN.finditer(html):
        try:
            chunks.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    joined = "".join(chunks)

    decoder = json.JSONDecoder()
    pos = joined.find('"entries":[{')
    if pos == -1:
        raise RuntimeError("could not locate entries payload in page")
    arr, _ = decoder.raw_decode(joined[joined.find("[{", pos):])
    if not isinstance(arr, list) or not arr:
        raise RuntimeError("entries payload is empty")
    return arr


def normalize_entry(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": e.get("rank"),
        "model_key": e.get("modelKey"),
        "model_display_name": e.get("modelDisplayName"),
        "organization": e.get("modelOrganization"),
        "elo": e.get("rating"),
        "elo_upper": e.get("ratingUpper"),
        "elo_lower": e.get("ratingLower"),
        "votes": e.get("votes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch LMArena text arena category leaderboards")
    parser.add_argument("--date", help="Override output date directory (default: today, UTC)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between requests in seconds")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    now = datetime.now(timezone.utc)
    date_str = args.date or now.strftime("%Y-%m-%d")
    fetched_at = now.isoformat()

    day_dir = repo_root / "data" / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "meta": {
            "endpoint": "lmarena-text-arena",
            "source_type": "page_payload",
            "source_urls": CATEGORIES,
            "parser_version": PARSER_VERSION,
            "fetched_at": fetched_at,
            "source": "https://lmarena.ai",
        },
        "categories": {},
    }

    failures = []
    for i, (cat, url) in enumerate(CATEGORIES.items()):
        print(f"Fetching lmarena {cat}...", end=" ", flush=True)
        try:
            html = fetch_text(url)
            entries = [normalize_entry(e) for e in extract_entries(html)]
            output["categories"][cat] = entries
            print(f"✓ {len(entries)} entries")
        except Exception as e:  # noqa: BLE001
            print(f"✗ {e}", file=sys.stderr)
            output["categories"][cat] = {"error": str(e)}
            failures.append(cat)
        if i < len(CATEGORIES) - 1:
            time.sleep(args.delay)

    out_path = day_dir / "lmarena.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
