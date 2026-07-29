#!/usr/bin/env python3
"""
Compute six-dimension radar scores (图灵坐标体系) from AA + LMArena data.

输入：
  data/<date>/llms.json     — Artificial Analysis 榜单（fetch_leaderboards.py）
  data/<date>/lmarena.json  — LMArena Text Arena 分类榜（fetch_lmarena.py）

输出：
  site/data.json — 前端雷达图直接消费

六维体系（BV1VZ3C6kEeV 2:16）：
  科学推理   = HLE 1/3 + GPQA Diamond 1/3 + CritPt 1/3
  长文本推理 = AA-LCR
  指令遵循   = Text Arena IF, (Elo-1200)/500
  工具调用   = τ³-Banking 1/2 + GDPval-AA v2 1/2
  代码编程   = SciCode 1/3 + Terminal-Bench v2.1 1/3 + Text Arena Coding 1/3 (Elo-1200)/500
  事实可靠性 = AA-Omniscience Index, (v+100)/200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIMENSIONS = [
    {"key": "scientific_reasoning", "label": "科学推理", "formula": "HLE 1/3 + GPQA Diamond 1/3 + CritPt 1/3"},
    {"key": "long_context", "label": "长文本推理", "formula": "AA-LCR"},
    {"key": "instruction_following", "label": "指令遵循", "formula": "Text Arena IF · (Elo−1200)/500 · 主观双盲"},
    {"key": "tool_use", "label": "工具调用", "formula": "τ³-Banking 1/2 + GDPval-AA v2 1/2"},
    {"key": "coding", "label": "代码编程", "formula": "SciCode 1/3 + Terminal-Bench v2.1 1/3 + Text Arena Coding 1/3 · 主观双盲"},
    {"key": "factual_reliability", "label": "事实可靠性", "formula": "AA-Omniscience Index · (v+100)/200"},
]

FLAGSHIP_CREATORS = ["OpenAI", "Anthropic", "Google", "SpaceXAI", "DeepSeek", "Alibaba", "Kimi", "Z AI"]

ELO_BASE, ELO_DIV = 1200.0, 500.0

# 人工别名表：AA 归一化名 -> LMArena model_key/display_name 归一化名
# （自动匹配失败时的兜底，可随数据变化扩充）
MANUAL_ALIASES: dict[str, str] = {
    "qwen37max": "qwen37plus",
}

# AA 括号标注 -> LMArena 后缀的等价映射
PAREN_EQUIV = {
    "max": ["xhigh", "max"],
    "xhigh": ["xhigh", "max"],
    "high": ["high"],
    "medium": ["medium"],
    "low": ["low"],
    "minimal": ["minimal", "low"],
    "maxeffort": ["max"],
    "adaptivereasoningmaxeffort": ["max"],
    "reasoning": ["thinking"],
    "thinking": ["thinking"],
}


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def elo_norm(elo: float | None) -> float | None:
    if elo is None:
        return None
    return clamp01((elo - ELO_BASE) / ELO_DIV)


def omni_norm(v: float | None) -> float | None:
    if v is None:
        return None
    return clamp01((v + 100.0) / 200.0)


def wmean(pairs: list[tuple[float | None, float]]) -> tuple[float | None, bool]:
    """加权平均；跳过 None 并按可用项重新归一化权重。返回 (值, 是否缺项)"""
    vals = [(v, w) for v, w in pairs if v is not None]
    if not vals:
        return None, True
    partial = len(vals) < len(pairs)
    total_w = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / total_w, partial


class ArenaIndex:
    """LMArena 榜单索引：归一化名 -> entry"""

    def __init__(self, entries: list[dict[str, Any]]):
        self.by_key: dict[str, dict[str, Any]] = {}
        for e in entries:
            cands = {norm(e.get("model_display_name")), norm(e.get("model_key"))}
            # LMArena 键常带 -text 尾缀，额外索引去尾缀变体
            for c in list(cands):
                if c and c.endswith("text") and len(c) > 6:
                    cands.add(c[:-4])
            for cand in cands:
                if not cand:
                    continue
                old = self.by_key.get(cand)
                if old is None or (e.get("votes") or 0) > (old.get("votes") or 0):
                    self.by_key[cand] = e
        self.keys = list(self.by_key.keys())

    def lookup(self, aa_name: str) -> dict[str, Any] | None:
        cands = name_candidates(aa_name)
        for c in cands:
            if c in MANUAL_ALIASES and MANUAL_ALIASES[c] in self.by_key:
                return self.by_key[MANUAL_ALIASES[c]]
        for c in cands:
            if c in self.by_key:
                return self.by_key[c]
        # 子串匹配：要求覆盖较短串 >= 85%，取票数最高者
        best, best_score = None, 0.0
        for c in cands:
            if len(c) < 6:
                continue
            for k in self.keys:
                shorter, longer = (c, k) if len(c) <= len(k) else (k, c)
                if shorter in longer:
                    score = len(shorter) / len(longer)
                    if len(shorter) / max(len(shorter), 1) and shorter == longer[: len(shorter)]:
                        score += 0.05
                    if score >= 0.85:
                        votes = (self.by_key[k].get("votes") or 0) / 1e6
                        if score + votes > best_score:
                            best, best_score = self.by_key[k], score + votes
        return best


def name_candidates(name: str) -> list[str]:
    """为 AA 模型名生成归一化候选：全名、去括号、括号内容变体、thinking/reasoning 互换"""
    out = []
    base = re.sub(r"\([^)]*\)", "", name).strip()
    parens = re.findall(r"\(([^)]*)\)", name)
    full = norm(name)
    nbase = norm(base)
    out.append(full)
    if nbase and nbase != full:
        out.append(nbase)
    for p in parens:
        np_ = norm(p)
        if np_:
            out.append(nbase + np_)
            for equiv in PAREN_EQUIV.get(np_, []):
                out.append(nbase + equiv)
    # thinking/reasoning/high 等价变体
    for suffix in ("thinking", "reasoning"):
        if suffix in full:
            out.append(full.replace(suffix, "reasoning" if suffix == "thinking" else "thinking"))
    if "nonreasoning" in full:
        out.append(full.replace("nonreasoning", ""))
    # 去重保序
    seen, dedup = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def compute(aa_path: Path, lm_path: Path | None, out_path: Path) -> None:
    aa = json.loads(aa_path.read_text(encoding="utf-8"))
    lm = None
    lm_ok = {"instruction-following": False, "coding": False}
    if lm_path and lm_path.exists():
        lm = json.loads(lm_path.read_text(encoding="utf-8"))
        for cat in lm_ok:
            v = (lm.get("categories") or {}).get(cat)
            lm_ok[cat] = isinstance(v, list) and len(v) > 0

    if_idx = ArenaIndex(lm["categories"]["instruction-following"]) if lm_ok["instruction-following"] else None
    cod_idx = ArenaIndex(lm["categories"]["coding"]) if lm_ok["coding"] else None

    models_out = []
    match_stats = {"if": 0, "coding": 0}
    for m in aa["models"]:
        ev = m["evaluations"]
        name = m["name"]

        if_entry = if_idx.lookup(name) if if_idx else None
        cod_entry = cod_idx.lookup(name) if cod_idx else None
        if if_entry:
            match_stats["if"] += 1
        if cod_entry:
            match_stats["coding"] += 1

        subs = {
            "hle": ev.get("hle"),
            "gpqa": ev.get("gpqa"),
            "critpt": ev.get("critpt"),
            "aa_lcr": ev.get("aa_lcr"),
            "tau3_banking": ev.get("tau3_banking"),
            "gdpval_aa_v2": ev.get("gdpval_aa_normalized"),
            "scicode": ev.get("scicode"),
            "terminal_bench_v21": ev.get("terminal_bench_v21"),
            "aa_omniscience": ev.get("aa_omniscience"),
            "arena_if_elo": if_entry.get("elo") if if_entry else None,
            "arena_coding_elo": cod_entry.get("elo") if cod_entry else None,
        }

        dims: dict[str, float | None] = {}
        partial: dict[str, bool] = {}
        dims["scientific_reasoning"], partial["scientific_reasoning"] = wmean(
            [(ev.get("hle"), 1), (ev.get("gpqa"), 1), (ev.get("critpt"), 1)]
        )
        dims["long_context"], partial["long_context"] = wmean([(ev.get("aa_lcr"), 1)])
        dims["instruction_following"], partial["instruction_following"] = wmean(
            [(elo_norm(if_entry.get("elo") if if_entry else None), 1)]
        )
        dims["tool_use"], partial["tool_use"] = wmean(
            [(ev.get("tau3_banking"), 1), (ev.get("gdpval_aa_normalized"), 1)]
        )
        dims["coding"], partial["coding"] = wmean(
            [(ev.get("scicode"), 1), (ev.get("terminal_bench_v21"), 1),
             (elo_norm(cod_entry.get("elo") if cod_entry else None), 1)]
        )
        dims["factual_reliability"], partial["factual_reliability"] = wmean(
            [(omni_norm(ev.get("aa_omniscience")), 1)]
        )

        avail = sum(1 for v in dims.values() if v is not None)
        models_out.append({
            "id": m["id"],
            "name": name,
            "short_name": m.get("short_name"),
            "creator": m["creator"]["name"],
            "creator_color": m["creator"].get("color"),
            "country": m["creator"].get("country"),
            "release_date": m.get("release_date"),
            "is_reasoning": m.get("reasoning_model"),
            "open_weights": (m.get("open_weights") or {}).get("is_open_weights"),
            "intelligence_index": ev.get("artificial_analysis_intelligence_index"),
            "dims": dims,
            "dims_partial": partial,
            "dims_available": avail,
            "subs": subs,
        })

    # 厂商旗舰×8：每家取智力指数最高且六维齐全的模型
    defaults = []
    for creator in FLAGSHIP_CREATORS:
        cands = [m for m in models_out
                 if m["creator"] == creator and not m.get("deprecated")
                 and m["dims_available"] >= 5 and m["intelligence_index"] is not None]
        if cands:
            best = max(cands, key=lambda m: m["intelligence_index"])
            defaults.append(best["id"])

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aa_fetched_at": aa["meta"].get("fetched_at"),
            "lmarena_fetched_at": lm["meta"].get("fetched_at") if lm else None,
            "lmarena_ok": lm_ok,
            "model_count": len(models_out),
            "arena_match": match_stats,
            "attribution": "Data: Artificial Analysis (artificialanalysis.ai) & LMArena (lmarena.ai)",
            "methodology": "六维评估体系参考 B 站 UP 主图灵坐标 BV1VZ3C6kEeV",
        },
        "dimensions": DIMENSIONS,
        "defaults": defaults,
        "models": models_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"models={len(models_out)} defaults={len(defaults)} "
          f"arena_match={match_stats} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="data/<date>/ 目录；默认读 data/latest.json")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.date:
        date_str = args.date
    else:
        latest = json.loads((repo_root / "data" / "latest.json").read_text(encoding="utf-8"))
        date_str = latest["date"]

    aa_path = repo_root / "data" / date_str / "llms.json"
    lm_path = repo_root / "data" / date_str / "lmarena.json"
    if not aa_path.exists():
        print(f"ERROR: {aa_path} not found", file=sys.stderr)
        sys.exit(1)
    compute(aa_path, lm_path if lm_path.exists() else None, repo_root / args.out)


if __name__ == "__main__":
    main()
