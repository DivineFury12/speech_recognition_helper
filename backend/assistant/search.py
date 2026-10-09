import re
from typing import Any

STOP_WORDS = {
    "про", "для", "что", "как", "где", "это", "была", "был", "были", "мне", "меня", "мой", "моя", "мои",
    "задача", "задачу", "встреча", "встречу", "запись", "напоминание", "найди", "поищи", "который", "которая",
}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))
    return {w[:5] if len(w) > 5 else w for w in words if len(w) >= 3 and w not in STOP_WORDS}


def score(query: str, title: str) -> float:
    q, t = _stems(query), _stems(title)
    if not q or not t:
        return 0.0
    hits = sum(1 for s in q if any(x.startswith(s) or s.startswith(x) for x in t))
    return hits / len(q)


def search_entries(query: str, entries: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ranked = [(score(query, e["title"]), e) for e in entries]
    ranked = [r for r in ranked if r[0] > 0]
    ranked.sort(key=lambda r: r[1]["created_at"], reverse=True)
    ranked.sort(key=lambda r: r[0], reverse=True)
    return [e for _, e in ranked[:limit]]
