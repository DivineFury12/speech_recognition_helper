import json
import logging
import re
from typing import Any

from .llm import LLMError, LLMProvider

log = logging.getLogger(__name__)

STOP_WORDS = {
    "про", "для", "что", "как", "где", "это", "эту", "этот", "была", "был", "были", "было", "будет",
    "мне", "меня", "мой", "моя", "мои", "моё", "мое", "когда", "когда-то", "ставил", "ставила", "создавал",
    "создавала", "задача", "задачу", "задачи", "встреча", "встречу", "встречи", "запись", "записи",
    "напоминание", "напоминания", "найди", "найти", "поищи", "ищи", "вспомни", "который", "которая",
    "которую", "есть", "там", "его", "её", "нее", "неё", "них", "кто", "какая", "какой", "какую",
}

MAX_SEMANTIC_CANDIDATES = 150
SEMANTIC_PROMPT = """Ты ищешь записи пользователя (задачи, напоминания, встречи) по смыслу запроса.
Тебе дан пронумерованный список записей и запрос. Верни ТОЛЬКО JSON вида {"matches": [номера]}:
номера записей, которые подходят по смыслу (синонимы, другие формулировки, родственные темы), лучшие первыми.
Если подходящих нет, верни {"matches": []}. Не придумывай номера, которых нет в списке."""


def _stem(word: str) -> str:
    return word[: max(4, len(word) - 2)]


def stems(text: str) -> set[str]:
    words = re.findall(r"[a-zа-я0-9]+(?:-[a-zа-я]+)?", text.lower().replace("ё", "е"))
    return {_stem(w) for w in words if len(w) >= 3 and w not in STOP_WORDS}


def score(query: str, title: str) -> float:
    q, t = stems(query), stems(title)
    if not q or not t:
        return 0.0
    hits = sum(1 for s in q if any(x.startswith(s) or s.startswith(x) for x in t))
    return hits / len(q)


def lexical(query: str, entries: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    ranked = [(score(query, e["title"]), e) for e in entries]
    ranked = [r for r in ranked if r[0] > 0]
    ranked.sort(key=lambda r: r[1]["created_at"], reverse=True)
    ranked.sort(key=lambda r: r[0], reverse=True)
    return ranked


def semantic(query: str, entries: list[dict[str, Any]], llm: LLMProvider) -> list[dict[str, Any]]:
    pool = entries[:MAX_SEMANTIC_CANDIDATES]
    if not pool:
        return []
    lines = [f"{i}. {e['title']} ({e['kind']}{', ' + e['date'] if e['date'] else ''})" for i, e in enumerate(pool, 1)]
    user = "Записи:\n" + "\n".join(lines) + f"\n\nЗапрос: «{query}»"
    try:
        raw = llm.complete(SEMANTIC_PROMPT, user)
        start, end = raw.find("{"), raw.rfind("}")
        matches = json.loads(raw[start: end + 1]).get("matches", []) if start != -1 else []
    except (LLMError, ValueError, json.JSONDecodeError, AttributeError) as e:
        log.warning("Семантический поиск недоступен, остаётся поиск по словам: %s", e)
        return []
    result, seen = [], set()
    for n in matches:
        if isinstance(n, int) and 1 <= n <= len(pool) and n not in seen:
            seen.add(n)
            result.append(pool[n - 1])
    return result


def search_entries(query: str, entries: list[dict[str, Any]], limit: int = 10,
                   llm: LLMProvider | None = None) -> list[dict[str, Any]]:
    ranked = lexical(query, entries)
    if ranked and ranked[0][0] >= 1.0 or llm is None or not stems(query):
        return [e for _, e in ranked[:limit]]
    found = semantic(query, entries, llm)
    ids = {e["id"] for e in found}
    found += [e for _, e in ranked if e["id"] not in ids]
    return found[:limit]
