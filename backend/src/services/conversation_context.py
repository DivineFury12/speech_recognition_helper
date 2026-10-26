"""Select bounded dialogue context while keeping the complete history in storage."""

import math
import re
from collections import Counter


RECENT_MESSAGES = 6
OLDER_PAIRS = 3
MAX_MESSAGES = 12
MAX_MESSAGE_CHARS = 1000
MAX_CONTEXT_CHARS = 7000


def _terms(text: str) -> set[str]:
    # Short prefixes make Russian inflections comparable without a phrase list.
    words = re.findall(r'[^\W_]+', text.casefold().replace('ё', 'е'))
    return {word[:5] if len(word) >= 5 else word for word in words if len(word) >= 4}


def _pairs(history: list[dict[str, str]]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for index, message in enumerate(history):
        if message['role'] == 'user' or not pairs:
            pairs.append([index])
        else:
            pairs[-1].append(index)
    return pairs


def select_conversation_context(
    history: list[dict[str, str]] | None, query: str,
) -> list[dict[str, str]]:
    """Keep recent turns and retrieve older turns by lexical topic overlap.

    Returns original chronological order. This is only model input selection;
    the conversation repository retains every message.
    """
    if not history:
        return []
    summary = next((message['text'][:1800] for message in history
                    if message['role'] == 'summary' and message['text']), None)
    history = [message for message in history if message['role'] != 'summary']
    if not history:
        return [{'role': 'summary', 'text': summary}] if summary else []
    pairs = _pairs(history)
    recent_count = min(len(pairs), RECENT_MESSAGES // 2)
    recent = pairs[-recent_count:]
    older = pairs[:-recent_count]
    query_terms = _terms(query)
    ranked: list[tuple[float, int]] = []
    if query_terms and older:
        documents = [_terms(' '.join(history[index]['text'] for index in pair))
                     for pair in older]
        frequency = Counter(term for document in documents for term in document)
        for number, document in enumerate(documents):
            overlap = query_terms & document
            if overlap:
                score = sum(math.log1p((len(older) + 1) / (frequency[term] + 1))
                            for term in overlap)
                ranked.append((score, number))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Reserve space for complete recent turns first, then add complete older
    # turns by relevance. Never leave an assistant answer without its question.
    preferred = list(reversed(recent)) + [older[number] for _, number in ranked[:OLDER_PAIRS]]
    selected: list[tuple[int, list[dict[str, str]]]] = []
    remaining = MAX_CONTEXT_CHARS - (len(summary) if summary else 0)
    count = 0
    for pair in preferred:
        excerpts = [{'role': history[index]['role'],
                     'text': history[index]['text'][:MAX_MESSAGE_CHARS]}
                    for index in pair]
        size = sum(len(message['text']) for message in excerpts)
        if size <= remaining and count + len(excerpts) <= MAX_MESSAGES:
            selected.append((pair[0], excerpts))
            remaining -= size
            count += len(excerpts)
    result = [message for _, pair in sorted(selected) for message in pair]
    return ([{'role': 'summary', 'text': summary}] if summary else []) + result
