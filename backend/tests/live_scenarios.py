"""Exercise natural multi-turn scenarios against the running assistant.

Usage: python -X utf8 tests/live_scenarios.py [base_url]
Each scenario uses a fresh guest session. Its created entries are removed at
the end; the separate CSV review data is never touched.
"""

import http.cookiejar
import json
import sys
import urllib.error
import urllib.request
import uuid


SCENARIOS = {
    'reminder_clarification': [
        'Напомни завтра погулять с собакой',
        'в 18:00',
    ],
    'meeting_context_move': [
        'Поставь встречу с командой на 2 ноября 2026 в 15:00',
        'А перенеси её на 3 ноября в 16:00',
    ],
    'mixed_batch': [
        'Создай задачу подготовить отчёт на 5 ноября 2026 и напоминание позвонить Ивану 6 ноября 2026 в 10:00',
    ],
    'spoken_hour': [
        'Поставь встречу с Иваном на 7 ноября 2026 в 7',
        'утра',
    ],
    'search_context': [
        'Создай задачу проверить сроки по хакатону на 30 октября 2026 в 12:30',
        'Какие задачи по хакатону у меня есть?',
    ],
    'undo_preview': [
        'Создай задачу подготовить доклад на 9 ноября 2026',
        'Отмени последнее действие',
        'да',
    ],
    'ambiguous_priority': [
        'Создай задачу подготовить отчёт по бюджету на 10 ноября 2026',
        'Создай задачу подготовить отчёт по проекту на 11 ноября 2026',
        'Сделай задачу про отчёт важной',
        'вторую',
    ],
}


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000').rstrip('/')
    selected = set(sys.argv[2:]) if len(sys.argv) > 2 else set(SCENARIOS)
    for name, prompts in SCENARIOS.items():
        if name not in selected:
            continue
        client = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        created = set()
        conversation_id = None

        def request(method, path, body=None):
            data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
            headers = {'X-Requested-With': 'assistant'}
            if data is not None:
                headers['Content-Type'] = 'application/json'
            req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
            try:
                with client.open(req, timeout=150) as response:
                    raw = response.read()
                    return response.status, json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                return exc.code, json.loads(raw) if raw else None

        request('GET', '/v1/session')
        try:
            for prompt in prompts:
                status, result = request('POST', '/v1/assistant/messages', {
                    'text': prompt, 'timezone': 'Europe/Moscow',
                    'conversation_id': conversation_id,
                    'request_id': str(uuid.uuid4()),
                })
                if not isinstance(result, dict):
                    result = {'detail': str(result)}
                conversation_id = result.get('conversation_id', conversation_id)
                entries_by_id = {entry['id']: entry for entry in result.get('entries') or []}
                for reply in result.get('responses') or []:
                    entries_by_id.update({entry['id']: entry for entry in reply.get('entries') or []})
                entries = list(entries_by_id.values())
                created.update(entry['id'] for entry in entries if 'id' in entry)
                print(json.dumps({
                    'scenario': name, 'prompt': prompt, 'http': status,
                    'status': result.get('status'), 'intent': result.get('intent'),
                    'answer': result.get('assistant_text', result.get('detail')),
                    'entries': [{key: entry.get(key) for key in
                                 ('id', 'kind', 'title', 'date', 'time')}
                                for entry in entries],
                }, ensure_ascii=False), flush=True)
        finally:
            failed = []
            for entry_id in created:
                try:
                    status, _ = request('DELETE', '/v1/entries/' + entry_id)
                    if status not in (204, 404):
                        failed.append(entry_id)
                except Exception:
                    failed.append(entry_id)
            print(json.dumps({'scenario': name, 'cleaned': len(created),
                              'cleanup_failures': failed}), flush=True)


if __name__ == '__main__':
    main()
