"""Reproduce the concrete prompts from test_evaluation_report.pdf on the live API.

Creates a fresh guest session and deletes every record returned by the assistant.
Usage: python tests/live_report_cases.py [base_url] [case_id ...]
"""

import http.cookiejar
import json
import sys
import urllib.error
import urllib.request
import uuid


CASES = [
    ('BUG-01', 'Подготовить техническое задание 2024-01-07 в 14:45'),
    ('BUG-02', 'Проверить сроки по хакатону 2026-10-30 в 12:30'),
    ('BUG-03', 'Обновить материалы по аналитике 2024-02-18 в 18:15'),
    ('BUG-04a', 'Подготовить отчёт'),
    ('BUG-04b', 'Собрать обратную связь по аналитике'),
    ('BUG-04c', 'Позвонить Ивану'),
    ('BUG-04d', 'Передать акт на согласование'),
    ('BUG-05a', 'Созвон с Маяком по бюджету 2027-01-05 в 08:00'),
    ('BUG-05b', 'Напомни 9 января 2027 в 20:15 про презентации Сергею'),
    ('BUG-06', 'Напомнить Дмитрием про мобильному приложению 2027-02-16 в 12:15'),
    ('BUG-07a', 'Подготовить протокол к 19:30'),
    ('BUG-07b', 'Отправить отчёт в 11:15'),
    ('CONTROL-search', 'Проверь, какие задачи по хакатону у меня есть'),
    ('CONTROL-update', 'Обнови дату уже созданной задачи про отчёт на завтра'),
    ('CONTROL-status', 'Проверь статус моей задачи про договор'),
    ('CONTROL-new', 'Обновить материалы по проекту'),
]


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000').rstrip('/')
    selected = set(sys.argv[2:]) if len(sys.argv) > 2 else None
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    created = set()

    def request(method, path, body=None):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        headers = {'X-Requested-With': 'assistant'}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with opener.open(req, timeout=150) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw) if raw else None

    try:
        for case, prompt in CASES:
            if selected is not None and case not in selected:
                continue
            status, result = request('POST', '/v1/assistant/messages', {
                'text': prompt, 'timezone': 'Europe/Moscow',
                'request_id': str(uuid.uuid4()),
            })
            entries = result.get('entries', []) if isinstance(result, dict) else []
            created.update(item['id'] for item in entries if 'id' in item)
            print(json.dumps({
                'case': case, 'prompt': prompt, 'http': status,
                'status': result.get('status') if isinstance(result, dict) else None,
                'intent': result.get('intent') if isinstance(result, dict) else None,
                'answer': result.get('assistant_text') if isinstance(result, dict) else str(result),
                'entries': [{k: item.get(k) for k in ('id', 'kind', 'title', 'date', 'time')}
                            for item in entries],
            }, ensure_ascii=False), flush=True)
    finally:
        failed = []
        for entry_id in created:
            try:
                status, _ = request('DELETE', '/v1/entries/' + entry_id)
                if status != 204:
                    failed.append(entry_id)
            except Exception:
                failed.append(entry_id)
        print(json.dumps({'created': len(created), 'cleanup_failures': failed}), flush=True)


if __name__ == '__main__':
    main()
