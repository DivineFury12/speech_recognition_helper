"""Run a reviewable CSV sample through the live assistant, preserving all entries.

Usage: python tests/live_csv_review.py EVENTS.csv OUTPUT.json [BASE_URL] [COUNT]

The CSV has no entry kind, so each selected row is explicitly requested as a
task. Requests share a conversation to exercise dialogue context. The output
contains the exact assistant replies and entry cards for frontend review.
"""

import csv
import datetime as dt
import http.cookiejar
import json
import pathlib
import sys
import urllib.error
import urllib.request
import uuid


TASK_VERBS = ('Проверить ', 'Подготовить ', 'Отправить ', 'Собрать ', 'Обновить ')
MOSCOW = dt.timezone(dt.timedelta(hours=3))
MONTHS = ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
          'августа', 'сентября', 'октября', 'ноября', 'декабря')


def main() -> int:
    source = pathlib.Path(sys.argv[1])
    output = pathlib.Path(sys.argv[2])
    base = (sys.argv[3] if len(sys.argv) > 3 else 'http://127.0.0.1:8000').rstrip('/')
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    if count < 1:
        raise ValueError('COUNT must be positive')
    today = dt.datetime.now(MOSCOW).date()
    with source.open(encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.DictReader(file)
                if dt.date.fromisoformat(row['date']) > today
                and row['text'].startswith(TASK_VERBS)]
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} eligible future rows, requested {count}')
    sample = [rows[i * (len(rows) - 1) // (count - 1)] for i in range(count)] if count > 1 else [rows[0]]

    client = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(method: str, path: str, body=None):
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

    session_status, session = request('GET', '/v1/session')
    if session_status != 200 or not session.get('unlocked'):
        raise RuntimeError(f'Assistant session is unavailable: HTTP {session_status}')

    report = {
        'generated_at': dt.datetime.now(MOSCOW).isoformat(),
        'source_rows': len(rows),
        'sample_size': count,
        'user_id': session['user_id'],
        'conversation_id': None,
        'messages': [],
        'runs': [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_name(output.name + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(output)

    for position, row in enumerate(sample, 1):
        day = dt.date.fromisoformat(row['date'])
        prompt = (f'Создай задачу «{row["text"]}» на {day.day} '
                  f'{MONTHS[day.month - 1]} {day.year} года в {row["time"]}')
        report['messages'].append({'id': f'csv-{row["id"]}-user', 'role': 'user', 'text': prompt})
        try:
            status, turn = request('POST', '/v1/assistant/messages', {
                'text': prompt,
                'timezone': 'Europe/Moscow',
                'conversation_id': report['conversation_id'],
                'request_id': str(uuid.uuid4()),
            })
            if not isinstance(turn, dict):
                turn = {'detail': str(turn)}
            if status == 200:
                report['conversation_id'] = turn.get('conversation_id')
                replies = turn.get('responses') or [{
                    'text': turn.get('assistant_text', ''),
                    'status': turn.get('status'),
                    'entries': turn.get('entries', []),
                    'suggestions': turn.get('suggestions', []),
                }]
                for index, reply in enumerate(replies):
                    report['messages'].append({
                        'id': f'csv-{row["id"]}-assistant-{index}',
                        'role': 'assistant',
                        'text': reply.get('text', ''),
                        'status': reply.get('status'),
                        'entries': reply.get('entries', []),
                        'suggestions': reply.get('suggestions', []),
                    })
            else:
                report['messages'].append({
                    'id': f'csv-{row["id"]}-error', 'role': 'error',
                    'text': turn.get('detail', f'HTTP {status}'),
                })
            actual = [(entry['kind'], entry['title'], entry['date'], entry['time'])
                      for entry in turn.get('entries', [])]
            expected = ('task', row['text'], row['date'], row['time'])
            passed = (status == 200 and turn.get('status') == 'completed'
                      and turn.get('intent') == 'create_entry' and actual == [expected])
            report['runs'].append({
                'csv_id': row['id'], 'prompt': prompt, 'http_status': status,
                'assistant_status': turn.get('status'),
                'expected': expected, 'actual': actual, 'passed': passed,
            })
        except Exception as exc:
            report['messages'].append({
                'id': f'csv-{row["id"]}-error', 'role': 'error', 'text': str(exc),
            })
            report['runs'].append({
                'csv_id': row['id'], 'prompt': prompt, 'passed': False, 'error': str(exc),
            })
        save()
        result = report['runs'][-1]
        print(f'{position}/{count}: {"PASS" if result["passed"] else "FAIL"} '
              f'csv_id={row["id"]}', flush=True)

    passed_count = sum(run['passed'] for run in report['runs'])
    print(f'Completed {passed_count}/{count}. Preserved conversation '
          f'{report["conversation_id"]} and all created entries. Report: {output}', flush=True)
    return 0 if passed_count == count else 1


if __name__ == '__main__':
    sys.exit(main())
