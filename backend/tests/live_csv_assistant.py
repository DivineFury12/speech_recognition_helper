"""Sample future CSV rows through the live text assistant and remove test records.

Usage: python tests/live_csv_assistant.py path/to/events.csv [base_url] [--bare] [csv_id ...]
The source has no entry kind or expected assistant answer. This check explicitly
asks for tasks, then compares the saved kind, title, date, and time to the CSV.
"""

import csv
import datetime
import http.cookiejar
import json
import sys
import urllib.error
import urllib.request
import uuid
from zoneinfo import ZoneInfo


TASK_VERBS = ('Проверить ', 'Подготовить ', 'Отправить ', 'Собрать ', 'Обновить ')
MONTHS = ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
          'августа', 'сентября', 'октября', 'ноября', 'декабря')


def main() -> int:
    source = sys.argv[1]
    base = (sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:8000').rstrip('/')
    bare = '--bare' in sys.argv[3:]
    requested = set(sys.argv[3:]) - {'--bare'}
    today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
    with open(source, encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.DictReader(file)
                if datetime.date.fromisoformat(row['date']) > today
                and row['text'].startswith(TASK_VERBS)]
    if requested:
        sample = [row for row in rows if row['id'] in requested]
        if len(sample) != len(requested):
            raise AssertionError('Some requested CSV IDs are unavailable')
    elif bare:
        sample = []
        for verb in ('Проверить ', 'Обновить '):
            matching = [row for row in rows if row['text'].startswith(verb)]
            if len(matching) < 10:
                raise AssertionError(f'Only {len(matching)} future rows for {verb!r}')
            sample.extend(matching[i * (len(matching) - 1) // 9] for i in range(10))
    else:
        if len(rows) < 30:
            raise AssertionError(f'Only {len(rows)} future task-like rows available')
        # Spread requests over the available dates, rather than taking one month.
        sample = [rows[i * (len(rows) - 1) // 29] for i in range(30)]
    opener = None
    created = []
    failures = []
    completed = 0

    def request(method, path, body=None, client=None):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        headers = {'X-Requested-With': 'assistant'}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with (client or opener).open(req, timeout=150) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw) if raw else None

    try:
        for row in sample:
            # Each row is an independent test; prior test records must not
            # change classification or duplicate detection for the next row.
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
            )
            day = datetime.date.fromisoformat(row['date'])
            phrase = (f'{row["text"]} {row["date"]} в {row["time"]}' if bare else
                      f'Создай задачу «{row["text"]}» на {day.day} '
                      f'{MONTHS[day.month - 1]} {day.year} года в {row["time"]}')
            status, turn = request('POST', '/v1/assistant/messages', {
                'text': phrase, 'timezone': 'Europe/Moscow', 'request_id': str(uuid.uuid4()),
            })
            entries = turn.get('entries', []) if isinstance(turn, dict) else []
            created.extend((opener, item['id']) for item in entries if 'id' in item)
            expected = ('task', row['text'], row['date'], row['time'])
            actual = [(item['kind'], item['title'], item['date'], item['time']) for item in entries]
            ok = (status == 200 and turn['status'] == 'completed' and
                  turn['intent'] == 'create_entry' and expected in actual)
            if ok:
                completed += 1
            else:
                failures.append({
                    'csv_id': row['id'], 'prompt': phrase, 'http': status,
                    'status': turn.get('status') if isinstance(turn, dict) else None,
                    'answer': turn.get('assistant_text') if isinstance(turn, dict) else str(turn),
                    'expected': expected, 'actual': actual,
                })
            print(f'{len(failures) + completed}/{len(sample)}: {"PASS" if ok else "FAIL"} '
                  f'csv_id={row["id"]}', flush=True)
    except Exception as exc:
        failures.append({'error': str(exc)})
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
    finally:
        cleanup_failures = []
        for client, entry_id in created:
            try:
                status, _ = request('DELETE', '/v1/entries/' + entry_id, client=client)
                if status != 204:
                    cleanup_failures.append(entry_id)
            except Exception:
                cleanup_failures.append(entry_id)
        print(json.dumps({
            'checked': completed + len(failures), 'passed': completed,
            'failures': failures, 'created': len(created),
            'cleanup_failures': cleanup_failures,
        }, ensure_ascii=False), flush=True)
    return 0 if not failures and not cleanup_failures else 1


if __name__ == '__main__':
    sys.exit(main())
