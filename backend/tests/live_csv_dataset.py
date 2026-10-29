"""Check a CSV dataset against the running entries API in an isolated session.

Usage: python tests/live_csv_dataset.py path/to/events.csv [base_url] [limit]
All imported entries are deleted in a finally block, including after a failed check.
The CSV has no kind column, so rows are imported as tasks for this API check.
"""

import collections
import csv
import datetime
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def main() -> int:
    if len(sys.argv) not in (2, 3, 4):
        print(__doc__.strip())
        return 2
    path = sys.argv[1]
    base = (sys.argv[2] if len(sys.argv) >= 3 else 'http://127.0.0.1:8000').rstrip('/')
    limit = int(sys.argv[3]) if len(sys.argv) == 4 else None
    with open(path, encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['id', 'date', 'time', 'text']:
            raise AssertionError(f'Unexpected CSV columns: {reader.fieldnames}')
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise AssertionError('CSV is empty')

    seen_ids = set()
    for number, row in enumerate(rows, 2):
        if not row['id'] or row['id'] in seen_ids or not row['text'].strip():
            raise AssertionError(f'Invalid id or title on CSV line {number}')
        seen_ids.add(row['id'])
        datetime.date.fromisoformat(row['date'])
        datetime.time.fromisoformat(row['time'])
        if len(row['time']) != 5 or len(row['text']) > 500:
            raise AssertionError(f'Invalid time or title length on CSV line {number}')

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    created = []
    failures = []
    start = time.monotonic()

    def request(method, endpoint, body=None):
        payload = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
        headers = {'X-Requested-With': 'assistant'}
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(base + endpoint, data=payload, method=method, headers=headers)
        try:
            with opener.open(req, timeout=120) as response:
                data = response.read()
                return response.status, json.loads(data) if data else None
        except urllib.error.HTTPError as exc:
            data = exc.read()
            return exc.code, json.loads(data) if data else None

    try:
        for index, row in enumerate(rows, 1):
            status, entry = request('POST', '/v1/entries', {
                'kind': 'task', 'title': row['text'], 'date': row['date'],
                'time': row['time'], 'timezone': 'Europe/Moscow',
            })
            if status != 201:
                raise AssertionError(f'CSV row {index}: create HTTP {status}: {entry}')
            created.append(entry['id'])
            if (entry['kind'], entry['title'], entry['date'], entry['time']) != (
                'task', row['text'], row['date'], row['time']
            ):
                failures.append(f'CSV row {index}: create response differs from input')
            if index % 100 == 0 or index == len(rows):
                print(f'created {index}/{len(rows)} in {time.monotonic() - start:.1f}s', flush=True)

        expected_by_month = collections.defaultdict(collections.Counter)
        for row in rows:
            expected_by_month[row['date'][:7]][row['date'], row['time'], row['text']] += 1
        listed = 0
        for month, expected in sorted(expected_by_month.items()):
            year, month_number = map(int, month.split('-'))
            first = datetime.date(year, month_number, 1)
            last = (datetime.date(year + 1, 1, 1) if month_number == 12 else
                    datetime.date(year, month_number + 1, 1)) - datetime.timedelta(days=1)
            query = urllib.parse.urlencode({
                'date_from': first.isoformat(), 'date_to': last.isoformat(), 'kind': 'task',
            })
            status, body = request('GET', '/v1/entries?' + query)
            if status != 200:
                failures.append(f'{month}: list HTTP {status}: {body}')
                continue
            actual = collections.Counter(
                (item['date'], item['time'], item['title']) for item in body['items']
            )
            listed += sum(actual.values())
            if actual != expected:
                failures.append(
                    f'{month}: expected {sum(expected.values())} rows, got {sum(actual.values())}; '
                    f'missing {sum((expected - actual).values())}, extra {sum((actual - expected).values())}'
                )

        unique_titles = sorted({row['text'] for row in rows})
        sample = unique_titles[::max(1, len(unique_titles) // 20)][:20]
        for title in sample:
            query = urllib.parse.urlencode({'q': title, 'kind': 'task'})
            status, body = request('GET', '/v1/entries/search?' + query)
            if status != 200 or not any(item['title'] == title for item in body.get('items', [])):
                failures.append(f'search failed for {title!r}: HTTP {status}')
        print(json.dumps({
            'rows': len(rows), 'months': len(expected_by_month), 'listed': listed,
            'search_checks': len(sample), 'failures': failures,
            'elapsed_seconds': round(time.monotonic() - start, 1),
        }, ensure_ascii=False), flush=True)
    except Exception as exc:
        failures.append(str(exc))
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
    finally:
        cleanup_failures = []
        for index, entry_id in enumerate(created, 1):
            try:
                status, _ = request('DELETE', '/v1/entries/' + entry_id)
                if status != 204:
                    cleanup_failures.append(entry_id)
            except Exception:
                cleanup_failures.append(entry_id)
            if index % 100 == 0 or index == len(created):
                print(f'cleaned {index}/{len(created)}', flush=True)
        if cleanup_failures:
            failures.append(f'Cleanup failed for {len(cleanup_failures)} ids: {cleanup_failures}')
        print(f'FINAL: {"PASS" if not failures else "FAIL"}; created={len(created)}; '
              f'cleanup_failures={len(cleanup_failures)}', flush=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
