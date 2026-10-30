"""Smoke-test a multi-turn dialogue against the running API in a fresh guest session.

Usage: python tests/live_unified_turn.py [base_url]
Any created record is deleted before exit.
"""

import http.cookiejar
import json
import sys
import urllib.error
import urllib.request
import uuid


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000').rstrip('/')
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    created = set()

    def request(method: str, path: str, body: dict | None = None):
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

    conversation_id = None
    outcomes = {}
    try:
        for label, prompt in (
            ('ask_title', 'Создай задачу'),
            ('supply_title', 'Купить хлеб'),
            ('move_reference', 'Перенеси её на завтра'),
            ('delete_reference', 'Удали её'),
            ('confirm_delete', 'да'),
            ('batch_create', 'Создай задачу проверить письма и создай задачу забрать посылку'),
        ):
            body = {'text': prompt, 'timezone': 'Europe/Moscow',
                    'request_id': str(uuid.uuid4())}
            if conversation_id is not None:
                body['conversation_id'] = conversation_id
            status, result = request('POST', '/v1/assistant/messages', body)
            if isinstance(result, dict):
                conversation_id = result.get('conversation_id', conversation_id)
                created.update(item['id'] for item in result.get('entries', []) if 'id' in item)
                outcomes[label] = (result.get('status'), len(result.get('entries', [])))
            print(json.dumps({
                'case': label, 'http': status,
                'status': result.get('status') if isinstance(result, dict) else None,
                'intent': result.get('intent') if isinstance(result, dict) else None,
                'answer': result.get('assistant_text') if isinstance(result, dict) else result,
                'entry_count': len(result.get('entries', [])) if isinstance(result, dict) else 0,
            }, ensure_ascii=False), flush=True)
            if status != 200:
                break
    finally:
        failed = []
        for entry_id in created:
            status, _ = request('DELETE', '/v1/entries/' + entry_id)
            if status not in (204, 404):
                failed.append(entry_id)
        print(json.dumps({'cleanup_failures': failed}), flush=True)
    expected = {
        'ask_title': ('needs_clarification', 0),
        'supply_title': ('completed', 1),
        'move_reference': ('completed', 1),
        'delete_reference': ('needs_clarification', 1),
        'confirm_delete': ('completed', 0),
        'batch_create': ('completed', 2),
    }
    if outcomes != expected or failed:
        raise AssertionError({'actual': outcomes, 'expected': expected,
                              'cleanup_failures': failed})


if __name__ == '__main__':
    main()
