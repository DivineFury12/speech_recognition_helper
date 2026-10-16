"""Run against a configured local backend; creates and removes only QA records.

Usage: python tests/live_smoke.py [base_url]
"""

import datetime
import json
import http.cookiejar
import sys
import time
import urllib.error
import urllib.request
import uuid


BASE = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000').rstrip('/')
TIMEZONE = 'Europe/Moscow'
created_ids = set()
# One browser = one user: keep the session cookie and send the CSRF header.
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def request(method, path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={'X-Requested-With': 'assistant',
                 **({'Content-Type': 'application/json'} if data else {})},
    )
    try:
        with opener.open(req, timeout=150) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None


def message(text, conversation_id=None, request_id=None):
    payload = {
        'text': text, 'timezone': TIMEZONE,
        'conversation_id': conversation_id,
        'request_id': request_id or str(uuid.uuid4()),
    }
    for attempt in range(2):
        status, turn = request('POST', '/v1/assistant/messages', payload)
        if status != 502 or attempt:
            break
        time.sleep(1)
    if status != 200:
        raise AssertionError(f'{text!r}: HTTP {status}: {turn}')
    if (turn['intent'] == 'create_entry' and turn['status'] == 'completed'
            and turn['assistant_text'].startswith('Готово, создал')):
        created_ids.update(entry['id'] for entry in turn['entries'])
    print(json.dumps({
        'text': text, 'status': turn['status'], 'intent': turn['intent'],
        'answer': turn['assistant_text'],
        'entries': [(entry['title'], entry['date'], entry['time'])
                    for entry in turn['entries']],
    }, ensure_ascii=False), flush=True)
    return turn


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def main():
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).date()
    tag = str(uuid.uuid4().int % 1000000)
    reminder = f'Напомни проверить тестовый маяк {tag} через неделю в 6 вечера'
    reminder_id = str(uuid.uuid4())
    try:
        first = message(reminder, request_id=reminder_id)
        check(first['status'] == 'completed' and first['intent'] == 'create_entry',
              'Reminder creation failed')
        check(len(first['entries']) == 1, 'Expected one reminder')
        entry = first['entries'][0]
        check(entry['kind'] == 'reminder', 'Wrong entry kind')
        check(entry['date'] == (today + datetime.timedelta(days=7)).isoformat(),
              'Wrong date for «через неделю»')
        check(entry['time'] == '18:00', 'Wrong time for «в 6 вечера»')

        repeat = message(reminder, request_id=reminder_id)
        check(repeat['turn_id'] == first['turn_id'], 'Same request_id did not replay turn')
        check(repeat['entries'][0]['id'] == entry['id'], 'Retry created another entry')
        conflict_status, _ = request('POST', '/v1/assistant/messages', {
            'text': 'Что у меня завтра?', 'timezone': TIMEZONE,
            'request_id': reminder_id,
        })
        check(conflict_status == 409, 'Changed payload with same request_id was accepted')

        duplicate = message(reminder)
        check(duplicate['entries'][0]['id'] == entry['id'],
              'Identical entry was created twice')

        found = message(f'Найди напоминание про тестовый маяк {tag}',
                        first['conversation_id'])
        check(any(item['id'] == entry['id'] for item in found['entries']),
              'Existing reminder was not found')

        moved = message('А перенеси его на завтра в 5 вечера', first['conversation_id'])
        check(moved['status'] == 'completed', 'Pronoun move did not complete')
        check(moved['entries'][0]['id'] == entry['id'], 'Moved wrong reminder')
        check(moved['entries'][0]['date'] == (today + datetime.timedelta(days=1)).isoformat(),
              'Wrong moved date')
        check(moved['entries'][0]['time'] == '17:00', 'Wrong moved time')

        task = message(f'Создай задачу разобрать тестовые заметки {tag}')
        check(task['status'] == 'completed' and task['entries'][0]['date'] is None
              and task['entries'][0]['time'] is None,
              'Task received an invented date or time')

        meeting = message(f'Создай встречу с тестовым Иваном {tag}')
        check(meeting['status'] == 'needs_clarification',
              'Meeting without schedule should ask for details')
        meeting_reply = message('Завтра в 16:00', meeting['conversation_id'])
        check(meeting_reply['status'] == 'completed'
              and meeting_reply['entries'][0]['date'] == (today + datetime.timedelta(days=1)).isoformat()
              and meeting_reply['entries'][0]['time'] == '16:00',
              'Meeting clarification failed')

        meeting_title = f'встречу с тестовым Сергеем {tag}'
        first_meeting = message(f'Создай {meeting_title} завтра в 10:00')
        second_meeting = message(f'Создай {meeting_title} послезавтра в 11:00')
        check(first_meeting['entries'][0]['id'] != second_meeting['entries'][0]['id'],
              'Distinct meetings collapsed into one')
        # Keep both titles identical even when the external model paraphrases
        # one creation request; this step tests selection, not title wording.
        exact_title = f'QA встреча с Сергеем {tag}'
        for created in (first_meeting, second_meeting):
            status, _ = request('PATCH', f"/v1/entries/{created['entries'][0]['id']}",
                                {'title': exact_title})
            check(status == 200, 'Could not prepare ambiguous meeting titles')
        selection = message(f'Перенеси {exact_title}', second_meeting['conversation_id'])
        check(selection['status'] == 'needs_clarification'
              and len(selection['entries']) == 2,
              'Ambiguous meeting should offer two candidates')
        choice = message('вторую', selection['conversation_id'])
        check(choice['status'] == 'needs_clarification',
              'Selected meeting should ask for replacement schedule')
        changed = message('через неделю в 17:00', selection['conversation_id'])
        check(changed['status'] == 'completed'
              and changed['entries'][0]['date'] == (today + datetime.timedelta(days=7)).isoformat()
              and changed['entries'][0]['time'] == '17:00',
              'Ambiguous meeting selection or follow-up failed')

        pending = message(f'Создай напоминание проверить тестовый звонок {tag}')
        check(pending['status'] == 'needs_clarification',
              'Incomplete reminder should ask for schedule')
        new_request = message(f'Мне нужно купить тестовое молоко {tag}',
                              pending['conversation_id'])
        check(new_request['status'] == 'completed'
              and new_request['intent'] == 'create_entry'
              and new_request['entries'][0]['kind'] == 'task'
              and new_request['entries'][0]['date'] is None,
              'New command was mixed with pending reminder')
        print('LIVE_SMOKE_OK', flush=True)
    finally:
        for entry_id in created_ids:
            status, _ = request('DELETE', f'/v1/entries/{entry_id}')
            print(f'cleanup {entry_id}: {status}', flush=True)


if __name__ == '__main__':
    main()
