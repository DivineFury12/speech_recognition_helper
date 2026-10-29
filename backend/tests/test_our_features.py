"""Confirmations, past dates, undo, recurrence, conflicts and quick replies."""

import datetime
import unittest
import uuid

from src.ports.conversations import ConversationState
from src.schemas.assistant import (AssistantIntent, AssistantStatus, ConversationMessage, Entry, EntryKind,
                                   EntryStatus, Recurrence)
from src.schemas.assistant_command import AssistantCommand
from src.services import assistant as assistant_module
from src.services.assistant import AssistantService
from src.services.entries import EntryService
from src.services.guards import find_shift, yes_no
from src.services.recurrence import find_recurrence, occurrences
from src.user_context import acting_as
from tests.test_entry_flow import FakeExtractor, FakeRepository

NOW = datetime.datetime(2026, 10, 16, 12, 0)
TODAY = NOW.date()
TZ = 'Europe/Moscow'


class RecurringRepository(FakeRepository):
    def create(self, *, kind, title, date, time, timezone, embedding, embedding_model,
               duration_minutes=None, recurrence=None, **extra):
        entry = super().create(kind=kind, title=title, date=date, time=time, timezone=timezone,
                               embedding=embedding, embedding_model=embedding_model,
                               duration_minutes=duration_minutes, topics=extra.pop('topics', None))
        entry = entry.model_copy(update={'recurrence': recurrence, **extra})
        self.items[-1] = entry
        return entry

    def create_once(self, *, kind, title, date, time, timezone, embedding, embedding_model,
                    duration_minutes=None, recurrence=None, **extra):
        for entry in self.items:
            if (entry.kind == kind and entry.title.casefold() == title.casefold() and entry.date == date
                    and entry.time == time and entry.status == EntryStatus.pending):
                return entry, False
        return self.create(kind=kind, title=title, date=date, time=time, timezone=timezone,
                           embedding=embedding, embedding_model=embedding_model,
                           duration_minutes=duration_minutes, recurrence=recurrence, **extra), True

    def all_entries(self):
        return list(self.items)

    def list_undated(self):
        return [entry for entry in self.items if entry.date is None]

    def restore(self, entry):
        self.items.append(entry)
        return entry

    def list_period(self, date_from, date_to, *, kind=None, status=None):
        result = []
        for entry in self.items:
            if entry.date is None or (kind is not None and entry.kind != kind):
                continue
            last = min(date_to, entry.recurrence_until) if entry.recurrence_until else date_to
            for day in occurrences(entry.date, entry.recurrence, date_from, last):
                if day not in set(entry.exdates or []):
                    result.append(entry.model_copy(update={'date': day}))
        return result


class Conversations:
    """Keeps pending commands, messages and the undo record like the real repository."""

    def __init__(self):
        self.states = {}
        self.undo = {}
        self.created = {}

    def get(self, conversation_id):
        return self.states.get(conversation_id)

    def save_turn(self, turn, pending_command):
        previous = self.states.get(turn.conversation_id)
        messages = list(previous.messages) if previous else []
        stamp = datetime.datetime.now(datetime.timezone.utc)
        messages += [
            ConversationMessage(id=uuid.uuid4(), role='user', text=turn.user_text, created_at=stamp),
            ConversationMessage(id=uuid.uuid4(), role='assistant', text=turn.assistant_text, created_at=stamp),
        ]
        last = previous.last_entry_id if previous else None
        if turn.status == AssistantStatus.completed and len(turn.entries) == 1:
            last = turn.entries[0].id
        self.states[turn.conversation_id] = ConversationState(
            id=turn.conversation_id, pending_command=pending_command, messages=messages, last_entry_id=last)

    def is_foreign(self, conversation_id):
        return False

    def get_undo(self, conversation_id):
        return self.undo.get(conversation_id)

    def set_undo(self, conversation_id, record):
        self.undo[conversation_id] = record

    def created_entries(self, conversation_id):
        return list(self.created.get(conversation_id, []))

    def add_created(self, conversation_id, entry_id):
        self.created.setdefault(conversation_id, []).append(entry_id)


def command(**fields):
    return AssistantCommand.model_validate(fields)


class Chat:
    def __init__(self, commands, repository=None):
        self.repository = repository or RecurringRepository()
        self.entries = EntryService(self.repository)
        self.service = AssistantService(FakeExtractor(commands), self.entries, Conversations())
        self.cid = None

    def __call__(self, text):
        turn = self.service.handle_message(text, TZ, self.cid)
        self.cid = turn.conversation_id
        return turn


def setUpModule():
    global TOKEN
    TOKEN = assistant_module._now.set(NOW)


def tearDownModule():
    assistant_module._now.reset(TOKEN)


class YesNoTests(unittest.TestCase):
    def test_typos_and_layout(self):
        for text, expected in [('да', 'yes'), ('жа', 'yes'), ('lf', 'yes'), ('Да!', 'yes'), ('ага', 'yes'),
                               ('нет', 'no'), ('ytn', 'no'), ('неа', 'no'), ('давайй', 'yes'),
                               ('на', None), ('но', None), ('завтра', None), ('удали встречу', None)]:
            with self.subTest(text=text):
                self.assertEqual(yes_no(text), expected)

    def test_shift_words(self):
        self.assertEqual(find_shift('Перенеси встречу на час позже'), 60)
        self.assertEqual(find_shift('сдвинь её на 30 минут раньше'), -30)
        self.assertEqual(find_shift('перенеси на полчаса позже'), 30)
        self.assertEqual(find_shift('перенеси отчёт на 3 дня'), 3 * 1440)
        self.assertIsNone(find_shift('перенеси на пятницу'))
        self.assertIsNone(find_shift('встреча на час'))

    def test_recurrence_words(self):
        self.assertEqual(find_recurrence('Планёрка каждый понедельник в 10'), Recurrence.weekly)
        self.assertEqual(find_recurrence('зарядка по будням'), Recurrence.weekdays)
        self.assertEqual(find_recurrence('каждый день пить воду'), Recurrence.daily)
        self.assertIsNone(find_recurrence('встреча в понедельник'))


class PastDateTests(unittest.TestCase):
    def test_yesterday_is_resolved_and_confirmed(self):
        chat = Chat([command(action='create_entry', kind='task', title='купить хлеб')])
        asked = chat('Поставь задачу на вчера купить хлеб')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertIn('Вчера — это 15 октября, этот день уже прошёл', asked.assistant_text)
        self.assertEqual(asked.suggestions, ['Да', 'Нет'])
        self.assertEqual(chat.repository.items, [])
        done = chat('да')
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(done.entries[0].date, TODAY - datetime.timedelta(days=1))
        self.assertEqual(done.entries[0].title, 'Купить хлеб')

    def test_other_date_in_reply_replaces_past_one(self):
        chat = Chat([command(action='create_entry', kind='task', title='купить хлеб')])
        chat('Поставь задачу на позавчера купить хлеб')
        moved = chat('нет, на завтра')
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].date, TODAY + datetime.timedelta(days=1))

    def test_no_cancels(self):
        chat = Chat([command(action='create_entry', kind='task', title='купить хлеб')])
        chat('Поставь задачу на вчера купить хлеб')
        self.assertEqual(chat('нет').assistant_text, 'Хорошо, не ставлю.')
        self.assertEqual(chat.repository.items, [])

    def test_passed_time_today_is_confirmed(self):
        chat = Chat([command(action='create_entry', kind='reminder', title='позвонить маме',
                             date=TODAY.isoformat(), time='09:00', time_phrase='в 9:00')])
        asked = chat('Напомни сегодня в 9:00 позвонить маме')
        self.assertIn('Сегодня в 09:00 — это время уже прошло', asked.assistant_text)


class ConfirmTests(unittest.TestCase):
    def setUp(self):
        self.chat = Chat([command(action='delete_entry', kind='meeting', target_query='встреча с командой')] * 5)
        self.entry = self.chat.entries.create(kind=EntryKind.meeting, title='Встреча с командой',
                                              date=TODAY + datetime.timedelta(days=1), time='10:00', timezone=TZ)

    def test_delete_asks_and_accepts_typo(self):
        asked = self.chat('Удали встречу с командой')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertIn('Удалить встречу «Встреча с командой», завтра с 10:00 до 11:00?', asked.assistant_text)
        self.assertIsNotNone(self.chat.repository.get(self.entry.id))
        done = self.chat('lf')
        self.assertEqual(done.assistant_text, 'Удалил встречу «Встреча с командой».')
        self.assertIsNone(self.chat.repository.get(self.entry.id))

    def test_unclear_reply_keeps_question(self):
        self.chat('Удали встречу с командой')
        again = self.chat('ээ')
        self.assertEqual(again.status, AssistantStatus.needs_clarification)
        self.assertIn('Не понял ответ. Удалить встречу', again.assistant_text)
        self.assertIn('Ответьте «да» или «нет»', again.assistant_text)
        self.assertEqual(self.chat('да').status, AssistantStatus.completed)

    def test_bare_yes_without_question(self):
        self.assertIn('ничего не спрашивал', self.chat('да').assistant_text)


class UndoTests(unittest.TestCase):
    def test_undo_create_update_delete(self):
        tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
        chat = Chat([
            command(action='create_entry', kind='meeting', title='встреча с Иваном', date=tomorrow, time='15:00',
                    date_phrase='завтра', time_phrase='в 15:00'),
            command(action='create_entry', kind='meeting', title='встреча с Иваном', date=tomorrow, time='15:00',
                    date_phrase='завтра', time_phrase='в 15:00'),
            command(action='update_entry', time='16:00', time_phrase='на 16:00'),
            command(action='delete_entry', target_query='встреча с Иваном'),
        ])
        created = chat('Поставь встречу с Иваном завтра в 15:00')
        self.assertEqual(created.suggestions, ['Отмени последнее'])
        self.assertEqual(chat('Отмени последнее').status, AssistantStatus.needs_clarification)
        self.assertIn('Отменил', chat('да').assistant_text)
        self.assertEqual(chat.repository.items, [])
        chat('Поставь встречу с Иваном завтра в 15:00')
        moved = chat('Перенеси её на 16:00')
        self.assertEqual(moved.entries[0].time, '16:00')
        self.assertEqual(chat('Верни как было').status, AssistantStatus.needs_clarification)
        self.assertEqual(chat('да').entries[0].time, '15:00')
        chat('Удали встречу с Иваном')
        chat('да')
        self.assertEqual(chat.repository.items, [])
        self.assertEqual(chat('Отмени последнее').status, AssistantStatus.needs_clarification)
        back = chat('да')
        self.assertEqual(back.entries[0].title, 'Встреча с Иваном')
        self.assertEqual(chat('Отмени последнее').status, AssistantStatus.needs_clarification)
        self.assertIn('Отменил', chat('да').assistant_text)
        self.assertEqual(chat.repository.items, [])
        self.assertIn('нечего', chat('Отмени последнее').assistant_text)


class RecurrenceAndConflictTests(unittest.TestCase):
    def test_daily_reminder_is_bounded_by_current_week(self):
        chat = Chat([command(
            action='create_entry', kind='reminder', title='гулять с собакой',
            date_phrase='на этой неделе', date_relation='period',
            time='18:00', time_phrase='в шесть вечера',
        )])
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            created = chat('На этой неделе каждый день гулять с собакой в шесть вечера')
            self.assertEqual(created.status, AssistantStatus.completed)
            saved = created.entries[0]
            self.assertEqual(saved.date, datetime.date(2026, 10, 20))
            self.assertEqual(saved.recurrence, Recurrence.daily)
            self.assertEqual(saved.recurrence_until, datetime.date(2026, 10, 25))
            days = [entry.date for entry in chat.entries.list_period(
                datetime.date(2026, 10, 19), datetime.date(2026, 10, 27),
            )]
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(days, [datetime.date(2026, 10, day) for day in range(20, 26)])

    def test_model_can_recognize_bounded_repeat_without_literal_every_day(self):
        class WindowExtractor(FakeExtractor):
            def extract_recurrence_window(self, text, today):
                return Recurrence.daily, (
                    datetime.date(2026, 10, 19), datetime.date(2026, 10, 25),
                )

        chat = Chat([command(
            action='create_entry', kind='task', title='гулять с собакой',
            date_phrase='на этой неделе', date_relation='period',
            time='18:00', time_phrase='в шесть вечера',
        )])
        chat.service._extractor = WindowExtractor(chat.service._extractor.command)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            created = chat('Все дни на этой неделе буду гулять с собакой в шесть вечера')
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].recurrence, Recurrence.daily)
        self.assertEqual(created.entries[0].recurrence_until, datetime.date(2026, 10, 25))

    def test_passed_weekday_inside_bounded_week_does_not_shift_series(self):
        chat = Chat([command(
            action='create_entry', kind='task', title='планёрка',
            date_phrase='каждый понедельник', date_relation='exact',
        )])
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            result = chat('Планёрка каждый понедельник на этой неделе')
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(result.status, AssistantStatus.no_action)
        self.assertEqual(chat.repository.items, [])

    def test_bounded_repeat_checks_duplicates_on_later_days(self):
        chat = Chat([command(
            action='create_entry', kind='reminder', title='гулять с собакой',
            date_phrase='на этой неделе', date_relation='period',
            time='18:00', time_phrase='в 18:00',
        )])
        chat.entries.create(kind=EntryKind.reminder, title='Гулять с собакой',
                            date=datetime.date(2026, 10, 22), time='18:00', timezone=TZ)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            result = chat('Каждый день на этой неделе гулять с собакой в 18:00')
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(result.status, AssistantStatus.no_action)
        self.assertIn('уже есть такая запись', result.assistant_text)
        self.assertEqual(len(chat.repository.items), 1)

    def test_bounded_meeting_checks_conflict_on_later_days(self):
        chat = Chat([command(
            action='create_entry', kind='meeting', title='планёрка',
            date_phrase='на этой неделе', date_relation='period',
            time='18:00', time_phrase='в 18:00',
        )])
        chat.entries.create(kind=EntryKind.meeting, title='Другая встреча',
                            date=datetime.date(2026, 10, 22), time='18:00', timezone=TZ)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            result = chat('Каждый день на этой неделе планёрка в 18:00')
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(result.status, AssistantStatus.needs_clarification)
        self.assertIn('В один из дней повтора', result.assistant_text)
        self.assertEqual(len(chat.repository.items), 1)

    def test_single_meeting_this_week_does_not_become_daily(self):
        chat = Chat([command(
            action='create_entry', kind='meeting', title='встреча с командой',
            date_phrase='на этой неделе', date_relation='period',
            time='18:00', time_phrase='в шесть вечера',
        )])
        asked = chat('На этой неделе встреча с командой в шесть вечера')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertEqual(chat.repository.items, [])

    def test_clear_week_skips_only_occurrences_inside_week(self):
        chat = Chat([command(action='unknown')])
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            saved = chat.entries.create(
                kind=EntryKind.task, title='Выгулять собаку',
                date=datetime.date(2026, 10, 19), time='18:00', timezone=TZ,
                recurrence=Recurrence.daily,
                recurrence_until=datetime.date(2026, 11, 1),
            )
            preview = chat('очисти события этой недели')
            self.assertEqual(preview.status, AssistantStatus.needs_clarification)
            self.assertIn('Повторы за пределами периода останутся', preview.assistant_text)
            self.assertEqual(chat.repository.get(saved.id).exdates, None)
            confirmed = chat('подтверждаю')
            self.assertEqual(confirmed.status, AssistantStatus.completed)
            self.assertIsNotNone(chat.repository.get(saved.id))
            this_week = chat.entries.list_period(
                datetime.date(2026, 10, 19), datetime.date(2026, 10, 25),
            )
            next_week = chat.entries.list_period(
                datetime.date(2026, 10, 26), datetime.date(2026, 11, 1),
            )
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(this_week, [])
        self.assertEqual(len(next_week), 7)

    def test_clear_week_removes_series_if_all_its_occurrences_are_selected(self):
        chat = Chat([command(action='unknown')])
        token = assistant_module._now.set(datetime.datetime(2026, 10, 20, 12, 0))
        try:
            chat.entries.create(
                kind=EntryKind.task, title='Выгулять собаку',
                date=datetime.date(2026, 10, 20), time='18:00', timezone=TZ,
                recurrence=Recurrence.daily,
                recurrence_until=datetime.date(2026, 10, 25),
            )
            preview = chat('очисти события этой недели')
            self.assertEqual(preview.status, AssistantStatus.needs_clarification)
            self.assertEqual(len(chat.repository.items), 1)
            self.assertEqual(chat('подтверждаю').status, AssistantStatus.completed)
            self.assertEqual(chat.repository.items, [])
        finally:
            assistant_module._now.reset(token)

    def test_weekly_entry_repeats_and_completion_moves_forward(self):
        monday = TODAY + datetime.timedelta(days=(7 - TODAY.weekday()) % 7 or 7)
        chat = Chat([
            command(action='create_entry', kind='task', title='отчёт', date=monday.isoformat(),
                    date_phrase='каждый понедельник'),
            command(action='complete_entry', target_query='отчёт'),
        ])
        created = chat('Отчёт каждый понедельник')
        self.assertEqual(created.entries[0].recurrence, Recurrence.weekly)
        self.assertIn('каждую неделю', created.assistant_text)
        listed = chat.entries.list_period(TODAY, TODAY + datetime.timedelta(days=20))
        self.assertEqual(len(listed), 3)
        done = chat('Отметь отчёт выполненным')
        self.assertIn('Следующий раз', done.assistant_text)
        self.assertEqual(done.entries[0].date, monday + datetime.timedelta(days=7))

    def test_shift_beats_model_time(self):
        tomorrow = TODAY + datetime.timedelta(days=1)
        chat = Chat([
            command(action='create_entry', kind='meeting', title='встреча с Петром', date=tomorrow.isoformat(),
                    time='11:00', date_phrase='завтра', time_phrase='в 11:00'),
            command(action='update_entry', time='14:00', time_phrase='на 2 дня'),
        ])
        chat('Поставь встречу с Петром завтра в 11:00')
        moved = chat('Перенеси её на 2 дня')
        self.assertEqual(moved.entries[0].date, tomorrow + datetime.timedelta(days=2))
        self.assertEqual(moved.entries[0].time, '11:00')

    def test_meeting_conflict_is_confirmed(self):
        tomorrow = TODAY + datetime.timedelta(days=1)
        chat = Chat([command(action='create_entry', kind='meeting', title='созвон', date=tomorrow.isoformat(),
                             time='10:30', date_phrase='завтра', time_phrase='в 10:30')])
        chat.entries.create(kind=EntryKind.meeting, title='Планёрка', date=tomorrow, time='10:00', timezone=TZ)
        asked = chat('Поставь созвон завтра в 10:30')
        self.assertIn('В это время уже есть встреча «Планёрка»', asked.assistant_text)
        moved = chat('давай в 12')
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].time, '12:00')

    def test_done_report_completes_existing_task(self):
        chat = Chat([command(action='unknown')])
        task = chat.entries.create(kind=EntryKind.task, title='Купить молоко', date=None, time=None, timezone=TZ)
        turn = chat('Я купил молоко')
        self.assertEqual(turn.intent, AssistantIntent.complete_entry)
        self.assertEqual(chat.repository.get(task.id).status, EntryStatus.completed)


class UserIsolationTests(unittest.TestCase):
    def test_context_user_is_used_by_default(self):
        with acting_as('someone'):
            from src.user_context import current_user
            self.assertEqual(current_user(), 'someone')


class UnsureClassifierTests(unittest.TestCase):
    def test_ordinal_still_selects_when_model_is_unsure(self):
        from src.schemas.dialogue import PendingReply, ReplyAct

        class Unsure(FakeExtractor):
            def classify_pending_reply(self, text, **kwargs):
                return PendingReply(act=ReplyAct.unclear)

        repository = RecurringRepository()
        entries = EntryService(repository)
        tomorrow = TODAY + datetime.timedelta(days=1)
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Сергеем', date=tomorrow, time='10:00',
                               timezone=TZ)
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Сергеем', date=tomorrow, time='15:00',
                                timezone=TZ)
        service = AssistantService(Unsure(command(action='delete_entry', kind='meeting',
                                                  target_query='Встреча с Сергеем')), entries, Conversations())
        asked = service.handle_message('Удали встречу с Сергеем', TZ, None)
        self.assertEqual(asked.suggestions, ['Первую', 'Вторую', 'Отмена'])
        done = service.handle_message('вторую', TZ, asked.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertIsNotNone(repository.get(first.id))
        self.assertIsNone(repository.get(second.id))


class NewFeatureTests(unittest.TestCase):
    def test_title_keeps_numbers_dropped_by_model(self):
        chat = Chat([command(action='create_entry', kind='task', title='встреча с тестовым Иваном')])
        turn = chat('Создай задачу встреча с тестовым Иваном 159516')
        self.assertEqual(turn.entries[0].title, 'Встреча с тестовым Иваном 159516')

    def test_two_commands_in_one_phrase_and_lead_time(self):
        tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
        chat = Chat([command(action='create_entry', kind='meeting', title='встреча с Иваном', date=tomorrow,
                             time='10:00', date_phrase='завтра', time_phrase='в 10:00')])
        turn = chat('Поставь встречу с Иваном завтра в 10:00 и напомни за час')
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertIn('напомню за час', turn.assistant_text)
        self.assertEqual(chat.repository.items[0].remind_minutes, 60)

    def test_lead_time_in_create_and_for_existing_entry(self):
        tomorrow = TODAY + datetime.timedelta(days=1)
        chat = Chat([command(action='create_entry', kind='meeting', title='созвон', date=tomorrow.isoformat(),
                             time='10:00', date_phrase='завтра', time_phrase='в 10:00')])
        created = chat('Поставь созвон завтра в 10:00, напомни за 15 минут')
        self.assertEqual(created.entries[0].remind_minutes, 15)
        other = chat.entries.create(kind=EntryKind.meeting, title='Встреча с Петром', date=tomorrow, time='15:00',
                                    timezone=TZ)
        turn = chat('Напомни за полчаса до встречи с Петром')
        self.assertEqual(chat.repository.get(other.id).remind_minutes, 30)
        self.assertIn('за 30 минут', turn.assistant_text)

    def test_series_until_only_this_and_stop(self):
        monday = TODAY + datetime.timedelta(days=(7 - TODAY.weekday()) % 7 or 7)
        chat = Chat([
            command(action='create_entry', kind='meeting', title='планёрка', date=monday.isoformat(), time='10:00',
                    date_phrase='каждый понедельник', time_phrase='в 10:00'),
            command(action='delete_entry', target_query='планёрка'),
            command(action='update_entry', target_query='планёрка', time='12:00', time_phrase='на 12:00'),
        ])
        created = chat('Планёрка каждый понедельник в 10:00 до конца года')
        self.assertEqual(created.entries[0].recurrence_until, datetime.date(TODAY.year, 12, 31))
        asked = chat('Удали только ближайшую планёрку')
        self.assertIn('Остальные повторы останутся', asked.assistant_text)
        chat('да')
        days = [e.date for e in chat.entries.list_period(TODAY, TODAY + datetime.timedelta(days=20))]
        self.assertNotIn(monday, days)
        self.assertIn(monday + datetime.timedelta(days=7), days)
        moved = chat('Перенеси только эту планёрку на 12:00')
        self.assertIn('только этот раз', moved.assistant_text)
        self.assertEqual(moved.entries[0].time, '12:00')
        self.assertIsNone(moved.entries[0].recurrence)
        stopped = chat('Больше не повторяй планёрку')
        self.assertIn('больше не повторяется', stopped.assistant_text)


class OnlyThisSurvivesClarification(unittest.TestCase):
    def test_semantic_confirmation_keeps_only_this(self):
        monday = TODAY + datetime.timedelta(days=(7 - TODAY.weekday()) % 7 or 7)
        chat = Chat([command(action='delete_entry', target_query='ближайшую планёрку')] * 3)
        series = chat.entries.create(kind=EntryKind.meeting, title='Планёрка', date=monday, time='10:00',
                                     timezone=TZ, recurrence=Recurrence.weekly)
        first = chat('Удали только ближайшую планёрку')
        while first.status == AssistantStatus.needs_clarification:
            first = chat('да')
        self.assertIsNotNone(chat.repository.get(series.id))
        days = [e.date for e in chat.entries.list_period(TODAY, TODAY + datetime.timedelta(days=20))]
        self.assertNotIn(monday, days)
        self.assertIn(monday + datetime.timedelta(days=7), days)


class PriorityTests(unittest.TestCase):
    def test_important_on_create_and_mark_later(self):
        chat = Chat([command(action='create_entry', kind='task', title='важную задачу сдать отчёт')])
        created = chat('Поставь важную задачу сдать отчёт')
        self.assertEqual(created.entries[0].priority, 'high')
        self.assertIn('важное', created.assistant_text)
        other = chat.entries.create(kind=EntryKind.task, title='Купить хлеб', date=None, time=None, timezone=TZ)
        marked = chat('Пометь купить хлеб как важную')
        self.assertEqual(chat.repository.get(other.id).priority, 'high')
        self.assertIn('важное', marked.assistant_text)
        chat('Сними важность с купить хлеб')
        self.assertEqual(chat.repository.get(other.id).priority, 'normal')
