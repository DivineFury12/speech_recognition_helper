"""Answers that need no model: schedule questions, bulk shortcuts, follow-ups like «а Ивану?»."""

import datetime
import re
import uuid

from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, EntryKind, EntryStatus, EntryUpdate
from src.schemas.assistant_command import AssistantCommand
from src.services import phrasing
from src.services.recurrence import find_interval
from src.services.spoken import impossible_date
from src.services.temporal import parse_date, parse_period, parse_time

CREATE_WORDS_RE = re.compile(
    r'\b(?:поставь|напомни\w*|создай|добавь|запланируй|запиши|задач\w*|встреч\w*|созвон\w*|напоминани\w*|'
    r'перенеси|удали|отметь)\b')


class QuickAnswers:
    """Mixed into AssistantService; uses its entries, turn and create helpers."""

    def _quick_answers(self, conversation_id: uuid.UUID, text: str, state,
                       timezone: str = 'Europe/Moscow') -> AssistantTurn | None:
        """Questions people ask a scheduler that are not commands, and a few shortcuts."""
        today = self._today()
        value = text.casefold().replace('ё', 'е').strip()
        nothing = AssistantCommand(action=AssistantIntent.unknown)
        listing = AssistantCommand(action=AssistantIntent.list_entries)

        wrong = impossible_date(text, today.year)
        if wrong:
            return self._turn(conversation_id, text, nothing, f'{wrong}. Назовите другую дату — я ничего не создал.',
                              AssistantStatus.no_action, [])

        if re.search(r'\bчто\s+я\s+(?:тебе\s+)?(?:только\s+что\s+|сейчас\s+|до\s+этого\s+)?'
                     r'(?:сказал\w*|спрашивал\w*|просил\w*|писал\w*|говорил\w*)', value) and state is not None:
            said = [m.text for m in state.messages if m.role == 'user']
            answer = (f'Вы написали: «{said[-1]}».' if said else 'В этом диалоге вы ещё ничего не говорили.')
            return self._turn(conversation_id, text, nothing, answer, AssistantStatus.no_action, [])

        plain_list = re.fullmatch(
            r'\s*(?:а\s+)?(?:что|какие\s+(?:дела|планы)|покажи(?:\s+мне)?(?:\s+все)?)\s+(?:у\s+меня\s+)?'
            r'(?:(?P<kind>задачи|встречи|напоминания|дела|записи)\s+)?(?:есть\s+|запланировано\s+)?'
            r'(?P<period>(?:на|в|во|за)?\s*(?:сегодня|завтра|послезавтра|(?:эт\w+|следующ\w+)\s+недел\w+|неделю|'
            r'(?:эти\s+)?выходные|понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье))\s*[?.!]*\s*',
            value)
        if plain_list:
            # Common list questions are answered without the model: faster and the same every time.
            period = parse_period(plain_list.group('period'), today)
            if period is not None:
                list_kind = {'задачи': EntryKind.task, 'встречи': EntryKind.meeting,
                             'напоминания': EntryKind.reminder}.get(plain_list.group('kind') or '')
                command = AssistantCommand(action=AssistantIntent.list_entries, kind=list_kind,
                                           date_from=period[0], date_to=period[1])
                found = self._entries.list_period(period[0], period[1], kind=list_kind)
                return self._turn(conversation_id, text, command, self._list_answer(command, found, today),
                                  AssistantStatus.completed, found)

        step = find_interval(text)
        if step is not None and step[0] == 'hours':
            return self._turn(conversation_id, text, nothing,
                              'Повторять чаще раза в день я пока не умею. Могу поставить несколько напоминаний '
                              'на конкретное время, например: «Напомни сегодня в 12:00, 14:00 и 16:00 пить воду».',
                              AssistantStatus.no_action, [])

        labelled = re.match(r'^\s*(задача|напоминание|встреча)\s*[:—–-]\s*(?P<body>\S.*)$', text, re.IGNORECASE)
        if labelled:
            body = labelled.group('body').strip(' .!')
            title = body
            for phrase in filter(None, [parse_time(body) and re.search(
                    r'(?i)\b(?:в|на|к)\s+\d{1,2}(?::\d{2})?\b', body), re.search(
                    r'(?i)\b(?:сегодня|завтра|послезавтра)\b', body)]):
                title = title.replace(phrase.group(0), ' ')
            title = ' '.join(title.split()) or body
            kind = {'задача': EntryKind.task, 'напоминание': EntryKind.reminder,
                    'встреча': EntryKind.meeting}[labelled.group(1).casefold()]
            entry_date = parse_date(body, today)
            entry_time = parse_time(body)
            if kind != EntryKind.task and (entry_date is None or entry_time is None):
                entry_date = entry_date or (today if entry_time else None)
            if kind == EntryKind.task or (entry_date and entry_time):
                create = AssistantCommand(
                    action=AssistantIntent.create_entry, kind=kind, title=title[:1].upper() + title[1:],
                    date=entry_date, time=entry_time,
                )
                return self._create(conversation_id, text, create, timezone)

        if re.search(r'\b(?:погод\w*|курс\w*\s+(?:доллар|евро|валют|рубл)\w*|новост\w*|анекдот\w*|шутк\w*|'
                     r'рецепт\w*|столиц\w*|кто\s+(?:такой|такая|ты)|что\s+такое)\b', value) \
                and not CREATE_WORDS_RE.search(value):
            return self._turn(conversation_id, text, nothing,
                              'С этим я не помогу: я веду задачи, встречи и напоминания. '
                              'Например: «Что у меня завтра?» или «Напомни в 19:00 позвонить маме».',
                              AssistantStatus.no_action, [])

        kinds = {'встреч': EntryKind.meeting, 'задач': EntryKind.task, 'напомин': EntryKind.reminder}
        kind = next((k for stem, k in kinds.items() if stem in value), None)

        nxt = re.search(r'\bкогда\s+(?:у\s+меня\s+)?(?:следующ|ближайш)\w*\s+(встреч|задач|напомин)\w*', value)
        if nxt:
            upcoming = [e for e in self._entries.list_period(today, today + datetime.timedelta(days=365), kind=kind)
                        if e.status != EntryStatus.completed
                        and (e.date > today or not e.time or e.time >= self._now().strftime('%H:%M'))]
            upcoming.sort(key=lambda e: (e.date, e.time or ''))
            if not upcoming:
                return self._turn(conversation_id, text, listing, 'Ближайших записей такого типа нет.',
                                  AssistantStatus.completed, [])
            first = upcoming[0]
            return self._turn(conversation_id, text, listing,
                              f'Ближайшая — {phrasing.describe(first, today)}.', AssistantStatus.completed, [first])

        count = re.search(r'\bсколько\s+(?:у\s+меня\s+)?(встреч|задач|напоминани|запис|дел|событи)\w*', value)
        if count:
            period = parse_period(text, today) or (today, today)
            found = [e for e in self._entries.list_period(*period, kind=kind) if e.status != EntryStatus.completed]
            noun = {EntryKind.meeting: ('встреча', 'встречи', 'встреч'), EntryKind.task: ('задача', 'задачи', 'задач'),
                    EntryKind.reminder: ('напоминание', 'напоминания', 'напоминаний')}.get(
                kind, ('запись', 'записи', 'записей'))
            label = (f'на {phrasing.format_date(period[0], today)}' if period[0] == period[1]
                     else f'с {phrasing.format_day(period[0], today)} по {phrasing.format_day(period[1], today)}')
            if not found:
                return self._turn(conversation_id, text, listing, f'{label.capitalize()} {noun[2]} нет.',
                                  AssistantStatus.completed, [])
            lines = [phrasing.entry_line(e, today, show_date=period[0] != period[1]) for e in found]
            return self._turn(conversation_id, text, listing,
                              f'{label.capitalize()} {len(found)} {phrasing.plural(len(found), *noun)}:\n'
                              + '\n'.join(lines), AssistantStatus.completed, found)

        windows = re.search(r'\b(?:когда|во\s+сколько)\s+(?:я\s+|у\s+меня\s+)?(?:свобод\w*|есть\s+время)'
                            r'|\bсвободн\w*\s+(?:окн\w*|время|слот\w*)|\bчто\s+(?:у\s+меня\s+)?свободно\b', value)
        if windows and not parse_time(text):
            day = parse_date(text, today) or today
            start_of_day = max(datetime.time(9, 0), self._now().time().replace(second=0, microsecond=0)) \
                if day == today else datetime.time(9, 0)
            cursor = datetime.datetime.combine(day, start_of_day)
            if cursor.minute % 30:
                cursor += datetime.timedelta(minutes=30 - cursor.minute % 30)
            end = datetime.datetime.combine(day, datetime.time(21, 0))
            busy = sorted((e for e in self._entries.list_period(day, day)
                           if e.status != EntryStatus.completed and e.time),
                          key=lambda e: e.time)
            gaps = []
            for e in busy:
                e_start = datetime.datetime.combine(day, datetime.time.fromisoformat(e.time))
                e_end = e_start + datetime.timedelta(
                    minutes=e.duration_minutes or (60 if e.kind == EntryKind.meeting else 15))
                if e_start - cursor >= datetime.timedelta(minutes=30):
                    gaps.append((cursor, min(e_start, end)))
                cursor = max(cursor, e_end)
            if end - cursor >= datetime.timedelta(minutes=30):
                gaps.append((cursor, end))
            gaps = [(a, b) for a, b in gaps if a < end]
            label_day = phrasing.format_date(day, today)
            if not gaps:
                return self._turn(conversation_id, text, listing,
                                  f'{label_day.capitalize()} с 9 до 21 свободных окон от получаса нет.',
                                  AssistantStatus.completed, busy)
            lines = [f'• {a:%H:%M}–{b:%H:%M}' for a, b in gaps]
            return self._turn(conversation_id, text, listing,
                              f'{label_day.capitalize()} свободно (с 9 до 21):\n' + '\n'.join(lines),
                              AssistantStatus.completed, busy)

        free = re.search(r'\b(?:свободен|свободна|свободны|свободно|занят|занята|заняты|занято)\b', value)
        if free and not CREATE_WORDS_RE.search(value) and (
                '?' in text or re.match(r'^\s*(?:а\s+)?(?:я\s+)?(?:свобод|занят)', value)):
            day = parse_date(text, today) or today
            moment = parse_time(text)
            busy = [e for e in self._entries.list_period(day, day) if e.status != EntryStatus.completed and e.time]
            if moment:
                start = datetime.datetime.combine(day, datetime.time.fromisoformat(moment))
                clashes = []
                for e in busy:
                    e_start = datetime.datetime.combine(day, datetime.time.fromisoformat(e.time))
                    e_end = e_start + datetime.timedelta(minutes=e.duration_minutes or (60 if e.kind == EntryKind.meeting else 15))
                    if e_start <= start < e_end:
                        clashes.append(e)
                if clashes:
                    return self._turn(conversation_id, text, listing,
                                      f'Нет, в это время у вас {phrasing.describe(clashes[0], today)}.',
                                      AssistantStatus.completed, clashes)
                return self._turn(conversation_id, text, listing,
                                  f'Да, {phrasing.format_date(day, today)} в {moment} свободно.',
                                  AssistantStatus.completed, [])
            if not busy:
                return self._turn(conversation_id, text, listing,
                                  f'{phrasing.format_date(day, today).capitalize()} весь день свободен.',
                                  AssistantStatus.completed, [])
            lines = [phrasing.entry_line(e, today, show_date=False) for e in busy]
            return self._turn(conversation_id, text, listing,
                              f'{phrasing.format_date(day, today).capitalize()} заняты:\n' + '\n'.join(lines),
                              AssistantStatus.completed, busy)

        mark_all = re.match(r'^\s*(?:отметь|пометь)\s+(?:все|всё)\s+(?P<rest>.+?)\s+(?:как\s+)?(?:выполнен\w*|сделан\w*)',
                            value)
        if mark_all:
            period = parse_period(mark_all.group('rest'), today)
            pool = (self._entries.list_period(*period) if period else self._entries.all_entries())
            open_ones = [e for e in {e.id: e for e in pool}.values() if e.status != EntryStatus.completed]
            todo = [e for e in open_ones
                    if (e.kind == kind if kind is not None else e.kind != EntryKind.meeting)]
            skipped = sum(1 for e in open_ones if kind is None and e.kind == EntryKind.meeting)
            originals = [self._entries.get(e.id) or e for e in todo]
            for entry in todo:
                self._entries.update(entry.id, EntryUpdate(status=EntryStatus.completed))
            command = AssistantCommand(action=AssistantIntent.complete_entry)
            if not todo:
                return self._turn(conversation_id, text, command, 'Невыполненных записей не нашёл.',
                                  AssistantStatus.no_action, [])
            noun = phrasing.plural(len(todo), 'запись', 'записи', 'записей')
            note = (f' Встречи не трогал ({skipped}) — их отмечать не нужно.' if skipped else '')
            return self._turn(conversation_id, text, command,
                              f'Отметил выполненными {len(todo)} {noun}: '
                              + ', '.join(f'«{e.title}»' for e in todo[:6]) + ('…' if len(todo) > 6 else '') + '.'
                              + note,
                              AssistantStatus.completed, todo[:8], undo=self._batch_undo('update', originals))

        move_all = re.match(r'^\s*(?:перенеси|сдвинь|передвинь|перекинь)\s+(?:все|всё)\s+(?P<what>[^\d]*?)\s*'
                            r'(?:с|со)\s+(?P<src>.+?)\s+на\s+(?P<dst>.+?)[.!?]*\s*$', value)
        if move_all:
            source_day = parse_date(move_all.group('src'), today)
            target_day = parse_date(move_all.group('dst'), today)
            command = AssistantCommand(action=AssistantIntent.update_entry)
            if source_day is None or target_day is None:
                return self._turn(conversation_id, text, command,
                                  'Не понял, с какого дня и на какой переносить. Например: «Перенеси все встречи '
                                  'с завтра на понедельник».', AssistantStatus.no_action, [])
            found = [e for e in self._entries.list_period(source_day, source_day, kind=kind)
                     if e.status != EntryStatus.completed]
            single = [e for e in found if not e.recurrence]
            series = len(found) - len(single)
            if not single:
                return self._turn(conversation_id, text, command,
                                  f'{phrasing.format_date(source_day, today).capitalize()} переносить нечего.',
                                  AssistantStatus.no_action, [])
            originals = [self._entries.get(e.id) or e for e in single]
            moved = [self._entries.update(e.id, EntryUpdate(date=target_day)) for e in single]
            noun = phrasing.plural(len(moved), 'запись', 'записи', 'записей')
            note = (f' Повторяющиеся ({series}) оставил на месте — их переносите по одной.' if series else '')
            return self._turn(conversation_id, text, command,
                              f'Перенёс {len(moved)} {noun} на {phrasing.format_date(target_day, today)}: '
                              + ', '.join(f'«{e.title}»' for e in moved[:6]) + '.' + note,
                              AssistantStatus.completed, moved[:8], undo=self._batch_undo('update', originals))

        weekday = re.search(r'\b(?:в|во)\s+(понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)\b', value)
        relative = re.search(r'\b(сегодня|послезавтра|завтра)\b', value)
        if weekday and relative and not re.search(r'\b(?:следующ|эт[уото])\w*\s+' + weekday.group(1), value):
            names = ['понедельник', 'вторник', 'среду', 'четверг', 'пятницу', 'субботу', 'воскресенье']
            day = today + datetime.timedelta(days={'сегодня': 0, 'завтра': 1, 'послезавтра': 2}[relative.group(1)])
            if names[day.weekday()] != weekday.group(1):
                nominative = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
                return self._turn(conversation_id, text, nothing,
                                  f'{relative.group(1).capitalize()} — {nominative[day.weekday()]}, '
                                  f'{phrasing.format_day(day, today)}, а не {nominative[names.index(weekday.group(1))]}. '
                                  f'Скажите ещё раз, на какой день поставить.', AssistantStatus.no_action, [])

        rename = re.match(r'^\s*(?:переименуй|назови)\s+(?:ее|его|это|эту|этот)\s+(?:в|как)\s+(?P<title>.+?)[.!]*\s*$', value)
        if rename and state is not None and state.last_entry_id is not None:
            target = self._entries.get(state.last_entry_id)
            if target is not None:
                title = text.strip().rstrip('.!')[-len(rename.group('title')):]
                title = title[:1].upper() + title[1:]
                updated = self._entries.update(target.id, EntryUpdate(title=title))
                return self._turn(conversation_id, text, AssistantCommand(action=AssistantIntent.update_entry),
                                  f'Переименовал: «{target.title}» → «{updated.title}».',
                                  AssistantStatus.completed, [updated], undo=self._undo('update', target))

        follow = re.match(r'^\s*а\s+(?:теперь\s+)?(?:для\s+|с\s+|к\s+)?(?P<who>[А-ЯЁа-яё]+)\b(?P<rest>.*?)[?.!]*\s*$', text)
        if (follow and state is not None and not re.search(
                r'\b(?:удали|перенеси|сдвинь|отметь|покажи|найди|поставь|создай|напомни|что|когда|сколько)\b', value)):
            source = self._reference_entry(conversation_id, text, state)
            if source is not None and source.title:
                words = source.title.split()
                who = follow.group('who')
                names = 0
                while names < len(words) - 1 and words[-1 - names][:1].isupper():
                    names += 1
                if names and who[:1].isalpha():
                    title = ' '.join(words[:-names] + [who[:1].upper() + who[1:]])
                    copy = AssistantCommand(
                        action=AssistantIntent.create_entry, kind=source.kind, title=title,
                        date=parse_date(text, today) or source.date, time=parse_time(text) or source.time,
                        duration_minutes=source.duration_minutes if source.kind == EntryKind.meeting else None,
                    )
                    return self._create(conversation_id, text, copy, source.timezone)
        return None
