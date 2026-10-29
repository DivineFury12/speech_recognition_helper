"""«Отмени последнее»: a stack of the last changes in a conversation, batches included."""

import uuid

from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, Entry, EntryUpdate
from src.schemas.assistant_command import AssistantCommand
from src.services import phrasing

UNDO_FIELDS = (
    'title', 'date', 'time', 'duration_minutes', 'recurrence',
    'recurrence_interval', 'recurrence_until', 'exdates', 'remind_minutes',
    'priority', 'status',
)


class UndoStack:
    """Mixed into AssistantService; keeps undo records through the conversation repository."""

    @staticmethod
    def _undo(op: str, entry: Entry) -> dict:
        return {'op': op, 'entry': entry.model_dump(mode='json')}

    @classmethod
    def _batch_undo(cls, op: str, entries: list[Entry]) -> dict:
        return {'op': 'batch', 'items': [cls._undo(op, entry) for entry in entries]}

    def _undo_stack(self, conversation_id: uuid.UUID) -> list[dict]:
        record = (self._conversations.get_undo(conversation_id)
                  if hasattr(self._conversations, 'get_undo') else None)
        if not record:
            return []
        return list(record['stack']) if 'stack' in record else [record]

    def _undo_last(self, conversation_id: uuid.UUID, text: str) -> AssistantTurn:
        stack = self._undo_stack(conversation_id)
        record = stack[-1] if stack else None

        def consume() -> None:
            remaining = stack[:-1]
            self._conversations.set_undo(
                conversation_id, {'stack': remaining} if remaining else None,
            )
        nothing = AssistantCommand(action=AssistantIntent.unknown)
        if not record or self._entries is None:
            return self._turn(conversation_id, text, nothing,
                              'Отменять нечего: в этом диалоге я ещё ничего не менял.', AssistantStatus.no_action, [])
        today = self._today()
        if record['op'] == 'batch':
            restored = []
            for item in reversed(record['items']):
                saved = Entry.model_validate(item['entry'])
                if item['op'] == 'create':
                    self._entries.delete(saved.id)
                elif item['op'] == 'delete':
                    if self._entries.get(saved.id) is None:
                        restored.append(self._entries.restore(saved))
                elif self._entries.get(saved.id) is not None:
                    fields = {name: getattr(saved, name) for name in UNDO_FIELDS}
                    fields = {k: v for k, v in fields.items() if v is not None or k in ('date', 'time', 'recurrence')}
                    restored.append(self._entries.update(saved.id, EntryUpdate(**fields)))
            noun = phrasing.plural(len(record['items']), 'запись', 'записи', 'записей')
            consume()
            return self._turn(conversation_id, text, AssistantCommand(action=AssistantIntent.update_entry),
                              f'Отменил: вернул как было {len(record["items"])} {noun}.',
                              AssistantStatus.completed, restored[:8], undo={})
        saved = Entry.model_validate(record['entry'])
        if record['op'] == 'create':
            gone = self._entries.delete(saved.id)
            command = AssistantCommand(action=AssistantIntent.delete_entry)
            answer = (f'Отменил: убрал {phrasing.KIND_ACC[saved.kind]} «{saved.title}».' if gone
                      else 'Этой записи уже нет, отменять нечего.')
            consume()
            return self._turn(conversation_id, text, command, answer,
                              AssistantStatus.completed if gone else AssistantStatus.no_action, [], undo={})
        if record['op'] == 'delete':
            restored = self._entries.restore(saved) if self._entries.get(saved.id) is None else saved
            command = AssistantCommand(action=AssistantIntent.create_entry)
            consume()
            return self._turn(conversation_id, text, command,
                              f'Вернул {phrasing.describe(restored, today, accusative=True)}.',
                              AssistantStatus.completed, [restored], undo={})
        if self._entries.get(saved.id) is None:
            return self._turn(conversation_id, text, nothing, 'Этой записи уже нет, отменять нечего.',
                              AssistantStatus.no_action, [], undo={})
        fields = {name: getattr(saved, name) for name in UNDO_FIELDS}
        fields = {k: v for k, v in fields.items() if v is not None or k in ('date', 'time', 'recurrence')}
        try:
            restored = self._entries.update(saved.id, EntryUpdate(**fields))
        except ValueError as exc:
            return self._turn(conversation_id, text, nothing, f'Не получилось вернуть: {exc}',
                              AssistantStatus.no_action, [])
        command = AssistantCommand(action=AssistantIntent.update_entry)
        consume()
        return self._turn(conversation_id, text, command,
                          f'Вернул как было: {phrasing.describe(restored, today)}.',
                          AssistantStatus.completed, [restored], undo={})
