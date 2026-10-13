"""Server dialogue state and interpretation of a reply to a pending question."""

import enum

import pydantic


class PendingStep(str, enum.Enum):
    fields = 'fields'
    candidate = 'candidate'
    update_details = 'update_details'
    time_choice = 'time_choice'


class ReplyAct(str, enum.Enum):
    continue_command = 'continue_command'
    new_command = 'new_command'
    confirm = 'confirm'
    reject = 'reject'
    select = 'select'
    cancel = 'cancel'
    unclear = 'unclear'


class ScheduleRole(str, enum.Enum):
    none = 'none'
    existing = 'existing'
    replacement = 'replacement'


class DialogueTransition(str, enum.Enum):
    continue_command = 'continue_command'
    new_command = 'new_command'
    cancel = 'cancel'
    repeat_question = 'repeat_question'


def transition_for_reply(step: PendingStep, act: ReplyAct) -> DialogueTransition:
    if act == ReplyAct.cancel:
        return DialogueTransition.cancel
    if act == ReplyAct.new_command:
        return DialogueTransition.new_command
    if act == ReplyAct.unclear:
        return DialogueTransition.repeat_question
    if step in (PendingStep.fields, PendingStep.update_details):
        if act in (ReplyAct.confirm, ReplyAct.reject, ReplyAct.select):
            return DialogueTransition.repeat_question
    elif step == PendingStep.time_choice:
        if act in (ReplyAct.confirm, ReplyAct.reject):
            return DialogueTransition.repeat_question
    return DialogueTransition.continue_command


class PendingReply(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    act: ReplyAct
    choice: int | None = pydantic.Field(default=None, ge=1)
    schedule_role: ScheduleRole = ScheduleRole.none

    @pydantic.model_validator(mode='after')
    def validate_choice(self) -> 'PendingReply':
        if (self.act == ReplyAct.select) != (self.choice is not None):
            raise ValueError('choice is required exactly for select')
        if self.act != ReplyAct.select and self.schedule_role != ScheduleRole.none:
            raise ValueError('schedule_role is meaningful only for select')
        return self

    @classmethod
    def provider_schema(cls) -> dict:
        schema = cls.model_json_schema()
        schema['required'] = list(schema['properties'])
        return schema
