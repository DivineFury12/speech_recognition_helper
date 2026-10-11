"""Read saved assistant messages."""

import uuid

import fastapi

from src.schemas.assistant import ConversationHistory


router = fastapi.APIRouter(prefix='/v1/conversations', tags=['Conversations'])


@router.get(
    '/{conversation_id}/messages',
    operation_id='listConversationMessages',
    response_model=ConversationHistory,
)
def list_conversation_messages(
    conversation_id: uuid.UUID, request: fastapi.Request,
) -> ConversationHistory:
    repository = request.app.state.conversations
    if repository is None:
        raise fastapi.HTTPException(status_code=501, detail='Хранение диалогов не подключено')
    messages = repository.list_messages(conversation_id)
    if messages is None:
        raise fastapi.HTTPException(status_code=404, detail='Диалог не найден')
    return ConversationHistory(conversation_id=conversation_id, messages=messages)
