import uuid
import hashlib
import json

import fastapi

import src.ports.assistant
import src.schemas.assistant
import src.services.transcriber
from src.api.transcribe.router import get_transcription_service
from src.user_context import current_user


router = fastapi.APIRouter(prefix='/v1/assistant', tags=['Assistant'])


def get_assistant_service(request: fastapi.Request) -> src.ports.assistant.AssistantService:
    limiter = getattr(request.app.state, 'assistant_limiter', None)
    if limiter is not None and not limiter.allow(current_user()):
        raise fastapi.HTTPException(status_code=429, detail='Слишком много запросов, подождите минуту')
    service = request.app.state.assistant_service
    if service is None:
        raise fastapi.HTTPException(
            status_code=501,
            detail='Обработка команд ассистента пока не подключена',
        )
    return service


@router.post(
    '/messages',
    operation_id='sendTextMessage',
    response_model=src.schemas.assistant.AssistantTurn,
    responses={409: {'description': 'request_id использован с другим содержимым'},
               501: {'description': 'Обработка команд пока не подключена'}},
)
def send_text_message(
    message: src.schemas.assistant.TextMessageRequest,
    request: fastapi.Request,
) -> src.schemas.assistant.AssistantTurn:
    assistant = get_assistant_service(request)
    return assistant.handle_message(
        text=message.text,
        timezone=message.timezone,
        conversation_id=message.conversation_id,
        request_id=message.request_id or uuid.uuid4(),
    )


@router.post(
    '/audio',
    operation_id='sendAudioMessage',
    response_model=src.schemas.assistant.AssistantTurn,
    responses={409: {'description': 'request_id использован с другим содержимым'},
               501: {'description': 'Обработка команд пока не подключена'}},
)
def send_audio_message(
    request: fastapi.Request,
    file: fastapi.UploadFile = fastapi.File(...),
    timezone: str = fastapi.Form(..., min_length=1),
    conversation_id: uuid.UUID | None = fastapi.Form(None),
    request_id: uuid.UUID | None = fastapi.Form(None),
    transcription: src.services.transcriber.TranscriptionService = fastapi.Depends(
        get_transcription_service
    ),
) -> src.schemas.assistant.AssistantTurn:
    timezone = timezone.strip()
    if not timezone:
        raise fastapi.HTTPException(status_code=422, detail='Укажите часовой пояс')
    assistant = get_assistant_service(request)
    audio_hash = hashlib.file_digest(file.file, 'sha256').hexdigest()
    file.file.seek(0)
    fingerprint = hashlib.sha256(json.dumps({
        'source': 'audio', 'audio_hash': audio_hash, 'timezone': timezone,
        'conversation_id': str(conversation_id) if conversation_id else None,
    }, sort_keys=True).encode('utf-8')).hexdigest()
    text = transcription.transribe(file.file, file.filename).strip()
    if not text:
        raise fastapi.HTTPException(status_code=400, detail='Речь не распознана')
    return assistant.handle_message(
        text=text,
        timezone=timezone,
        conversation_id=conversation_id,
        request_id=request_id or uuid.uuid4(),
        request_fingerprint=fingerprint,
    )
