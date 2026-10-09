import fastapi

import src.adapters.transcriber
import src.schemas.transcribe
import src.services.transcriber

router = fastapi.APIRouter()


def get_transcription_service(request: fastapi.Request):
    return request.app.state.transcription_service


@router.get('/health')
def health(request: fastapi.Request) -> dict:
    return {
        'status': 'ok',
        **request.app.state.health_info,
    }


@router.post('/transcribe')
def transcribe(
    file: fastapi.UploadFile = fastapi.File(...),
    service: src.services.transcriber.TranscriptionService = (
        fastapi.Depends(get_transcription_service)
    ),
) -> src.schemas.transcribe.TranscribeResponse:
    text = service.transribe(file.file, file.filename)
    
    return src.schemas.transcribe.TranscribeResponse(text=text)


@router.post('/transcribe_longform')
def transcribe_longform(
    file: fastapi.UploadFile = fastapi.File(...),
    service: src.services.transcriber.TranscriptionService = (
        fastapi.Depends(get_transcription_service)
    ),
) -> src.schemas.transcribe.TranscribeLongformResponse:
    segments = service.transcribe_longform(
        file.file,
        file.filename,
    )
    
    return src.schemas.transcribe.TranscribeLongformResponse(
        segments=segments
    )
