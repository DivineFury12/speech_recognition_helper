"""Direct list and semantic search routes used by the frontend."""

import datetime

import fastapi

from src.schemas.assistant import EntryKind, EntryList, EntryStatus
from src.services.entries import EntryService


router = fastapi.APIRouter(prefix='/v1/entries', tags=['Entries'])


def get_entries_service(request: fastapi.Request) -> EntryService:
    service = request.app.state.entries_service
    if service is None:
        raise fastapi.HTTPException(status_code=503, detail='База данных не подключена')
    return service


@router.get('', operation_id='listEntries', response_model=EntryList)
def list_entries(
    request: fastapi.Request,
    date_from: datetime.date,
    date_to: datetime.date,
    kind: EntryKind | None = None,
    status: EntryStatus | None = None,
) -> EntryList:
    if date_from > date_to:
        raise fastapi.HTTPException(status_code=422, detail='date_from позже date_to')
    entries = get_entries_service(request).list_period(
        date_from, date_to, kind=kind, status=status,
    )
    return EntryList(items=entries)


@router.get('/search', operation_id='searchEntries', response_model=EntryList)
def search_entries(
    request: fastapi.Request,
    q: str = fastapi.Query(min_length=1),
    kind: EntryKind | None = None,
) -> EntryList:
    if not q.strip():
        raise fastapi.HTTPException(status_code=422, detail='Пустой поисковый запрос')
    return EntryList(items=get_entries_service(request).search(q, kind=kind))
