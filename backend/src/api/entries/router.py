"""Direct list and semantic search routes used by the frontend."""

import datetime
import uuid

import fastapi

from src.schemas.assistant import Entry, EntryKind, EntryList, EntryStatus, EntryUpdate
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


@router.get('/{entry_id}', operation_id='getEntry', response_model=Entry)
def get_entry(entry_id: uuid.UUID, request: fastapi.Request) -> Entry:
    entry = get_entries_service(request).get(entry_id)
    if entry is None:
        raise fastapi.HTTPException(status_code=404, detail='Запись не найдена')
    return entry


@router.patch('/{entry_id}', operation_id='updateEntry', response_model=Entry)
def update_entry(
    entry_id: uuid.UUID, patch: EntryUpdate, request: fastapi.Request,
) -> Entry:
    try:
        entry = get_entries_service(request).update(entry_id, patch)
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=422, detail=str(exc)) from exc
    if entry is None:
        raise fastapi.HTTPException(status_code=404, detail='Запись не найдена')
    return entry


@router.delete('/{entry_id}', operation_id='deleteEntry', status_code=204)
def delete_entry(entry_id: uuid.UUID, request: fastapi.Request) -> None:
    if not get_entries_service(request).delete(entry_id):
        raise fastapi.HTTPException(status_code=404, detail='Запись не найдена')
