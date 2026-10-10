"""PostgreSQL connection settings shared by the app and migrations."""

import os

from sqlalchemy.engine import URL, make_url


def get_database_url() -> URL | None:
    explicit = os.getenv('DATABASE_URL', '').strip()
    if explicit:
        return make_url(explicit)

    host = os.getenv('POSTGRES_HOST', '').strip()
    if not host:
        return None
    return URL.create(
        'postgresql+psycopg',
        username=os.getenv('POSTGRES_USER', 'assistant'),
        password=os.getenv('POSTGRES_PASSWORD', 'assistant'),
        host=host,
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'assistant'),
    )
