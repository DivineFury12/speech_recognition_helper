"""Backfill vectors for the selected EMBEDDING_MODEL without loading the ASR model."""

import argparse

from src.adapters.entry_repository import SqlAlchemyEntryRepository
from src.adapters.local_embeddings import create_embedding_model
from src.db import create_session_factory
from src.settings import Settings
from src.services.entries import EntryService


def main() -> None:
    parser = argparse.ArgumentParser(description='Index entries with the selected embedding model')
    parser.add_argument('--limit', type=int, help='Maximum number of entries to index')
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error('--limit must be positive')

    config = Settings.from_env()
    if config.database_url is None:
        parser.error('DATABASE_URL or POSTGRES_HOST is required')
    model = create_embedding_model(
        config.embedding_model_name, str(config.cache_dir.parent / 'embeddings'),
    )
    sessions, engine = create_session_factory(config.database_url)
    try:
        count = EntryService(SqlAlchemyEntryRepository(sessions), model).reindex_missing(
            limit=arguments.limit,
        )
        print(f'Indexed {count} entries with {model.name}')
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
