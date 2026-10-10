"""SQLAlchemy engine and sessions for entry storage."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(database_url: URL) -> tuple[sessionmaker[Session], Engine]:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False), engine
