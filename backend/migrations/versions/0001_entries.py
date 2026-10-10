"""Create entry storage with pgvector embeddings.

Revision ID: 0001_entries
Revises:
"""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa


revision = '0001_entries'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.create_table(
        'entries',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('kind', sa.String(16), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('date', sa.Date(), nullable=True),
        sa.Column('time', sa.String(5), nullable=True),
        sa.Column('timezone', sa.String(64), nullable=False),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.VECTOR(1024), nullable=True),
    )
    op.create_index('ix_entries_date', 'entries', ['date'])


def downgrade() -> None:
    op.drop_index('ix_entries_date', table_name='entries')
    op.drop_table('entries')
