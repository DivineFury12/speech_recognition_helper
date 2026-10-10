"""Store vectors by model so E5 and FRIDA can coexist.

Revision ID: 0003_embedding_models
Revises: 0002_local_embeddings
"""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa


revision = '0003_embedding_models'
down_revision = '0002_local_embeddings'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'entry_embeddings',
        sa.Column('entry_id', sa.Uuid(), sa.ForeignKey('entries.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('model_name', sa.String(128), primary_key=True),
        sa.Column('embedding', pgvector.sqlalchemy.VECTOR(), nullable=False),
    )
    op.create_index('ix_entry_embeddings_model_name', 'entry_embeddings', ['model_name'])
    op.execute("""
        INSERT INTO entry_embeddings (entry_id, model_name, embedding)
        SELECT id, 'intfloat/multilingual-e5-small', embedding
        FROM entries WHERE embedding IS NOT NULL
    """)
    op.drop_column('entries', 'embedding')


def downgrade() -> None:
    op.add_column('entries', sa.Column('embedding', pgvector.sqlalchemy.VECTOR(384), nullable=True))
    op.execute("""
        UPDATE entries AS e SET embedding = v.embedding::vector(384)
        FROM entry_embeddings AS v
        WHERE v.entry_id = e.id AND v.model_name = 'intfloat/multilingual-e5-small'
    """)
    op.drop_index('ix_entry_embeddings_model_name', table_name='entry_embeddings')
    op.drop_table('entry_embeddings')
