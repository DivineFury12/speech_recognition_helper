"""Use 384-dimensional local multilingual E5 vectors.

Revision ID: 0002_local_embeddings
Revises: 0001_entries
"""

from alembic import op


revision = '0002_local_embeddings'
down_revision = '0001_entries'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The remote embedding API returned 402 during setup, so earlier vectors
    # cannot be mixed with the local model's vector space.
    op.execute('UPDATE entries SET embedding = NULL WHERE embedding IS NOT NULL')
    op.execute('ALTER TABLE entries ALTER COLUMN embedding TYPE vector(384)')


def downgrade() -> None:
    op.execute('UPDATE entries SET embedding = NULL WHERE embedding IS NOT NULL')
    op.execute('ALTER TABLE entries ALTER COLUMN embedding TYPE vector(1024)')
