"""Remember the last selected entry for conversational references.

Revision ID: 0005_conversation_entry_context
Revises: 0004_conversations
"""

from alembic import op
import sqlalchemy as sa


revision = '0005_conversation_entry_context'
down_revision = '0004_conversations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('conversations', sa.Column('last_entry_id', sa.Uuid(), nullable=True))


def downgrade() -> None:
    op.drop_column('conversations', 'last_entry_id')
