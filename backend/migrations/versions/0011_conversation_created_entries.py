"""Remember which records a conversation created, in order («как в первом сообщении»).

Revision ID: 0011_conversation_created
Revises: 0010_priority
"""

from alembic import op
import sqlalchemy as sa


revision = '0011_conversation_created'
down_revision = '0010_priority'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('conversations', sa.Column('created_entry_ids', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('conversations', 'created_entry_ids')
