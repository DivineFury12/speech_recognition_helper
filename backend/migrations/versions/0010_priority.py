"""Entry priority: «важная задача».

Revision ID: 0010_priority
Revises: 0009_reminders_series
"""

from alembic import op
import sqlalchemy as sa


revision = '0010_priority'
down_revision = '0009_reminders_series'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('entries', sa.Column('priority', sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column('entries', 'priority')
