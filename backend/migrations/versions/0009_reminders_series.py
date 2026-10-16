"""Advance reminders, skipped occurrences and the end date of a recurring series.

Revision ID: 0009_reminders_series
Revises: 0008_users_recurrence_undo
"""

from alembic import op
import sqlalchemy as sa


revision = '0009_reminders_series'
down_revision = '0008_users_recurrence_undo'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('entries', sa.Column('remind_minutes', sa.Integer(), nullable=True))
    op.add_column('entries', sa.Column('recurrence_until', sa.Date(), nullable=True))
    op.add_column('entries', sa.Column('exdates', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('entries', 'exdates')
    op.drop_column('entries', 'recurrence_until')
    op.drop_column('entries', 'remind_minutes')
