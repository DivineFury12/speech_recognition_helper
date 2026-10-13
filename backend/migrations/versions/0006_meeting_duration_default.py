"""Give existing meetings the new one-hour default duration.

Revision ID: 0006_meeting_duration_default
Revises: 0005_conversation_entry_context
"""

from alembic import op
import sqlalchemy as sa


revision = '0006_meeting_duration_default'
down_revision = '0005_conversation_entry_context'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "UPDATE entries SET duration_minutes = 60 "
        "WHERE kind = 'meeting' AND duration_minutes IS NULL"
    ))


def downgrade() -> None:
    # Existing 60-minute durations cannot be distinguished from backfilled ones.
    pass
