"""Persist idempotent assistant request results.

Revision ID: 0007_assistant_request_receipts
Revises: 0006_meeting_duration_default
"""

from alembic import op
import sqlalchemy as sa


revision = '0007_assistant_request_receipts'
down_revision = '0006_meeting_duration_default'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'assistant_requests',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('fingerprint', sa.String(length=64), nullable=False),
        sa.Column('response', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('assistant_requests')
