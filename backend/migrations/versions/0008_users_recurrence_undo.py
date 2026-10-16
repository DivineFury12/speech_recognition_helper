"""Separate users, recurring entries, undo of the last action, calendar link version.

Revision ID: 0008_users_recurrence_undo
Revises: 0007_assistant_request_receipts
"""

from alembic import op
import sqlalchemy as sa


revision = '0008_users_recurrence_undo'
down_revision = '0007_assistant_request_receipts'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('entries', sa.Column('user_id', sa.String(length=64), nullable=False,
                                       server_default='local'))
    op.add_column('entries', sa.Column('recurrence', sa.String(length=16), nullable=True))
    op.create_index('ix_entries_user_date', 'entries', ['user_id', 'date'])
    op.add_column('conversations', sa.Column('user_id', sa.String(length=64), nullable=False,
                                             server_default='local'))
    op.add_column('conversations', sa.Column('undo_record', sa.JSON(), nullable=True))
    op.create_table(
        'user_settings',
        sa.Column('user_id', sa.String(length=64), primary_key=True),
        sa.Column('feed_version', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_table('user_settings')
    op.drop_column('conversations', 'undo_record')
    op.drop_column('conversations', 'user_id')
    op.drop_index('ix_entries_user_date', table_name='entries')
    op.drop_column('entries', 'recurrence')
    op.drop_column('entries', 'user_id')
