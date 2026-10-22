"""Recurrence step: «каждые 3 дня», «раз в 2 недели».

Revision ID: 0012_recurrence_interval
Revises: 0011_conversation_created
"""

from alembic import op
import sqlalchemy as sa


revision = '0012_recurrence_interval'
down_revision = '0011_conversation_created'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('entries', sa.Column('recurrence_interval', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('entries', 'recurrence_interval')
