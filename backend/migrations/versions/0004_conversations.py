"""Add conversation memory and ordered messages.

Revision ID: 0004_conversations
Revises: 0003_embedding_models
"""

from alembic import op
import sqlalchemy as sa


revision = '0004_conversations'
down_revision = '0003_embedding_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('pending_command', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        'conversation_messages',
        sa.Column('sequence', sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column('id', sa.Uuid(), nullable=False, unique=True),
        sa.Column('conversation_id', sa.Uuid(), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(16), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_conversation_messages_conversation_id', 'conversation_messages', ['conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_conversation_messages_conversation_id', table_name='conversation_messages')
    op.drop_table('conversation_messages')
    op.drop_table('conversations')
