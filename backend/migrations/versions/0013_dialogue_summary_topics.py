"""Persist conversation summaries and entry topics.

Revision ID: 0013_dialogue_summary_topics
Revises: 0012_recurrence_interval
"""

from alembic import op
import sqlalchemy as sa


revision = '0013_dialogue_summary_topics'
down_revision = '0012_recurrence_interval'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('conversations', sa.Column('summary', sa.Text(), nullable=True))
    op.add_column('conversations', sa.Column(
        'summarized_message_count', sa.Integer(), nullable=False, server_default='0',
    ))
    op.add_column('entries', sa.Column('topics', sa.JSON(), nullable=True))
    op.execute(sa.text(
        "UPDATE entries SET topics = '[\"здоровье\"]'::json "
        "WHERE title ~* '(врач|доктор|поликлиник|больниц|диспансер|медицин|"
        "лекарств|аптек|анализ(ы|ов|ами)|анализ (крови|мочи)|обследован|"
        "стоматолог|зубн|лечен|здоров)'"
    ))


def downgrade() -> None:
    op.drop_column('entries', 'topics')
    op.drop_column('conversations', 'summarized_message_count')
    op.drop_column('conversations', 'summary')
