"""rebalance log

Revision ID: 1c2d3e4f5a6b
Revises: f31e64e74ed7
Create Date: 2026-02-06 13:35:00.000000+05:30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1c2d3e4f5a6b'
down_revision = 'f31e64e74ed7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'rebalance_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('fiscal_year', sa.String(length=9), nullable=False),
        sa.Column('rebalance_date', sa.Date(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('fiscal_year')
    )
    op.create_index(op.f('ix_rebalance_log_fiscal_year'), 'rebalance_log', ['fiscal_year'], unique=True)
    op.create_index(op.f('ix_rebalance_log_rebalance_date'), 'rebalance_log', ['rebalance_date'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_rebalance_log_rebalance_date'), table_name='rebalance_log')
    op.drop_index(op.f('ix_rebalance_log_fiscal_year'), table_name='rebalance_log')
    op.drop_table('rebalance_log')
