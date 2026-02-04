# alembic/versions/001_initial.py

"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2026-01-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create monthly_config table
    op.create_table('monthly_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('month', sa.Date(), nullable=False),
        sa.Column('monthly_capital', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('base_capital', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('tactical_capital', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('trading_days', sa.Integer(), nullable=False),
        sa.Column('daily_tranche', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('strategy_version', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('month')
    )

    # Create daily_decision table
    op.create_table('daily_decision',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('monthly_config_id', sa.Integer(), nullable=False),
        sa.Column('decision_type', sa.String(), nullable=False),
        sa.Column('nifty_change_pct', sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column('suggested_total_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('actual_investable_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('unused_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('remaining_base_capital', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('remaining_tactical_capital', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('strategy_version', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date')
    )

    # Create etf_decision table
    op.create_table('etf_decision',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('daily_decision_id', sa.Integer(), nullable=False),
        sa.Column('etf_symbol', sa.String(), nullable=False),
        sa.Column('ltp', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('effective_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('units', sa.Integer(), nullable=False),
        sa.Column('actual_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['daily_decision_id'], ['daily_decision.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create executed_investment table - CRITICAL: etf_decision_id is NULLABLE for base investments
    op.create_table('executed_investment',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('etf_decision_id', sa.Integer(), nullable=True),  # ✅ NULLABLE for base investments
        sa.Column('etf_symbol', sa.String(), nullable=False),
        sa.Column('units', sa.Integer(), nullable=False),
        sa.Column('executed_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('total_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('slippage_pct', sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column('capital_bucket', sa.String(), nullable=False),
        sa.Column('executed_at', sa.DateTime(), nullable=False),
        sa.Column('execution_notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['etf_decision_id'], ['etf_decision.id'], ),  # Foreign key but allows NULL
        sa.PrimaryKeyConstraint('id')
    )

    # Create extra_capital_injection table
    op.create_table('extra_capital_injection',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('month', sa.Date(), nullable=False),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.Column('injected_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create crash_opportunity_signal table
    op.create_table('crash_opportunity_signal',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('nifty_change_pct', sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column('crash_magnitude', sa.String(), nullable=False),
        sa.Column('recommended_deployment', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date')
    )

    # Create monthly_summary table
    op.create_table('monthly_summary',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('month', sa.Date(), nullable=False),
        sa.Column('total_invested', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('base_deployed', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('tactical_deployed', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('extra_deployed', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('unused_base', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('unused_tactical', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('tactical_carried_forward', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('investment_days', sa.Integer(), nullable=False),
        sa.Column('total_units_purchased', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('month')
    )

    # Create trading_holiday table
    op.create_table('trading_holiday',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date')
    )

    # Create market_data_cache table
    op.create_table('market_data_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('high', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('low', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('close', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'date')
    )


def downgrade():
    op.drop_table('market_data_cache')
    op.drop_table('trading_holiday')
    op.drop_table('monthly_summary')
    op.drop_table('crash_opportunity_signal')
    op.drop_table('extra_capital_injection')
    op.drop_table('executed_investment')
    op.drop_table('etf_decision')
    op.drop_table('daily_decision')
    op.drop_table('monthly_config')