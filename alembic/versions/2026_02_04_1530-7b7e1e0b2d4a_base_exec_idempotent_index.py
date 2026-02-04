"""base execution idempotent index

Revision ID: 7b7e1e0b2d4a
Revises: 9f3c2a7c2b1a
Create Date: 2026-02-04 15:30:00.000000+05:30

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "7b7e1e0b2d4a"
down_revision = "9f3c2a7c2b1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_base_exec_month_symbol
        ON executed_investment (etf_symbol, date_trunc('month', executed_at))
        WHERE capital_bucket = 'base'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_base_exec_month_symbol")
