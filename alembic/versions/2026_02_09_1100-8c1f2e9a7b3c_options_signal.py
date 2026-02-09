"""add options_signal table

Revision ID: 8c1f2e9a7b3c
Revises: 2f9b7a1c0d3e
Create Date: 2026-02-09 11:00:00.000000+05:30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "8c1f2e9a7b3c"
down_revision = "2f9b7a1c0d3e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS options_signal (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            signal_ts TIMESTAMP NOT NULL,
            underlying VARCHAR(50) NOT NULL,
            signal VARCHAR(20) NOT NULL,
            entry NUMERIC(12,2) NOT NULL,
            stop_loss NUMERIC(12,2) NOT NULL,
            target NUMERIC(12,2) NOT NULL,
            rr NUMERIC(6,2) NOT NULL,
            estimated_profit NUMERIC(12,2) NOT NULL,
            entry_source VARCHAR(20) NOT NULL,
            blocked BOOLEAN NOT NULL DEFAULT false,
            reason TEXT NULL,
            payload JSON NULL,
            created_at TIMESTAMP NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_date ON options_signal (date)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_underlying ON options_signal (underlying, signal_ts)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_options_signal_underlying")
    op.execute("DROP INDEX IF EXISTS ix_options_signal_date")
    op.execute("DROP TABLE IF EXISTS options_signal")
