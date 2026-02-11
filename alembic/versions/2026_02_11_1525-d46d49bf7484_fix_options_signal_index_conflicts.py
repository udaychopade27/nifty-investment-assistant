"""fix options_signal index conflicts

Revision ID: d46d49bf7484
Revises: a12b34c56d78
Create Date: 2026-02-11 15:25:03.308529+05:30

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "d46d49bf7484"
down_revision = "a12b34c56d78"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Replace old composite index name with current model-compatible indexes.
    op.execute("DROP INDEX IF EXISTS ix_options_signal_underlying")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_underlying ON options_signal (underlying)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_signal ON options_signal (signal)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_signal_ts ON options_signal (signal_ts)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_underlying_ts ON options_signal (underlying, signal_ts)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_options_signal_underlying_ts")
    op.execute("DROP INDEX IF EXISTS ix_options_signal_signal_ts")
    op.execute("DROP INDEX IF EXISTS ix_options_signal_signal")
    op.execute("DROP INDEX IF EXISTS ix_options_signal_underlying")
    op.execute("CREATE INDEX IF NOT EXISTS ix_options_signal_underlying ON options_signal (underlying, signal_ts)")
