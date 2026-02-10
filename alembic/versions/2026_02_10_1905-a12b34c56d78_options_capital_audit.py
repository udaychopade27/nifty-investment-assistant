"""options capital audit tables

Revision ID: a12b34c56d78
Revises: 8c1f2e9a7b3c
Create Date: 2026-02-10 19:05:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a12b34c56d78"
down_revision: Union[str, Sequence[str], None] = "8c1f2e9a7b3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "options_capital_month",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("monthly_capital", sa.Numeric(12, 2), nullable=False),
        sa.Column("initialized", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("month", name="uq_options_capital_month_month"),
    )
    op.create_index("ix_options_capital_month_month", "options_capital_month", ["month"], unique=True)

    op.create_table(
        "options_capital_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("rollover_applied", sa.Numeric(12, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("previous_capital", sa.Numeric(12, 2), nullable=True),
        sa.Column("new_capital", sa.Numeric(12, 2), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_options_capital_event_month", "options_capital_event", ["month"], unique=False)
    op.create_index("ix_options_capital_event_event_type", "options_capital_event", ["event_type"], unique=False)
    op.create_index(
        "ix_options_capital_event_month_created",
        "options_capital_event",
        ["month", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_options_capital_event_month_created", table_name="options_capital_event")
    op.drop_index("ix_options_capital_event_event_type", table_name="options_capital_event")
    op.drop_index("ix_options_capital_event_month", table_name="options_capital_event")
    op.drop_table("options_capital_event")

    op.drop_index("ix_options_capital_month_month", table_name="options_capital_month")
    op.drop_table("options_capital_month")
