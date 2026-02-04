"""base investment plan

Revision ID: 9f3c2a7c2b1a
Revises: d90825186645
Create Date: 2026-02-04 14:50:00.000000+05:30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9f3c2a7c2b1a"
down_revision = "d90825186645"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "base_investment_plan",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("base_capital", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("strategy_version", sa.String(length=50), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("month"),
    )
    op.create_index(
        op.f("ix_base_investment_plan_month"),
        "base_investment_plan",
        ["month"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_base_investment_plan_month"), table_name="base_investment_plan")
    op.drop_table("base_investment_plan")
