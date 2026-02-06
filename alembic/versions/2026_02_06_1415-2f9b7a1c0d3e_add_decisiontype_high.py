"""add decision type HIGH

Revision ID: 2f9b7a1c0d3e
Revises: 1c2d3e4f5a6b
Create Date: 2026-02-06 14:15:00.000000+05:30

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '2f9b7a1c0d3e'
down_revision = '1c2d3e4f5a6b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE decisiontypeenum ADD VALUE IF NOT EXISTS 'HIGH'")


def downgrade() -> None:
    # No safe downgrade for enum value removal in Postgres.
    pass
