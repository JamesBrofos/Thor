"""empty message

Revision ID: 310e89e07410
Revises: ac2e8f487e03
Create Date: 2017-03-23 07:50:46.936233

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '310e89e07410'
down_revision = 'ac2e8f487e03'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('experiments', sa.Column('_id', sa.String(length=80), nullable=True))
    op.create_unique_constraint(None, 'experiments', ['_id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'experiments', type_='unique')
    op.drop_column('experiments', '_id')
    # ### end Alembic commands ###
