"""empty message

Revision ID: 5f020b8f6e19
Revises: 19b9b8f342e6
Create Date: 2017-03-16 19:27:23.515147

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5f020b8f6e19'
down_revision = '19b9b8f342e6'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('experiments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=80), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('experiments_for_user',
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('experiment_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('user_id', 'experiment_id')
    )
    op.add_column('users', sa.Column('auth_token', sa.String(length=80), nullable=True))
    op.alter_column('users', '_password',
               existing_type=sa.VARCHAR(length=200),
               type_=sa.String(length=100),
               existing_nullable=True)
    op.create_unique_constraint(None, 'users', ['auth_token'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'users', type_='unique')
    op.alter_column('users', '_password',
               existing_type=sa.String(length=100),
               type_=sa.VARCHAR(length=200),
               existing_nullable=True)
    op.drop_column('users', 'auth_token')
    op.drop_table('experiments_for_user')
    op.drop_table('experiments')
    # ### end Alembic commands ###
