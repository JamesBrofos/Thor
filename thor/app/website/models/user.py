import datetime as dt
from flask_login import UserMixin
from sqlalchemy.ext.hybrid import hybrid_property
from .. import db, bcrypt


class User(db.Model, UserMixin):
    """User Class"""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(80), unique=True)
    _password = db.Column(db.String(100))
    confirmed = db.Column(db.Boolean, default=False)
    auth_token = db.Column(db.String(80), unique=True)
    experiments = db.relationship(
        "Experiment",
        backref="users",
        lazy="dynamic"
    )

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
        self.auth_token = bcrypt.generate_password_hash(
            username + str(dt.datetime.today())
        ).decode("utf-8")

    @hybrid_property
    def password(self):
        return self._password

    @password.setter
    def _set_password(self, plaintext):
        self._password = bcrypt.generate_password_hash(
            plaintext
        ).decode("utf-8")

    def is_correct_password(self, plaintext):
        return bcrypt.check_password_hash(self._password, plaintext)
