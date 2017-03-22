from flask_wtf import Form
from wtforms import TextField, SubmitField, PasswordField, BooleanField
from wtforms.validators import (
    Required, Email, ValidationError, Length, Regexp
)
from .validators import Exists, Active, CorrectPassword, Unique
from ..models import User


class SignupForm(Form):
    username = TextField("Username", validators=[
        Required(),
        Length(
            min=4,
            max=25,
            message="Your username must be between 4 to 25 characters"
        ),
        Regexp(
            '^[A-za-z][A-za-z0-9_]*$',
            0,
            "Usernames must start with a letter and contain only letters,"
            " underscores, and numbers"
        ),
        Unique(
            User,
            User.username,
            message="There is already an account with that username"
        )
    ])
    password = PasswordField("Password", validators=[Required()])
    email = TextField("Email", validators=[
        Required(),
        Email(),
        Unique(
            User,
            User.email,
            message="There is already an account with that email"
        )
    ])
    submit = SubmitField("Submit")
