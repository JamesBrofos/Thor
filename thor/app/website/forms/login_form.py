from flask_wtf import Form
from wtforms import TextField, SubmitField, PasswordField, BooleanField
from wtforms.validators import Required, Email, ValidationError
from .validators import Exists, Active, CorrectPassword
from ..models import User


class LoginForm(Form):
    username = TextField("Username", validators=[
        Required(),
        Exists(User, User.username, message="Username does not exist."),
        Active(),
    ])
    password = PasswordField("Password", validators=[
        Required(), CorrectPassword()
    ])
    submit = SubmitField("Submit")
