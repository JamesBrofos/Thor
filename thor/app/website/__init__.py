import os
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_bcrypt import Bcrypt
from flask_login import LoginManager


# Initialize app.
app = Flask(__name__)
# App configurations.
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://thor_server:thor@localhost:5432/thor"
)
app.config["SECRET_KEY"] = "50601550186443463175"


# Add-ons for the app.
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
mail = Mail(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login.login_page"


# Load in the blueprints for individual pages. Notice that we have to perform
# the views import at this late stage after all of the other dependencies have
# been created (namely the database).
from .views import blueprints
from .models import User

for blueprint in blueprints:
    app.register_blueprint(blueprint)

# This callback is used to reload the user object from the user ID stored in the
# session.
@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()
