from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_user
from ..forms import SignupForm
from ..models import User
from .. import db


signup = Blueprint("signup", __name__)

@signup.route("/signup/", methods=["GET", "POST"])
def page():
    form = SignupForm()

    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            password=form.password.data,
            email=form.email.data
        )
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("index.page"))

    return render_template("signup.jinja2", form=form)
