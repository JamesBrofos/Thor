from flask import Blueprint, render_template, redirect, flash, request, url_for
from flask_login import login_user, logout_user
from ..forms import LoginForm
from ..models import User


login = Blueprint("login", __name__)

@login.route("/login/", methods=["GET", "POST"])
def login_page():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(
            username=form.username.data
        ).first()
        login_user(user)
        flash("Logged in successfully!", "success")
        next = request.args.get("next")

        return redirect(next or url_for("index.page"))

    return render_template("login.jinja2", form=form)

@login.route("/logout/")
def logout_page():
    logout_user()
    flash("Logged out successfully!", "success")
    return redirect(url_for("index.page"))
