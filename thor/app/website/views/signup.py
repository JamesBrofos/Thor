from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_user
from ..utils import require_unauthed, ts, send_email
from ..forms import SignupForm
from ..models import User
from .. import db


signup = Blueprint("signup", __name__)

@signup.route("/signup/", methods=["GET", "POST"])
@require_unauthed
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

        # Email confirmation link
        subject = "Confirm your email"
        token = ts.dumps(user.email, salt="email-confirm-key")
        confirm_url = url_for("signup.confirm_email", token=token,
                              _external=True)
        html = render_template("email/activate.jinja2", confirm_url=confirm_url)
        send_email(user.email, subject, html)
        # Show notification of email confirmation.
        flash("Sent confirmation email to {}".format(form.email.data), "success")

        return redirect(url_for("index.page"))

    return render_template("signup.jinja2", form=form)


@signup.route("/confirm/<string:token>")
@require_unauthed
def confirm_email(token):
    try:
        email = ts.loads(token, salt="email-confirm-key", max_age=86400)
    except:
        abort(404)

    user = User.query.filter_by(email=email).first_or_404()
    user.email_confirmed = True
    db.session.commit()

    flash("Successfuly confirmed {}.".format(email), "success")

    return redirect(url_for("login.login_page"))
