from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from ... import db

api = Blueprint("api", __name__)

@api.route("/api/")
@login_required
def page():
    return render_template("api.jinja2")


@api.route("/api/delete/<int:experiment_id>/", methods=["POST"])
@login_required
def delete_experiment(experiment_id):
    # Query for the corresponding experiment to delete.
    exp = current_user.experiments.filter_by(id=experiment_id).first()
    if exp:
        db.session.delete(exp)
        db.session.commit()

    return redirect(url_for("index.page"))


