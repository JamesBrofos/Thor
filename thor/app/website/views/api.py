import datetime as dt
import numpy as np
# Flask imports.
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from flask_api import status
# Thor imports.
from thor.acquisitions import ExpectedImprovement
from thor.models.tuning import fit_marginal_likelihood
from thor.optimizers import BayesianOptimization
from thor.kernels import SquaredExponentialKernel, MaternKernel
# Thor Server imports.
from .. import db
from ..models import User, Experiment, Dimension, Observation
from ..utils import (
    encode_recommendation,
    decode_recommendation,
    create_space
)


api = Blueprint("api", __name__)

@api.route("/api/")
@login_required
def page():
    return render_template("api.jinja2")


@api.route("/api/create_experiment/", methods=["POST"])
def create_experiment():
    # Extract parameters to create an experiment.
    name = request.json["name"]
    auth_token = request.json["auth_token"]
    date = dt.datetime.today()
    dims = request.json["dimensions"]
    # Query for the corresponding user.
    user = User.query.filter_by(
        auth_token=auth_token
    ).first()

    # Create an experiment for this user.
    e = Experiment(name, date)
    user.experiments.append(e)
    # Add dimensions to the experiment.
    for d in dims:
        e.dimensions.append(
            Dimension(d["name"], d["dim_type"], d["low"], d["high"])
        )

    db.session.commit()
    return jsonify(e.to_dict())

@api.route("/api/experiment_for_name/", methods=["POST"])
def experiment_for_name():
    # Extract name.
    name = request.json["name"]
    auth_token = request.json["auth_token"]
    # Query for the corresponding user.
    user = User.query.filter_by(
        auth_token=auth_token
    ).first()
    # Get experiments for user.
    experiment = user.experiments.filter(name==name).first()
    return jsonify(experiment.to_dict())

@api.route("/api/create_recommendation/", methods=["POST"])
def create_recommendation():
    # Extract parameters.
    experiment_id = request.json["experiment_id"]
    auth_token = request.json["auth_token"]
    # Get the experiment corresponding to this observation.
    e = Experiment.query.filter_by(id=experiment_id).first()
    u = User.query.filter_by(id=e.user_id, auth_token=auth_token).first()
    dims = e.dimensions.all()
    space = create_space(dims)

    # Either use Bayesian optimization or generate a random point depending on
    # the number of observations collected so far.
    if e.observations.count() <= 10:
        n_dims = len(e.dimensions.all())
        rec = encode_recommendation(space.sample().ravel(), dims)
    else:
        # TODO: Recommend a point using Bayesian optimization.
        X, y = decode_recommendation(e.observations.all(), dims)
        # Do Bayesian optimization.
        gp = fit_marginal_likelihood(X, y, kernel_class=MaternKernel)
        acq = ExpectedImprovement(gp)
        bo = BayesianOptimization(acq, space)
        recs, acqs = bo.recommend(10)
        rec = encode_recommendation(recs[acqs.argmax()], dims)

    # Submit recommendation to user and store in the Thor database. It is
    # created initially without a response and is marked as pending.
    obs = Observation(str(rec))
    e.observations.append(obs)
    # Commit changes.
    db.session.commit()

    return jsonify(obs.to_dict())

@api.route("/api/submit_recommendation/", methods=["POST"])
def submit_recommendation():
    # Extract recommendation identifier and observed value.
    identifier = request.json["recommendation_id"]
    value = request.json["value"]
    # Update observation.
    obs = Observation.query.filter_by(id=identifier).first()
    obs.y = value
    obs.pending = False
    # Commit changes.
    db.session.commit()

    return ('', status.HTTP_204_NO_CONTENT)

@api.route("/api/pending_recommendations/")
def pending_recommendations():
    # Get all the pending observations for this experiment.
    pending = Experiment.query.filter_by(
        id=identifer
    ).first().observations.filter(pending=True)
