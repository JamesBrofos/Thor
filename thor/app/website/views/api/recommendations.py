import datetime as dt
import numpy as np
from flask import Blueprint, request, jsonify
from flask_api import status
from thor.acquisitions import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ImprovementProbability
)
from thor.models.tuning import fit_marginal_likelihood
from thor.optimizers import BayesianOptimization
from thor.kernels import SquaredExponentialKernel, MaternKernel
from ...models import Experiment, Observation
from ... import db
from ...utils import (
    require_apikey,
    encode_recommendation,
    decode_recommendation,
    create_space
)


recommendations = Blueprint("recommendations", __name__)


@recommendations.route("/api/create_recommendation/", methods=["POST"])
@require_apikey
def create_recommendation(user):
    # Extract parameters.
    experiment_id = request.json["experiment_id"]
    date = dt.datetime.today()
    # Get the experiment corresponding to this observation.
    e = Experiment.query.filter_by(id=experiment_id).first()
    dims = e.dimensions.all()
    space = create_space(dims)
    # Number of randomly positioned observations to create.
    n_random = 5 * len(dims)

    # Either use Bayesian optimization or generate a random point depending on
    # the number of observations collected so far.
    if e.observations.count() <= n_random:
        rec = encode_recommendation(space.sample().ravel(), dims)
    else:
        # Get pending and non-pending observations.
        observed = e.observations.filter(Observation.pending==False).all()
        pending = e.observations.filter(Observation.pending==True).all()
        # Do Bayesian optimization.
        X, y = decode_recommendation(observed, dims)
        # First fit the Gaussian process using observed data.
        prior_mean = 0.
        gp = fit_marginal_likelihood(
            X, y, e.n_restarts, MaternKernel, prior_mean
        )
        # Create fantasy observations for the pending values.
        if len(pending) > 0:
            pass

        acq = {
            "expected_improvement": ExpectedImprovement,
            "improvement_probability": ImprovementProbability,
            "upper_confidence_bound": UpperConfidenceBound
        }[e.acq_func](gp)
        bo = BayesianOptimization(acq, space)
        recs, acqs = bo.recommend(e.n_recs)
        rec = encode_recommendation(recs[acqs.argmax()], dims)

    # Submit recommendation to user and store in the Thor database. It is
    # created initially without a response and is marked as pending.
    obs = Observation(str(rec), date)
    e.observations.append(obs)
    # Commit changes.
    db.session.commit()

    return jsonify(obs.to_dict())

@recommendations.route("/api/submit_recommendation/", methods=["POST"])
@require_apikey
def submit_recommendation(user):
    # Extract recommendation identifier and observed value.
    identifier = request.json["recommendation_id"]
    value = request.json["value"]
    # Update observation.
    obs = Observation.query.filter_by(id=identifier).first()

    if obs is not None:
        e = Experiment.query.filter_by(id=obs.experiment_id).first()
        if e.user_id == user.id:
            obs.y = value
            obs.pending = False
            # Commit changes.
            db.session.commit()

            return jsonify({"id": identifier, "submitted": True})
    else:
        err = {"error": "Observation does not exist."}
        print(err)
        return (jsonify(err), status.HTTP_400_BAD_REQUEST)

@recommendations.route("/api/pending_recommendations/", methods=["POST"])
@require_apikey
def pending_recommendations(user):
    # Extract experiment.
    experiment_id = request.json["experiment_id"]
    exp = user.experiments.query.filter(Experiment.id==experiment_id).first()

    if exp:
        pending_obs = exp.observations.filter(Observation.pending==False).all()
        return jsonify([o.to_json() for o in pending_obs])
    else:
        err = {
            "error": "Experiment with identifier {} does not exist.".format(
                experiment_id
            )
        }
        return (jsonify(err), status.HTTP_400_BAD_REQUEST)

