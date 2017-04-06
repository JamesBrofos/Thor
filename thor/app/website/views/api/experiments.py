import datetime as dt
from flask import Blueprint
from flask import request, jsonify
from flask_api import status
from ...utils import require_apikey
from ...models import Experiment, Dimension, AcquisitionFunction
from ... import db


experiments = Blueprint("experiments", __name__)


@experiments.route("/api/create_experiment/", methods=["POST"])
@require_apikey
def create_experiment(user):
    # Extract name.
    name = request.json["name"]
    # Check if the experiment already exists.
    exists = db.session.query(
        user.experiments.filter(Experiment.name==name).exists()
    ).scalar() > 0

    if exists:
        err = {"error": "Experiment named '{}' already exists.".format(name)}
        print(err)
        return (jsonify(err), status.HTTP_400_BAD_REQUEST)
    else:
        # Create an experiment for this user.
        e = Experiment.from_json(request.json)
        e.acq_func = AcquisitionFunction.from_json(request.json)
        user.experiments.append(e)

        db.session.commit()
        return jsonify(e.to_dict())

@experiments.route("/api/experiment_for_name/", methods=["POST"])
@require_apikey
def experiment_for_name(user):
    # Extract name.
    name = request.json["name"]
    # Get experiments for user.
    experiment = user.experiments.filter(Experiment.name==name).first()
    if experiment:
        return jsonify(experiment.to_dict())
    else:
        err = {"error": "Experiment named '{}' does not exist.".format(name)}
        print(err)
        return (jsonify(err), status.HTTP_400_BAD_REQUEST)

