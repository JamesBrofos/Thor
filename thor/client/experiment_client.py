import requests
import json


base_url = "http://127.0.0.1:5000/api/{}/"


class Recommendation(object):
    def __init__(self, identifier, config, auth_token):
        """Initialize the parameters of the experiment object."""
        self.recommendation_id = identifier
        self.config = config
        self.auth_token = auth_token

    def submit_recommendation(self, value):
        """Submit the returned metric value for a point that was recommended by
        the Bayesian optimization routine.
        """
        post_data = {
            "auth_token": self.auth_token,
            "recommendation_id": self.recommendation_id,
            "value": value
        }
        requests.post(
            url=base_url.format("submit_recommendation"), json=post_data
        )

    @classmethod
    def from_dict(cls, dictionary, auth_token):
        return cls(
            identifier=dictionary["id"],
            config=json.loads(dictionary["x"]),
            auth_token=auth_token
        )

class Experiment(object):
    """Experiment Class"""
    def __init__(self, identifier, name, date, dims, auth_token):
        """Initialize parameters of the experiment object."""
        self.experiment_id = identifier
        self.name = name
        self.date = date
        self.dims = dims
        self.auth_token = auth_token

    def create_recommendation(self):
        """Get a recommendation for a point to evaluate next."""
        post_data = {
            "auth_token": self.auth_token,
            "experiment_id": self.experiment_id
        }
        return Recommendation.from_dict(json.loads(requests.post(
            url=base_url.format("create_recommendation"), json=post_data
        ).text), self.auth_token)

    @classmethod
    def from_dict(cls, dictionary, auth_token):
        """Create an experiment object from a dictionary representation. Pass
        the authentication token as an additional parameter.

        TODO: Can the authentication token be a return parameter?
        """
        return cls(
            identifier=dictionary["id"],
            name=dictionary["name"],
            date=dictionary["date"],
            dims=dictionary["dimensions"],
            auth_token=auth_token
        )


class ExperimentClient(object):
    def __init__(self, auth_token):
        """Initialize the parameters of the experiments API client."""
        self.auth_token = auth_token

    def create_experiment(self, name, dimensions):
        """Create an experiment."""
        post_data = {
            "name": name,
            "auth_token": self.auth_token,
            "dimensions": dimensions
        }
        return Experiment.from_dict(json.loads(requests.post(
            url=base_url.format("create_experiment"), json=post_data
        ).text), self.auth_token)

    def experiment_for_name(self, name):
        """Get an experiment with a given name."""
        post_data = {"name": name, "auth_token": self.auth_token}
        return Experiment.from_dict(json.loads(requests.post(
            url=base_url.format("experiment_for_name"), json=post_data
        ).text), self.auth_token)

