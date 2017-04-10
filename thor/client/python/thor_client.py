import requests
import json
from .base_url import base_url
from .experiment_client import ExperimentClient
from .json_parser import json_parser


class ThorClient(object):
    """Thor Client Class"""
    def __init__(self, auth_token):
        """Initialize the parameters of the Thor API client."""
        self.auth_token = auth_token

    def create_experiment(self, name, dimensions, acq_func):
        """Create an experiment."""
        post_data = {
            "name": name,
            "auth_token": self.auth_token,
            "dimensions": dimensions,
            "acq_func": acq_func
        }
        result = requests.post(
            url=base_url.format("create_experiment"),
            json=post_data
        )
        return json_parser(result, self.auth_token, ExperimentClient)

    def experiment_for_name(self, name):
        """Get an experiment with a given name."""
        post_data = {"name": name, "auth_token": self.auth_token}
        result = requests.post(
            url=base_url.format("experiment_for_name"),
            json=post_data
        )
        return json_parser(result, self.auth_token, ExperimentClient)
