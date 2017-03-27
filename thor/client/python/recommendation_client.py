import requests
import json
from .base_url import base_url
from .json_parser import json_parser


class RecommendationClient(object):
    def __init__(self, identifier, config, auth_token):
        """Initialize the parameters of the recommendation client object."""
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
        result = requests.post(
            url=base_url.format("submit_recommendation"),
            json=post_data
        )
        return json_parser(result, self.auth_token)

    @classmethod
    def from_dict(cls, dictionary, auth_token):
        return cls(
            identifier=dictionary["id"],
            config=json.loads(dictionary["x"]),
            auth_token=auth_token
        )
