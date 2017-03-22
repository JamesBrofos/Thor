import unittest
import numpy as np
from thor.client import ExperimentClient



if True:
    auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."
else:
    auth_token = "FAKE"

class ClientTest(unittest.TestCase):
    def test_create_experiment(self):
        ec = ExperimentClient(auth_token)
        dims = [
            {
                "name": "x",
                "dim_type": "linear",
                "low": 0.,
                "high": 1.
            }
        ]
        experiment = ec.create_experiment("Test Experiment Name", dims)

    def test_create_recommendation(self):
        ec = ExperimentClient(auth_token)
        rec = ec.experiment_for_name("Test Experiment Name").create_recommendation()
        rec.submit_recommendation(np.random.uniform())
        print(rec.config)

if __name__ == "__main__":
    unittest.main()
