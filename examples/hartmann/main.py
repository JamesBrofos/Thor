import numpy as np
from thor.client.python import ThorClient
from thor.evaluation import hartmann_6


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."

# Create space.
dims = [
    {"name": "x1", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "x2", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "x3", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "x4", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "x5", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "x6", "dim_type": "linear", "low": 0., "high": 1.},
]

# Create experiment.
ec = ThorClient(auth_token)
try:
    exp = ec.create_experiment(
        "Hartmann 6-D",
        dims,
        100,
        100,
        "expected_improvement"
    )
except ValueError:
    exp = ec.experiment_for_name("Hartmann 6-D")

for i in range(200):
    rec = exp.create_recommendation()
    x = rec.config
    val = -hartmann_6(np.array([
        x["x1"], x["x2"], x["x3"], x["x4"], x["x5"], x["x6"]
    ]))
    rec.submit_recommendation(val)
    print((i, x, val))
