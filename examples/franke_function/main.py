import numpy as np
from thor.client.python import ThorClient
from thor.evaluation import franke


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."

# Create space.
dims = [
    {"name": "x", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "y", "dim_type": "linear", "low": 0., "high": 1.}
]

# Create experiment.
ec = ThorClient(auth_token)
try:
    exp = ec.create_experiment(
        "Franke Function",
        dims,
        50,
        50,
        "expected_improvement"
    )
except ValueError:
    exp = ec.experiment_for_name("Franke Function")

for i in range(30):
    rec = exp.create_recommendation()
    x = rec.config
    val = franke(np.array([x["x"], x["y"]]))
    rec.submit_recommendation(val)
    print((i, x, val))
