import numpy as np
from thor.client.python import ThorClient
from thor.evaluation import branin_hoo


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."

# Create space.
dims = [
    {"name": "x1", "dim_type": "linear", "low": -5., "high": 10.},
    {"name": "x2", "dim_type": "linear", "low": 0., "high": 15.},
]

# Create experiment.
ec = ThorClient(auth_token)
try:
    exp = ec.create_experiment(
        "Branin-Hoo",
        dims,
        50,
        50,
        "expected_improvement"
    )
except ValueError:
    exp = ec.experiment_for_name("Branin-Hoo")

for i in range(200):
    rec = exp.create_recommendation()
    x = rec.config
    val = -branin_hoo(np.array([x["x1"], x["x2"]]))
    rec.submit_recommendation(val)
    print((i, x, val))
