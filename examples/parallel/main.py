import numpy as np
from thor.client.python import ThorClient
from thor.evaluation import franke


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."


# Create experiment.
ec = ThorClient(auth_token)
try:
    # Create space.
    dims = [
        {"name": "x", "dim_type": "linear", "low": 0., "high": 1.},
        {"name": "y", "dim_type": "linear", "low": 0., "high": 1.}
    ]
    exp = ec.create_experiment(
        "Parallel Franke Function",
        dims,
        50,
        50,
        "hedge"
    )
except ValueError:
    exp = ec.experiment_for_name("Franke Function")

for i in range(5):
    rec = exp.create_recommendation()

# Query for the pending recommendations.
pending = exp.pending_recommendations()
print(pending)
