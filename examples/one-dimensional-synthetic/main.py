import numpy as np
from thor_client import ThorClient
from thor.evaluation import franke


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."


# Create experiment.
tc = ThorClient(auth_token)
name = "One-Dimensional Synthetic"
# Create space.
dims = {"name": "x", "dim_type": "linear", "low": -1., "high": 1.}
exp = tc.create_experiment(name, dims, "hedge")


# Main optimization loop.
for i in range(30):
    rec = exp.create_recommendation()
    val = -(rec.config["x"] ** 2)
    rec.submit_recommendation(val)
    print((i, val))

# Get best configuration.
print("Best configuration:")
print(exp.best_configuration())

