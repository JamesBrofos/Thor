import numpy as np
from thor_client import ThorClient
from thor.evaluation import franke


# Authentication token.
auth_token = "$2b$12$.wA/rDDnUeNFoXOxBcJ6ze2ZzIF16ThQMM8hPfvuTwtTwZYVDlpXK"


# Create experiment.
tc = ThorClient(auth_token)
name = "Franke Function"
# Create space.
dims = [
    {"name": "x", "dim_type": "linear", "low": 0., "high": 1.},
    {"name": "y", "dim_type": "linear", "low": 0., "high": 1.}
]
exp = tc.create_experiment(name, dims, "expected_improvement")


# Main optimization loop.
for i in range(60):
    rec = exp.create_recommendation(rand_prob=1.)
    x = rec.config
    val = franke(np.array([x["x"], x["y"]]))
    rec.submit_recommendation(val)
    print((i, x, val))

# Get best configuration.
print("Best configuration:")
print(exp.best_configuration())

