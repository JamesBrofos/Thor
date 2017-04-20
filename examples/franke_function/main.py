import numpy as np
from thor_client import ThorClient
from thor.evaluation import franke


# Authentication token.
auth_token = "$2b$12$.wA/rDDnUeNFoXOxBcJ6ze2ZzIF16ThQMM8hPfvuTwtTwZYVDlpXK"


# Create experiment.
tc = ThorClient(auth_token)
name = "Franke Function"
try:
    # Create space.
    dims = [
        {"name": "x", "dim_type": "linear", "low": 0., "high": 1.},
        {"name": "y", "dim_type": "linear", "low": 0., "high": 1.}
    ]
    exp = tc.create_experiment(name, dims, "hedge")
except ValueError:
    exp = tc.experiment_for_name(name)


# Main optimization loop.
for i in range(30):
    rec = exp.create_recommendation()
    x = rec.config
    val = franke(np.array([x["x"], x["y"]]))
    rec.submit_recommendation(val)
    print((i, x, val))

# Get best configuration.
print("Best configuration:")
print(exp.best_configuration())

# Submit an arbitrary observation.
o = np.random.uniform(size=(2, ))
exp.submit_observation({"x": o[0], "y": o[1]}, franke(o))
