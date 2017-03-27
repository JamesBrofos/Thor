import numpy as np
import matplotlib.pyplot as plt
from thor.client.python import ThorClient


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."

# Define a latent function that will serve as the objective to be maximized in
# this one-dimensional example.
def f(x):
    return np.sin(10 * x) * (1 - np.tanh(4 * x ** 2))

# Create space.
dims = [{"name": "x", "dim_type": "linear", "low": 0., "high": 1.}]

# Create experiment.
ec = ThorClient(auth_token)
exp = ec.create_experiment("One Dimensional Synthetic", dims)

for i in range(20):
    rec = exp.create_recommendation()
    x = np.array(list(rec.config.values())[0])
    val = f(x)
    rec.submit_recommendation(val)

    print((i, x, val))

rec.recommendation_id = 123456
rec.submit_recommendation(1.)
