import numpy as np
import json


def decode_recommendation(observations, dims):
    X = np.zeros((len(observations), len(dims)))
    y = np.zeros((len(observations, )))
    for i, obs in enumerate(observations):
        o = json.loads(obs.x)
        X[i] = np.array([o[dim.name] for dim in dims])
        y[i] = obs.y

    return X, y
