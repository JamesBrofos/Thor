import json


def encode_recommendation(rec, dims):
    return json.dumps({d.name: v for v, d in zip(rec, dims)})
