import json


def encode_recommendation(rec, dims):
    return json.dumps({
        d.name: int(v) if d.dim_type == "integer" else v
        for v, d in zip(rec, dims)
    })
