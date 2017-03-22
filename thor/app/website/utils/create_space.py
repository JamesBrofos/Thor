from thor.space import LinearDimension, Space


def create_space(model_dims):
    dims = []
    for d in model_dims:
        dims.append(LinearDimension(d.low, d.high))
    return Space(dims)
