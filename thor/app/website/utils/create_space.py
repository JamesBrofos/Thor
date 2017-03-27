from thor.space import LinearDimension, Space


def create_space(model_dims):
    return Space([d.to_thor_dimension() for d in model_dims])
