from .index import index
from .login import login
from .signup import signup
from .about import about
from .api import api_endpoints
from .experiment import experiment


blueprints = [
    index,
    login,
    signup,
    about,
    experiment,
] + api_endpoints
