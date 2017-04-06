from functools import wraps
from flask import request, abort
from ..models import User
from .. import db

# The actual decorator function.
def require_apikey(view_function):
    @wraps(view_function)
    # The new, post-decoration function. Note *args and **kwargs here.
    def decorated_function(*args, **kwargs):
        user = User.query.filter_by(auth_token=request.json["auth_token"]).first()
        # This checks that a user exists with the associated API key and that
        # the user has verified their email address.
        if user and user.confirmed:
            return view_function(user, *args, **kwargs)
        else:
            abort(401)

    return decorated_function
