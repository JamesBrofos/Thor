from functools import wraps
from flask import request, abort
from ..models import User
from .. import db

# The actual decorator function
def require_apikey(view_function):
    @wraps(view_function)
    # the new, post-decoration function. Note *args and **kwargs here.
    def decorated_function(*args, **kwargs):
        # if request.args.get('key') and request.args.get('key') == APPKEY_HERE:
        user = User.query.filter_by(auth_token=request.json["auth_token"]).first()
        if user:
            return view_function(user, *args, **kwargs)
        else:
            abort(401)
    return decorated_function
