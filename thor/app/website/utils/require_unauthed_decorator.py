from functools import wraps
from flask import flash, redirect, url_for, abort
from flask_login import current_user


def require_unauthed(func):
    """This decorator augments the decorated function to redirect the user to
    the homepage if they're already logged in.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated:
            flash("You're already authenticated!", "danger")
            return redirect(url_for("home.page"))

        return func(*args, **kwargs)

    return decorated_function
