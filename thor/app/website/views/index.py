from flask import Blueprint, render_template


index = Blueprint("index", __name__)

@index.route("/")
def page():
    return render_template("index.jinja2")
