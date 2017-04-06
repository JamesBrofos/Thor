import numpy as np
import json
from flask import Blueprint, render_template, abort, request
from flask_login import login_required, current_user
from flask_api import status
# Bokeh imports.
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
# Thor Server imports.
from ..models import Experiment, Dimension
from ..utils import decode_recommendation


experiment = Blueprint("experiment", __name__)
js_resources = INLINE.render_js()
css_resources = INLINE.render_css()


@experiment.route("/experiment/<string:name>/analysis/")
@login_required
def analysis_page(name):
    # Query for the corresponding experiment.
    experiment = Experiment.query.filter_by(
        name=name, user_id=current_user.id
    ).first()
    # Grab the inputs arguments from the URL.
    args = request.args
    # Variable selector for analysis.
    selected_dim = int(args.get("variable", 0))

    if experiment:
        dims = experiment.dimensions.all()
        if experiment.observations.filter_by(pending=False).count() > 1:
            obs = experiment.observations.filter_by(
                pending=False
            ).order_by("date").all()
            # Extract best observation so far.
            X, y = decode_recommendation(obs, dims)
            # Visualize.
            fig = figure(
                title="Metric vs. Variable Scatter",
                tools="pan,box_zoom,reset",
                plot_height=225,
                responsive=True,
                x_axis_label="Variable",
            )
            fig.circle(X[:, selected_dim], y)
            fig.toolbar.logo = None
            script, div = components(fig)
        else:
            script, div = "", ""

        return encode_utf8(
            render_template(
                "experiment.jinja2",
                tab="analysis",
                selected_dim=selected_dim,
                experiment=experiment,
                plot_script=script,
                plot_div=div,
                js_resources=js_resources,
                css_resources=css_resources,
            )
        )
    else:
        abort(404)

@experiment.route("/experiment/<string:name>/")
@login_required
def overview_page(name):
    # Query for the corresponding experiment.
    experiment = Experiment.query.filter_by(
        name=name, user_id=current_user.id
    ).first()

    if experiment:
        dims = experiment.dimensions.all()
        if experiment.observations.filter_by(pending=False).count() > 1:
            obs = experiment.observations.filter_by(
                pending=False
            ).order_by("date").all()
            # Extract best observation so far.
            X, y = decode_recommendation(obs, dims)
            # Visualize.
            cummax = np.maximum.accumulate(y)
            r = np.arange(1, cummax.shape[0] + 1, step=1)
            fig = figure(
                title="Metric Improvement",
                tools="pan,box_zoom,reset",
                plot_height=225,
                responsive=True,
                x_axis_label="Iterations",
            )
            fig.line(r, cummax, line_width=2)
            fig.toolbar.logo = None
            script, div = components(fig)
        else:
            script, div = "", ""

        return encode_utf8(
            render_template(
                "experiment.jinja2",
                tab="overview",
                experiment=experiment,
                plot_script=script,
                plot_div=div,
                js_resources=js_resources,
                css_resources=css_resources,
            )
        )
    else:
        abort(404)

@experiment.errorhandler(404)
def page_not_found(e):
    return render_template("404.jinja2"), status.HTTP_404_NOT_FOUND

