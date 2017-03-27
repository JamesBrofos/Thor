import numpy as np
import json
from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user
from flask_api import status
# Bokeh imports.
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
# Thor Server imports.
from ..models import Experiment, Dimension


experiment = Blueprint("experiment", __name__)
js_resources = INLINE.render_js()
css_resources = INLINE.render_css()

@experiment.route("/experiment/<string:name>/")
@login_required
def page(name):
    experiment = Experiment.query.filter_by(
        name=name, user_id=current_user.id
    ).first()
    if experiment:
        if experiment.observations.filter_by(pending=False).count() > 1:
            obs = experiment.observations.filter_by(
                pending=False
            ).order_by("date").all()
            # Extract best observation so far.
            targets = np.array([o.y for o in obs])
            # Visualize.
            cummax = np.maximum.accumulate(targets)
            r = np.arange(0, cummax.shape[0], step=1)
            fig = figure(
                title="Metric Improvement",
                tools="pan,box_zoom,reset",
                plot_height=300,
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

