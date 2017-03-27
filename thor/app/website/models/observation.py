import json
from sqlalchemy.ext.hybrid import hybrid_property
from .. import db


class Observation(db.Model):
    """Observation Class"""
    __tablename__ = "observations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"))
    x = db.Column(db.Text)
    y = db.Column(db.Float, default=None)
    pending = db.Column(db.Boolean, default=True)
    date = db.Column(db.DateTime)

    def __init__(self, x, date):
        self.x = x
        self.date = date

    @hybrid_property
    def config(self):
        return json.loads(self.x)

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "x": self.x,
            "y": self.y,
            "pending": self.pending
        }
