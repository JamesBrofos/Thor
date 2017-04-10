import numpy as np
import datetime as dt
from sqlalchemy.ext.hybrid import hybrid_property
from .observation import Observation
from .dimension import Dimension
from .. import db


class Experiment(db.Model):
    """Experiment Class"""
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    name = db.Column(db.String(80))
    date = db.Column(db.DateTime)
    # acq_func = db.Column(db.String(80))
    acq_func = db.relationship(
        "AcquisitionFunction",
        uselist=False,
        backref="experiments",
        cascade="all, delete-orphan"
    )
    dimensions = db.relationship(
        "Dimension",
        lazy="dynamic",
        backref="experiments",
        cascade="all, delete-orphan"
    )
    observations = db.relationship(
        "Observation",
        lazy="dynamic",
        backref="experiments",
        cascade="all, delete-orphan"
    )

    def __init__(self, name, date):
        self.name = name
        self.date = date

    def to_dict(self):
        dims = [d.to_dict() for d in self.dimensions.all()]
        return {
            "name": self.name,
            "date": self.date,
            "dimensions": dims,
            "id": self.id
        }

    @classmethod
    def from_json(cls, json):
        date = dt.datetime.today()
        name = json["name"]
        dims = json["dimensions"]
        e = cls(name, date)
        for d in dims:
            e.dimensions.append(Dimension.from_json(d))

        return e

    @hybrid_property
    def percent_improvement(self):
        first = self.observations.order_by("date").first()
        best = self.maximal_observation
        return np.abs((best.y - first.y) / first.y) * 100.

    @hybrid_property
    def maximal_observation(self):
        return self.observations.filter_by(
            pending=False
        ).order_by("y desc").first()

