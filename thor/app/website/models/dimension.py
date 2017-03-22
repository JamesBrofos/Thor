from .. import db


class Dimension(db.Model):
    """Dimension Class"""
    __tablename__ = "dimensions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"))
    name = db.Column(db.String(80))
    dim_type = db.Column(db.String(80))
    low = db.Column(db.Float)
    high = db.Column(db.Float)

    def __init__(self, name, dim_type, low, high):
        self.name = name
        self.dim_type = dim_type
        self.low = low
        self.high = high

    def to_dict(self):
        return {
            "name": self.name,
            "dim_type": self.dim_type,
            "low": self.low,
            "high": self.high
        }
