from .. import db


class AcquisitionFunction(db.Model):
    """Acquisition Function Database Class"""
    __tablename__ = "acquisitions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"))
    name = db.Column(db.String(80))
    params = db.Column(db.Text)

    def __init__(self, name):
        self.name = name

    @classmethod
    def from_json(cls, json):
        return cls(json.get("acq_func", "expected_improvement"))
