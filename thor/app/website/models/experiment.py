from .. import db


class Experiment(db.Model):
    """Experiment Class"""
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    name = db.Column(db.String(80))
    date = db.Column(db.DateTime)
    dimensions = db.relationship(
        "Dimension",
        lazy="dynamic",
        backref="experiments",
    )
    observations = db.relationship(
        "Observation",
        lazy="dynamic",
        backref="experiments"
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
