from wtforms.validators import ValidationError
from ..models import User


class Unique(object):
    def __init__(self, model, field, message=u"This element already exists."):
        self.model = model
        self.field = field
        self.message = message

    def __call__(self, form, field):
        check = self.model.query.filter(
            self.field == field.data
        ).first()

        if check:
            raise ValidationError(self.message)

class Exists(object):
    def __init__(self, model, field, message=u"This element does not exist."):
        self.model = model
        self.field = field
        self.message = message

    def __call__(self, form, field):
        check = self.model.query.filter(
            self.field == field.data
        ).first()

        if check is None:
            raise ValidationError(self.message)


class Active(object):
    def __call__(self, form, field):
        user = User.query.filter_by(username=form.username.data).first()
        if user is not None:
            if not user.is_active:
                raise ValidationError("This account is inactive.")


class CorrectPassword(object):
    def __call__(self, form, field):
        user = User.query.filter_by(username=form.username.data).first()
        if (
                user is not None and user.is_active
                and not user.is_correct_password(field.data)
        ):
            raise ValidationError("Incorrect password.")
