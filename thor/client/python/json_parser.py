import json
from flask_api import status


def json_parser(result, auth_token, cls=None):
    json_data = json.loads(result.text)
    if result.status_code == status.HTTP_400_BAD_REQUEST:
        raise ValueError(json_data["error"])
    else:
        if cls:
            if isinstance(json_data, list):
                return [cls.from_dict(d, auth_token) for d in json_data]
            else:
                return cls.from_dict(json_data, auth_token)
