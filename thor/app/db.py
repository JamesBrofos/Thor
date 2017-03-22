"""
Commands to run:
python db.py db init
python db.py db migrate
python db.py db upgrade head
"""

from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from website import app, db


# Initialize extension.
migrate = Migrate(app, db, compare_type=True)

# Provide command line extensions.
manager = Manager(app)
manager.add_command("db", MigrateCommand)


if __name__ == "__main__":
    manager.run()
