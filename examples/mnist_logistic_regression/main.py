from mnist_trainer import mnist_trainer
from thor.client.python import ThorClient


# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."

# Create space.
dims = [
    {"name": "batch_size", "dim_type": "integer", "low": 20, "high": 2000},
    {"name": "epochs", "dim_type": "integer", "low": 5, "high": 2000},
    {"name": "lr", "dim_type": "logarithmic", "low": 1e-6, "high": 1.},
    {"name": "l2_weight", "dim_type": "linear", "low": 0., "high": 1.}
]

# Create experiment.
ec = ThorClient(auth_token)
try:
    exp = ec.create_experiment("MNIST Logistic Regression", dims)
except ValueError:
    exp = ec.experiment_for_name("MNIST Logistic Regression")

for i in range(100):
    rec = exp.create_recommendation()
    x = rec.config
    val = mnist_trainer(
        x["batch_size"],
        x["epochs"],
        x["lr"],
        x["l2_weight"]
    )
    rec.submit_recommendation(val)
    print((i, x, val))
