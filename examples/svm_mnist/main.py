from thor_client import ThorClient
from evaluate import evaluate

# Authentication token.
auth_token = "$2b$12$xSc4ji2UJy96N.vPOlM/iOR.OBJ3BKmhgSU60/8Asi8YfI6pPbB8."


# Create experiment.
ec = ThorClient(auth_token)
name = "MNIST SVM"
try:
    # Create space.
    dims = [
        {"name": "gamma", "dim_type": "linear", "low": 1e-5, "high": 1e-2},
        {"name": "C", "dim_type": "linear", "low": 0.5, "high": 1.5}
    ]
    exp = ec.create_experiment(
        name,
        dims,
        "hedge"
    )
except ValueError:
    exp = ec.experiment_for_name(name)

# Baseline.
print(evaluate(1., 0.001))

for i in range(100):
    rec = exp.create_recommendation()
    x = rec.config
    val = evaluate(x["C"], x["gamma"])
    rec.submit_recommendation(val)
    print((i, x, val))
