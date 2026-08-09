import mlflow


EXPERIMENT_NAME = "country-risk-prediction"


def setup_mlflow():

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    return mlflow