import os

import mlflow


EXPERIMENT_NAME = "country-risk-prediction"


def setup_mlflow():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:

        mlflow.set_tracking_uri(
            tracking_uri
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    return mlflow