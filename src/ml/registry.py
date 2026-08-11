import mlflow


REGISTERED_MODEL_NAME = "country-risk-prediction-model"


def register_model(
    model_uri: str,
    model_name: str = REGISTERED_MODEL_NAME,
):
    """
    Register a logged MLflow model in the Model Registry.

    Parameters
    ----------
    model_uri:
        MLflow model URI to register.

    model_name:
        Name of the registered model.

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        Registered model version.
    """

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(
        f"Registered model: "
        f"{registered_model.name}"
    )

    print(
        f"Model version: "
        f"{registered_model.version}"
    )

    return registered_model
