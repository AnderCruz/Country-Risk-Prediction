import mlflow

from ml.registry import REGISTERED_MODEL_NAME


# =============================================================================
# MODEL PROMOTION
# =============================================================================

CHAMPION_ALIAS = "champion"


def promote_model(
    model_name: str = REGISTERED_MODEL_NAME,
    model_version: str | int | None = None,
):
    """
    Promote a validated MLflow model version to the 'champion' alias.

    A model can only be promoted when the source MLflow run contains:

        validation_status == "passed"

    Parameters
    ----------
    model_name:
        Name of the registered MLflow model.

    model_version:
        Version of the registered model to promote.

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        Promoted model version.

    Raises
    ------
    ValueError
        If model_version is not provided.

    ValueError
        If the registered model version has no associated run.

    ValueError
        If model validation did not pass.
    """

    # =========================================================================
    # VALIDATE INPUT
    # =========================================================================

    if model_version is None:

        raise ValueError(
            "model_version must be provided."
        )

    # =========================================================================
    # MLflow CLIENT
    # =========================================================================

    client = mlflow.MlflowClient()

    # =========================================================================
    # GET MODEL VERSION
    # =========================================================================

    registered_version = client.get_model_version(
        name=model_name,
        version=str(model_version),
    )

    # =========================================================================
    # GET SOURCE RUN
    # =========================================================================

    run_id = registered_version.run_id

    if run_id is None:

        raise ValueError(
            "Registered model version has no associated run."
        )

    run = client.get_run(
        run_id
    )

    # =========================================================================
    # VALIDATION STATUS
    # =========================================================================

    validation_status = run.data.tags.get(
        "validation_status"
    )

    if validation_status != "passed":

        raise ValueError(
            "Model validation failed. "
            f"Current validation_status: "
            f"{validation_status!r}"
        )

    # =========================================================================
    # PROMOTE TO CHAMPION
    # =========================================================================

    client.set_registered_model_alias(
        name=model_name,
        alias=CHAMPION_ALIAS,
        version=str(
            registered_version.version
        ),
    )

    print(
        f"Model '{model_name}' version "
        f"{registered_version.version} "
        f"promoted to '{CHAMPION_ALIAS}'."
    )

    return registered_version