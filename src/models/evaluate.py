import mlflow

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    y_true,
    y_pred,
):

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    # -------------------------------------------------------------------------
    # MLflow metrics
    # -------------------------------------------------------------------------

    if mlflow.active_run() is not None:

        mlflow.log_metric(
            "mae",
            mae,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.log_metric(
            "r2",
            r2,
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }