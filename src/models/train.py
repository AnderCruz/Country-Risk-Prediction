from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature
from ml.registry import register_model

# =============================================================================
# PATHS
# =============================================================================

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
):
    """
    Train a Random Forest regression model using a temporal split.

    The trained model is:
        1. Saved locally as a .pkl file
        2. Logged to MLflow as a model artifact
        3. Optionally registered in the MLflow Model Registry
           when the active MLflow run contains the appropriate tags.
    """

    print("\nTraining Random Forest")

    # =========================================================================
    # PREPARE DATA
    # =========================================================================

    dataset = df.copy()

    required_columns = (
        feature_columns
        + [target_column]
        + ["date"]
    )

    dataset = dataset.dropna(
        subset=required_columns
    )

    dataset = dataset.sort_values(
        "date"
    )

    print(
        f"Training samples: {len(dataset)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print("\nFeatures used:")

    for feature in feature_columns:

        print(
            f"- {feature}"
        )

    # =========================================================================
    # TEMPORAL SPLIT
    # =========================================================================

    years = sorted(
        dataset["date"].unique()
    )

    if len(years) < 2:

        raise ValueError(
            "Not enough years available "
            "for temporal split."
        )

    split_index = int(
        len(years) * 0.80
    )

    train_years = years[:split_index]
    test_years = years[split_index:]

    train_start = train_years[0]
    train_end = train_years[-1]

    test_start = test_years[0]
    test_end = test_years[-1]

    print("\nTemporal Split")

    print(
        f"Train period: "
        f"{train_start} - {train_end}"
    )

    print(
        f"Test period : "
        f"{test_start} - {test_end}"
    )

    train_data = dataset[
        dataset["date"].isin(train_years)
    ]

    test_data = dataset[
        dataset["date"].isin(test_years)
    ]

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        target_column
    ]

    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        target_column
    ]

    print(
        f"\nTrain size : {len(X_train)}"
    )

    print(
        f"Test size  : {len(X_test)}"
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    # =========================================================================
    # SAVE MODEL LOCALLY
    # =========================================================================

    model_path = (
        MODEL_DIR /
        "random_forest.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved: "
        f"{model_path}"
    )

    # =========================================================================
    # LOG MODEL TO MLFLOW
    # =========================================================================

    if mlflow.active_run() is not None:

        print(
            "\nLogging model to MLflow..."
        )

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        input_example = X_train.head(3)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest",
            signature=signature,
            input_example=input_example,
        )

        print(
            "Model logged to MLflow successfully."
        )

        # =====================================================================
        # MODEL REGISTRY
        # =====================================================================

        run = mlflow.get_run(
            mlflow.active_run().info.run_id
        )

        if run.data.tags.get("register_model") == "true":

            registered_model_name = run.data.tags.get(
                "registered_model_name",
                "country-risk-prediction-model",
            )

            print(
                "\nRegistering model..."
            )

            register_model(
                model_uri=model_info.model_uri,
                model_name=registered_model_name,
            )


    else:

        print(
            "\nWarning: no active MLflow run."
        )

    # =========================================================================
    # RETURN
    # =========================================================================

    return (
        model,
        X_test,
        y_test,
        predictions,
    )
