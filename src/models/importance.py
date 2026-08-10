import mlflow
import pandas as pd
from pathlib import Path

REPORTS_DIR = Path("reports")


def feature_importance_report(model, feature_names):

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    REPORTS_DIR.mkdir(
        exist_ok=True
    )

    filepath = REPORTS_DIR / "feature_importance.csv"

    importance.to_csv(
        filepath,
        index=False,
    )

    # Log artifact to MLflow
    if mlflow.active_run() is not None:
        mlflow.log_artifact(
            str(filepath),
            artifact_path="reports",
        )

    print("\nFeature Importance")
    print("----------------------------")

    print(importance)

    print(f"\nSaved: {filepath}")

    return importance