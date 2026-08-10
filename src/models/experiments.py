from pathlib import Path

import mlflow
import pandas as pd

from models.train import train_model
from models.evaluate import evaluate_model


# =============================================================================
# PATHS
# =============================================================================

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiments(
    df,
    target_column,
):

    experiments = [

        {
            "name": "Baseline",
            "features": [
                "gdp_per_capita",
                "inflation",
                "life_expectancy",
                "population",
                "population_growth",
                "unemployment",
                "exports",
            ],
        },

        {
            "name": "Baseline + Lag",
            "features": [
                "gdp_per_capita",
                "inflation",
                "life_expectancy",
                "population",
                "population_growth",
                "unemployment",
                "exports",
                "gdp_lag1",
                "inflation_lag1",
                "life_expectancy_lag1",
            ],
        },

        {
            "name": "Baseline + Lag + Economic Risk",
            "features": [
                "gdp_per_capita",
                "inflation",
                "life_expectancy",
                "population",
                "population_growth",
                "unemployment",
                "exports",
                "gdp_lag1",
                "inflation_lag1",
                "life_expectancy_lag1",
                "economic_risk",
            ],
        },

        {
            "name": "Baseline + Lag + Economic Risk PCA",
            "features": [
                "gdp_per_capita",
                "inflation",
                "life_expectancy",
                "population",
                "population_growth",
                "unemployment",
                "exports",
                "gdp_lag1",
                "inflation_lag1",
                "life_expectancy_lag1",
                "economic_risk_pca",
            ],
        },

        {
            "name": "Full Risk Model",
            "features": [
                "gdp_per_capita",
                "inflation",
                "life_expectancy",
                "population",
                "population_growth",
                "unemployment",
                "exports",
                "gdp_lag1",
                "inflation_lag1",
                "life_expectancy_lag1",
                "economic_risk",
                "governance_risk",
            ],
        },
    ]

    results = []

    for experiment in experiments:

        print(
            "\n"
            + "=" * 70
        )

        print(
            experiment["name"]
        )

        print(
            "=" * 70
        )

        # ---------------------------------------------------------------------
        # MLflow Run
        # ---------------------------------------------------------------------

        with mlflow.start_run(
            run_name=experiment["name"]
        ):

            features = experiment["features"]

            # -----------------------------------------------------------------
            # Parameters
            # -----------------------------------------------------------------

            mlflow.log_param(
                "experiment",
                experiment["name"],
            )

            mlflow.log_param(
                "model_type",
                "RandomForestRegressor",
            )

            mlflow.log_param(
                "n_features",
                len(features),
            )

            mlflow.log_param(
                "target",
                target_column,
            )

            mlflow.log_param(
                "features",
                ", ".join(features),
            )

            # -----------------------------------------------------------------
            # Train
            # -----------------------------------------------------------------

            model, X_test, y_test, predictions = train_model(
                df,
                features,
                target_column,
            )

            # -----------------------------------------------------------------
            # Evaluate
            # -----------------------------------------------------------------

            metrics = evaluate_model(
                y_test,
                predictions,
            )

            # -----------------------------------------------------------------
            # Results
            # -----------------------------------------------------------------

            results.append(
                {
                    "experiment": experiment["name"],
                    "n_features": len(features),
                    "mae": round(
                        metrics["mae"],
                        4,
                    ),
                    "rmse": round(
                        metrics["rmse"],
                        4,
                    ),
                    "r2": round(
                        metrics["r2"],
                        4,
                    ),
                }
            )

    # =========================================================================
    # SAVE EXPERIMENT RESULTS
    # =========================================================================

    results = pd.DataFrame(
        results
    )

    output = (
        REPORT_DIR /
        "experiments.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        results
    )

    print(
        f"\nSaved: {output}"
    )

    return results