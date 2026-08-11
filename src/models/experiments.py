from pathlib import Path

import mlflow
import pandas as pd

from models.train import train_model
from models.evaluate import evaluate_model
from models.importance import feature_importance_report
from ml.validation import validate_model
from ml.performance import (
    analyse_model_performance,
    build_performance_dataset,
)


# =============================================================================
# PATHS
# =============================================================================

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# MLFLOW MODEL REGISTRY
# =============================================================================

REGISTERED_MODEL_NAME = "country-risk-prediction-model"


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiments(
    df,
    target_column,
    baseline_metrics=None,
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
            # Model Registry configuration
            # -----------------------------------------------------------------

            if experiment["name"] == "Full Risk Model":

                mlflow.set_tag(
                    "register_model",
                    "true",
                )

                mlflow.set_tag(
                    "registered_model_name",
                    REGISTERED_MODEL_NAME,
                )

                print(
                    "\nRegistering Full Risk Model..."
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
            # Model Validation
            # -----------------------------------------------------------------

            validation_result = None

            if (
                experiment["name"] == "Full Risk Model"
                and baseline_metrics is not None
            ):

                validation_result = validate_model(
                    metrics,
                    baseline_metrics,
                )

                mlflow.log_metric(
                    "baseline_mae",
                    baseline_metrics["mae"],
                )

                mlflow.log_metric(
                    "baseline_rmse",
                    baseline_metrics["rmse"],
                )

                mlflow.log_metric(
                    "baseline_r2",
                    baseline_metrics["r2"],
                )

                mlflow.log_metric(
                    "mae_improvement",
                    validation_result["mae_improvement"],
                )

                mlflow.log_metric(
                    "rmse_improvement",
                    validation_result["rmse_improvement"],
                )

                mlflow.log_metric(
                    "r2_improvement",
                    validation_result["r2_improvement"],
                )

                mlflow.set_tag(
                    "validation_status",
                    "passed"
                    if validation_result["passed"]
                    else "failed",
                )

                print(
                    "\nModel Validation"
                )

                print(
                    f"Status: "
                    f"{'PASSED' if validation_result['passed'] else 'FAILED'}"
                )

                print(
                    f"MAE improvement : "
                    f"{validation_result['mae_improvement']:.4f}"
                )

                print(
                    f"RMSE improvement: "
                    f"{validation_result['rmse_improvement']:.4f}"
                )

                print(
                    f"R² improvement  : "
                    f"{validation_result['r2_improvement']:.4f}"
                )

            feature_importance_report(
                model,
                X_test.columns,
            )

            #-----------------------------------------------------------------
            # Performance Analysis
            # -----------------------------------------------------------------

            performance_available = (
                {"country", "date"}.issubset(df.columns)
                and len(X_test) == len(y_test)
                and len(y_test) == len(predictions)
                and len(X_test) > 0
            )

            if performance_available:

                performance_data = build_performance_dataset(
                    X_test,
                    y_test,
                    predictions,
                    df,
                )

                performance = analyse_model_performance(
                    performance_data,
                    "actual",
                    "prediction",
                )

                global_performance = performance["global"]
                yearly_performance = performance["yearly"]
                country_performance = performance["country"]

                # -------------------------------------------------------------
                # Save Performance Reports
                # -------------------------------------------------------------

                experiment_slug = (
                    experiment["name"]
                    .lower()
                    .replace(" ", "_")
                )

                yearly_output = (
                    REPORT_DIR /
                    f"performance_by_year_{experiment_slug}.csv"
                )

                country_output = (
                    REPORT_DIR /
                    f"performance_by_country_{experiment_slug}.csv"
                )

                yearly_performance.to_csv(
                    yearly_output,
                    index=False,
                )

                country_performance.to_csv(
                    country_output,
                    index=False,
                )

                print(
                    f"Saved: {yearly_output}"
                )

                print(
                    f"Saved: {country_output}"
                )

                # -------------------------------------------------------------
                # MLflow Performance Metrics
                # -------------------------------------------------------------

                mlflow.log_metric(
                    "performance_mae",
                    global_performance["mae"],
                )

                mlflow.log_metric(
                    "performance_rmse",
                    global_performance["rmse"],
                )

                mlflow.log_metric(
                    "performance_r2",
                    global_performance["r2"],
                )

            else:

                print(
                    "\nPerformance analysis skipped: "
                    "test data is not suitable."
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