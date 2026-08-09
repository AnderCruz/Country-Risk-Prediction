from data.loader import (load_all_datasets,)

from data.merger import (merge_datasets,save_dataset,)

from config import (FEATURE_COLUMNS,TARGET_COLUMN,)

from data.validator import DataValidator

from models.baseline import evaluate_naive_risk_baseline

from features.clean_data import clean_dataset
from features.imputer import CountryDataImputer
from features.engineering import create_features
from features.eda import generate_eda_report
from features.target import create_future_risk_target

from risk.crci import CountryRiskIndex
from risk.economic import EconomicRisk
from risk.economic_pca import PCAEconomicRisk
from risk.governance import GovernanceRisk

from models.train import train_model
from models.evaluate import evaluate_model
from models.importance import feature_importance_report
from models.experiments import run_experiments

import mlflow
from ml.tracking import setup_mlflow


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 70)
    print("Country Risk Prediction Platform")
    print("=" * 70)

    setup_mlflow()

    # -------------------------------------------------------------------------
    # STEP 1 - Load datasets
    # -------------------------------------------------------------------------

    print("\n[1/7] Loading datasets...")

    datasets = load_all_datasets()

    print(
        f"{len(datasets)} datasets loaded."
    )

    # -------------------------------------------------------------------------
    # STEP 2 - Merge datasets
    # -------------------------------------------------------------------------

    print("\n[2/7] Merging datasets...")

    merged = merge_datasets(
        datasets
    )

    # -------------------------------------------------------------------------
    # STEP 3 - Validate dataset
    # -------------------------------------------------------------------------

    print("\n[3/7] Validating dataset...")

    validator = DataValidator(
        merged
    )

    validator.run()

    # -------------------------------------------------------------------------
    # STEP 4 - Feature Engineering
    # -------------------------------------------------------------------------

    print("\n[4/7] Feature Engineering...")

    # Clean
    merged = clean_dataset(
        merged
    )

    # Impute macroeconomic data
    imputer = CountryDataImputer()

    merged = imputer.transform(
        merged
    )

    # Create features
    merged = create_features(
        merged
    )

    # -------------------------------------------------------------------------
    # Economic Risk
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Economic Risk..."
    )

    economic_risk = EconomicRisk()

    merged = economic_risk.fit_transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Economic Risk PCA
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Economic Risk PCA..."
    )

    economic_risk_pca = PCAEconomicRisk()

    merged = economic_risk_pca.fit_transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Governance Risk
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Governance Risk..."
    )

    governance_risk = GovernanceRisk()

    merged = governance_risk.fit_transform(
        merged
    )

    print(
        "Governance Risk created successfully."
    )

    # -------------------------------------------------------------------------
    # Country Risk Composite Index
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Country Risk Composite Index..."
    )

    country_risk_index = CountryRiskIndex()

    merged = country_risk_index.fit_transform(
        merged
    )

    print(
        "Country Risk Composite Index created successfully."
    )

    print(
    "\nCreating future Country Risk target..."
    )

    merged = create_future_risk_target(
        merged
    )

    print(
        "Future Country Risk target created successfully."
    )


    # -------------------------------------------------------------------------
    # STEP 5 - Reports
    # -------------------------------------------------------------------------

    print(
        "\n[5/7] Generating reports..."
    )

    generate_eda_report(
        merged
    )

    save_dataset(
        merged
    )

    print(
        f"\nFinal dataset shape: "
        f"{merged.shape}"
    )

    # -------------------------------------------------------------------------
    # STEP 6 - Train
    # -------------------------------------------------------------------------

    print(
        "\n[6/7] Training Machine Learning model..."
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "n_features",
            len(FEATURE_COLUMNS),
        )

        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        model, X_test, y_test, predictions = train_model(
            merged,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
        )

        metrics = evaluate_model(
            y_test,
            predictions,
        )

        feature_importance_report(
            model,
            X_test.columns,
        )

    # -------------------------------------------------------------------------
    # STEP 7 - Experiments
    # -------------------------------------------------------------------------

    print("\nEvaluating naive Country Risk baseline...")

    evaluate_naive_risk_baseline(
        merged
)
    
    print(
        "\n[7/7] Running Experiments..."
    )

    run_experiments(
        merged,
        TARGET_COLUMN,
    )

    print(
        "\nPipeline completed successfully!"
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()