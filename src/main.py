from data.loader import (
    download_all_datasets,
    load_all_datasets,
)

from data.merger import (
    merge_datasets,
    save_dataset,
)

from data.validator import DataValidator

from features.clean_data import clean_dataset
from features.engineering import create_features
from features.eda import generate_eda_report
from features.crci import CountryRiskIndex

from models.train import train_model
from models.evaluate import evaluate_model
from features.imputer import CountryDataImputer
from models.importance import feature_importance_report


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 70)
    print("Country Risk Prediction Platform")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1 - Download datasets
    # -------------------------------------------------------------------------

    print("\n[1/7] Downloading datasets...")

    download_all_datasets()

    # -------------------------------------------------------------------------
    # STEP 2 - Load datasets
    # -------------------------------------------------------------------------

    print("\n[2/7] Loading datasets...")

    datasets = load_all_datasets()

    print(f"{len(datasets)} datasets loaded.")

    # -------------------------------------------------------------------------
    # STEP 3 - Merge datasets
    # -------------------------------------------------------------------------

    print("\n[3/7] Merging datasets...")

    merged = merge_datasets(datasets)

    # -------------------------------------------------------------------------
    # STEP 4 - Validate dataset
    # -------------------------------------------------------------------------

    print("\n[4/7] Validating dataset...")

    validator = DataValidator(merged)

    validator.run()

    # -------------------------------------------------------------------------
    # STEP 5 - Feature Engineering
    # -------------------------------------------------------------------------

    print("\n[5/7] Feature Engineering...")

    merged = clean_dataset(merged)

    imputer = CountryDataImputer()

    merged = imputer.transform(merged)

    merged = create_features(merged)

    print("\nCalculating Economic Score...")

    crci = CountryRiskIndex()

    merged = crci.fit_transform(merged)



    print("Economic Score created successfully.")


    # -------------------------------------------------------------------------
    # STEP 6 - Generate Reports
    # -------------------------------------------------------------------------

    print("\n[6/7] Generating reports...")

    generate_eda_report(merged)

    save_dataset(merged)

    print(f"\nFinal dataset shape: {merged.shape}")

    # -------------------------------------------------------------------------
    # STEP 7 - Train Model
    # -------------------------------------------------------------------------

    print("\n[7/7] Training Machine Learning model...")

    model, X_test, y_test, predictions = train_model(merged)

    evaluate_model(y_test, predictions)

    feature_importance_report(
        model,
        X_test.columns
    )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()