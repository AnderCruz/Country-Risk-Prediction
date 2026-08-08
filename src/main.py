from config import PROCESSED_DATA_DIR
from data.loader import download_all_datasets, load_all_datasets
from data.merger import merge_datasets
from data.validator import DataValidator
from features.clean_data import clean_dataset
from features.engineering import create_features
from features.eda import generate_eda_report
from models.train import train_model
from models.evaluate import evaluate_model


def main():

    print("=" * 60)
    print("Country Risk Prediction Platform")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1 - Download data
    # -------------------------------------------------------------------------

    print("\n[1/3] Downloading datasets...")

    download_all_datasets()

    # -------------------------------------------------------------------------
    # STEP 2 - Load datasets
    # -------------------------------------------------------------------------

    print("\n[2/3] Loading datasets...")

    datasets = load_all_datasets()

    print(f"{len(datasets)} datasets loaded.")

    # -------------------------------------------------------------------------
    # STEP 3 - Merge datasets
    # -------------------------------------------------------------------------

    print("\n[3/3] Merging datasets...")

    merged = merge_datasets(datasets)

    validator = DataValidator(merged)

    validator.run()

    merged = clean_dataset(merged)

    merged = create_features(merged)

    generate_eda_report(merged)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_DIR / "country_risk_dataset.csv"

    merged.to_csv(output_file, index=False)

    print(f"\nDataset saved: {output_file}")

    print(f"Final shape: {merged.shape}")

    print("\nPipeline completed successfully!")


    model, X_test, y_test, predictions = train_model(merged)

    evaluate_model(y_test, predictions)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()