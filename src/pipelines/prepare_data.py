from data.loader import load_all_datasets
from data.merger import merge_datasets, save_dataset

from features.clean_data import clean_dataset
from features.imputer import CountryDataImputer
from features.engineering import create_features
from features.target import create_future_risk_target

from risk.economic import EconomicRisk
from risk.economic_pca import PCAEconomicRisk
from risk.governance import GovernanceRisk
from risk.crci import CountryRiskIndex


def main():

    print("=" * 70)
    print("DVC DATA PREPARATION PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print("\n[1/8] Loading datasets...")

    datasets = load_all_datasets()

    print(
        f"{len(datasets)} datasets loaded."
    )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    print("\n[2/8] Merging datasets...")

    merged = merge_datasets(
        datasets
    )

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    print("\n[3/8] Cleaning dataset...")

    merged = clean_dataset(
        merged
    )

    # -------------------------------------------------------------------------
    # Imputation
    # -------------------------------------------------------------------------

    print("\n[4/8] Imputing macroeconomic data...")

    imputer = CountryDataImputer()

    merged = imputer.transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------

    print("\n[5/8] Creating features...")

    merged = create_features(
        merged
    )

    # -------------------------------------------------------------------------
    # Economic Risk
    # -------------------------------------------------------------------------

    print("\n[6/8] Calculating Economic Risk...")

    economic_risk = EconomicRisk()

    merged = economic_risk.fit_transform(
        merged
    )

    economic_risk_pca = PCAEconomicRisk()

    merged = economic_risk_pca.fit_transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Governance Risk
    # -------------------------------------------------------------------------

    print("\nCalculating Governance Risk...")

    governance_risk = GovernanceRisk()

    merged = governance_risk.fit_transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Country Risk Index
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Country Risk Composite Index..."
    )

    country_risk_index = CountryRiskIndex()

    merged = country_risk_index.fit_transform(
        merged
    )

    # -------------------------------------------------------------------------
    # Future Risk Target
    # -------------------------------------------------------------------------

    print(
        "\nCreating future Country Risk target..."
    )

    merged = create_future_risk_target(
        merged
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    print(
        "\n[7/8] Saving processed dataset..."
    )

    save_dataset(
        merged
    )

    print(
        f"\nFinal dataset shape: "
        f"{merged.shape}"
    )

    print(
        "\n[8/8] Data preparation completed."
    )


if __name__ == "__main__":
    main()