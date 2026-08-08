import pandas as pd

from config import (
    MERGE_COLUMNS,
    PROCESSED_DATA_DIR,
)


# =============================================================================
# MERGE
# =============================================================================

def merge_datasets(datasets: dict) -> pd.DataFrame:
    """
    Merge every dataframe using the configured merge columns.

    Parameters
    ----------
    datasets : dict
        Dictionary containing all loaded datasets.

    Returns
    -------
    pd.DataFrame
        Final merged dataframe.
    """

    print("\nMerging datasets...")

    dataset_names = list(datasets.keys())

    merged = datasets[dataset_names[0]]

    for dataset_name in dataset_names[1:]:

        print(f"Merging {dataset_name}")

        merged = pd.merge(
            merged,
            datasets[dataset_name],
            on=MERGE_COLUMNS,
            how="outer",
        )

    merged = merged.sort_values(
        by=["countryiso3code", "date"]
    ).reset_index(drop=True)

    return merged


# =============================================================================
# SAVE
# =============================================================================

def save_dataset(df: pd.DataFrame) -> None:
    """
    Save final processed dataset.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DATA_DIR /
        "country_risk_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nDataset saved: {output_file}")