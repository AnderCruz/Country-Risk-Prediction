import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning before Machine Learning.
    """

    print("\nCleaning dataset...")

    # Remove rows without ISO country code
    df = df.dropna(subset=["countryiso3code"])

    # Keep only ISO3 country codes (3 letters)
    df = df[df["countryiso3code"].str.len() == 3]

    # Convert year to integer
    df["date"] = df["date"].astype(int)

    # Sort dataset
    df = df.sort_values(
        by=["countryiso3code", "date"]
    ).reset_index(drop=True)

    print(f"Dataset after cleaning: {df.shape}")

    return df