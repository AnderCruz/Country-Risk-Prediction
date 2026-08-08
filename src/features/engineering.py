import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for Machine Learning.
    """

    print("\nCreating features...")

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    df = df.sort_values(
        by=["countryiso3code", "date"]
    )

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    df["gdp_growth"] = (
        df.groupby("countryiso3code")["gdp_per_capita"]
        .pct_change() * 100
    )

    df["population_growth"] = (
        df.groupby("countryiso3code")["population"]
        .pct_change() * 100
    )

    # ------------------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------------------

    df["gdp_lag1"] = (
        df.groupby("countryiso3code")["gdp_per_capita"]
        .shift(1)
    )

    df["inflation_lag1"] = (
        df.groupby("countryiso3code")["inflation"]
        .shift(1)
    )

    df["life_expectancy_lag1"] = (
        df.groupby("countryiso3code")["life_expectancy"]
        .shift(1)
    )

    print("Features created successfully.")

    return df