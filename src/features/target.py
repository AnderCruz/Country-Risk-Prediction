import pandas as pd


def create_future_risk_target(
    df: pd.DataFrame,
    target_column: str = "country_risk_index",
) -> pd.DataFrame:
    """
    Create a one-year-ahead Country Risk target.

    The target for year t is the Country Risk Index
    observed in year t+1.
    """

    data = df.copy()

    data = data.sort_values(
        ["country", "date"]
    )

    data["future_country_risk"] = (
        data.groupby("country")[target_column]
        .shift(-1)
    )

    print("\nFuture Risk Target")
    print("-" * 40)

    print(
        "Target column: future_country_risk"
    )

    print(
        f"Valid observations: "
        f"{data['future_country_risk'].notna().sum()}"
    )

    print(
        f"Missing observations: "
        f"{data['future_country_risk'].isna().sum()}"
    )

    return data