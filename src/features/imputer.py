import pandas as pd
import numpy as np


class CountryDataImputer:
    """
    Impute missing values for macroeconomic time series.

    Strategy
    --------
    1. Sort by country and year
    2. Linear interpolation
    3. Forward fill
    4. Backward fill
    5. Median (last resort)
    """

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        before = data.isna().sum().sum()

        data = data.sort_values(
            ["country", "date"]
        )

        numeric_columns = data.select_dtypes(
            include=np.number
        ).columns.tolist()

        if "date" in numeric_columns:
            numeric_columns.remove("date")

        for column in numeric_columns:

            data[column] = (
                data.groupby("country")[column]
                .transform(
                    lambda x: x.interpolate(
                        limit_direction="both"
                    )
                )
            )

            data[column] = (
                data.groupby("country")[column]
                .transform(lambda x: x.ffill())
            )

            data[column] = (
                data.groupby("country")[column]
                .transform(lambda x: x.bfill())
            )

            data[column] = data[column].fillna(
                data[column].median()
            )

        after = data.isna().sum().sum()

        print("\nImputation Report")
        print("----------------------------")
        print(f"Missing values before : {before}")
        print(f"Missing values after  : {after}")

        return data