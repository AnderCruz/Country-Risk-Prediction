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

    Governance indicators are NOT imputed.
    """

    def __init__(self):

        # -------------------------------------------------------------
        # Macroeconomic variables that can be imputed
        # -------------------------------------------------------------

        self.imputation_columns = [
            "exports",
            "gdp_growth",
            "gdp_per_capita",
            "inflation",
            "life_expectancy",
            "population",
            "population_growth",
            "unemployment",
        ]

        # -------------------------------------------------------------
        # WGI variables
        #
        # These values must remain missing when no observation exists.
        # We do NOT invent governance data for years without WGI data.
        # -------------------------------------------------------------

        self.governance_columns = [
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_corruption",
        ]

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = df.copy()

        before = data.isna().sum().sum()

        # -------------------------------------------------------------
        # Sort
        # -------------------------------------------------------------

        data = data.sort_values(
            ["country", "date"]
        )

        # -------------------------------------------------------------
        # Check which columns actually exist
        # -------------------------------------------------------------

        imputation_columns = [
            column
            for column in self.imputation_columns
            if column in data.columns
        ]

        governance_columns = [
            column
            for column in self.governance_columns
            if column in data.columns
        ]

        # -------------------------------------------------------------
        # IMPUTE MACROECONOMIC DATA
        # -------------------------------------------------------------

        for column in imputation_columns:

            # Linear interpolation
            data[column] = (
                data.groupby("country")[column]
                .transform(
                    lambda x: x.interpolate(
                        limit_direction="both"
                    )
                )
            )

            # Forward fill
            data[column] = (
                data.groupby("country")[column]
                .transform(
                    lambda x: x.ffill()
                )
            )

            # Backward fill
            data[column] = (
                data.groupby("country")[column]
                .transform(
                    lambda x: x.bfill()
                )
            )

            # Median as last resort
            data[column] = data[column].fillna(
                data[column].median()
            )

        after = data.isna().sum().sum()

        # -------------------------------------------------------------
        # REPORT
        # -------------------------------------------------------------

        print("\nImputation Report")
        print("----------------------------")

        print(
            f"Missing values before : {before}"
        )

        print(
            f"Missing values after  : {after}"
        )

        # -------------------------------------------------------------
        # GOVERNANCE REPORT
        # -------------------------------------------------------------

        if governance_columns:

            print(
                "\nGovernance indicators "
                "were NOT imputed."
            )

            print(
                "Missing WGI values:"
            )

            print(
                data[governance_columns]
                .isna()
                .sum()
            )

        return data