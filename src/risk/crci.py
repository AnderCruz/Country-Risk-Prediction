import pandas as pd


class CountryRiskIndex:
    """
    Country Risk Composite Index (CRCI).

    Version 1.0

    Combines the available Country Risk components:

    - Economic Risk
    - Governance Risk

    Lower values indicate lower risk.
    Higher values indicate higher risk.
    """

    def __init__(self):

        self.components = [
            "economic_risk",
            "governance_risk",
        ]

        self.weights = {
            "economic_risk": 0.50,
            "governance_risk": 0.50,
        }

    def transform(
        self,
        df: pd.DataFrame,
    ):

        data = df.copy()

        # -------------------------------------------------------------
        # Check required columns
        # -------------------------------------------------------------

        missing = [
            column
            for column in self.components
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                f"Missing risk components: {missing}"
            )

        # -------------------------------------------------------------
        # Calculate CRCI
        #
        # Only calculate when all components are available.
        # -------------------------------------------------------------

        mask = data[
            self.components
        ].notna().all(axis=1)

        data["country_risk_index"] = pd.NA

        if mask.any():

            risk_score = 0

            for component, weight in self.weights.items():

                risk_score += (
                    data.loc[
                        mask,
                        component,
                    ] * weight
                )

            data.loc[
                mask,
                "country_risk_index",
            ] = risk_score

        data["country_risk_index"] = pd.to_numeric(
            data["country_risk_index"],
            errors="coerce",
        )

        # -------------------------------------------------------------
        # Report
        # -------------------------------------------------------------

        print("\nCountry Risk Composite Index")
        print("-" * 40)

        print(
            data[
                "country_risk_index"
            ].describe()
        )

        print(
            "\nValid CRCI observations:",
            data[
                "country_risk_index"
            ].notna().sum(),
        )

        return data

    def fit_transform(
        self,
        df: pd.DataFrame,
    ):

        return self.transform(df)