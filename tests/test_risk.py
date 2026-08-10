import pandas as pd
import pytest

from risk.crci import CountryRiskIndex


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


def test_country_risk_index_is_missing_when_component_is_missing():
    df = pd.DataFrame(
        {
            "economic_risk": [20.0, None],
            "governance_risk": [40.0, 60.0],
        }
    )

    result = CountryRiskIndex().transform(df)

    assert result["country_risk_index"].iloc[0] == pytest.approx(30.0)
    assert pd.isna(result["country_risk_index"].iloc[1])


def test_country_risk_index_raises_error_for_missing_component():
    df = pd.DataFrame(
        {
            "economic_risk": [20.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="Missing risk components"):
        CountryRiskIndex().transform(df)


def test_country_risk_index_fit_transform_matches_transform():
    df = pd.DataFrame(
        {
            "economic_risk": [20.0, 40.0],
            "governance_risk": [40.0, 60.0],
        }
    )

    transformer = CountryRiskIndex()

    transformed = transformer.transform(df)
    fitted = transformer.fit_transform(df)

    pd.testing.assert_series_equal(
        transformed["country_risk_index"],
        fitted["country_risk_index"],
    )


def test_country_risk_index_calculates_weighted_average():
    df = pd.DataFrame(
        {
            "economic_risk": [20.0, 40.0],
            "governance_risk": [40.0, 60.0],
        }
    )

    result = CountryRiskIndex().transform(df)

    assert result["country_risk_index"].iloc[0] == pytest.approx(30.0)
    assert result["country_risk_index"].iloc[1] == pytest.approx(50.0)