import pandas as pd
import pytest

from risk.crci import CountryRiskIndex
from risk.economic import EconomicRisk
from risk.governance import GovernanceRisk
from risk.economic_pca import PCAEconomicRisk


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


def test_economic_risk_creates_risk_column():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 20000.0],
            "inflation": [2.0, 4.0],
            "unemployment": [5.0, 10.0],
            "life_expectancy": [70.0, 80.0],
            "exports": [100.0, 200.0],
        }
    )

    result = EconomicRisk().fit_transform(df)

    assert "economic_risk" in result.columns
    assert result["economic_risk"].notna().all()


def test_economic_risk_applies_feature_weights():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 20000.0],
            "inflation": [2.0, 4.0],
            "unemployment": [5.0, 10.0],
            "life_expectancy": [70.0, 80.0],
            "exports": [100.0, 200.0],
        }
    )

    result = EconomicRisk().fit_transform(df)

    assert result["economic_risk"].iloc[0] == pytest.approx(-0.10)
    assert result["economic_risk"].iloc[1] == pytest.approx(0.10)


def test_economic_risk_fit_initializes_scaler():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 20000.0],
            "inflation": [2.0, 4.0],
            "unemployment": [5.0, 10.0],
            "life_expectancy": [70.0, 80.0],
            "exports": [100.0, 200.0],
        }
    )

    model = EconomicRisk()

    model.fit(df)

    assert hasattr(model.scaler, "mean_")
    assert len(model.scaler.mean_) == 5


def test_governance_risk_creates_risk_column():
    df = pd.DataFrame(
        {
            "voice_accountability": [1.0, 2.0],
            "political_stability": [1.0, 2.0],
            "government_effectiveness": [1.0, 2.0],
            "regulatory_quality": [1.0, 2.0],
            "rule_of_law": [1.0, 2.0],
            "control_corruption": [1.0, 2.0],
        }
    )

    result = GovernanceRisk().fit_transform(df)

    assert "governance_risk" in result.columns
    assert result["governance_risk"].notna().all()


def test_governance_risk_is_inverted():
    df = pd.DataFrame(
        {
            "voice_accountability": [1.0, 2.0],
            "political_stability": [1.0, 2.0],
            "government_effectiveness": [1.0, 2.0],
            "regulatory_quality": [1.0, 2.0],
            "rule_of_law": [1.0, 2.0],
            "control_corruption": [1.0, 2.0],
        }
    )

    result = GovernanceRisk().fit_transform(df)

    assert result["governance_risk"].iloc[0] == pytest.approx(1.0)
    assert result["governance_risk"].iloc[1] == pytest.approx(-1.0)


def test_governance_risk_is_missing_when_wgi_is_incomplete():
    df = pd.DataFrame(
        {
            "voice_accountability": [1.0, 2.0],
            "political_stability": [1.0, 2.0],
            "government_effectiveness": [1.0, None],
            "regulatory_quality": [1.0, 2.0],
            "rule_of_law": [1.0, 2.0],
            "control_corruption": [1.0, 2.0],
        }
    )

    result = GovernanceRisk().fit_transform(df)

    assert result["governance_risk"].notna().sum() == 1
    assert pd.isna(result["governance_risk"].iloc[1])


def test_pca_economic_risk_creates_risk_column():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 15000.0, 20000.0],
            "inflation": [2.0, 3.0, 4.0],
            "unemployment": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 75.0, 80.0],
            "exports": [100.0, 150.0, 200.0],
        }
    )

    result = PCAEconomicRisk().fit_transform(df)

    assert "economic_risk_pca" in result.columns
    assert result["economic_risk_pca"].notna().all()


def test_pca_economic_risk_uses_one_component():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 15000.0, 20000.0],
            "inflation": [2.0, 3.0, 4.0],
            "unemployment": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 75.0, 80.0],
            "exports": [100.0, 150.0, 200.0],
        }
    )

    model = PCAEconomicRisk()

    model.fit(df)

    assert model.pca.n_components_ == 1
    assert model.pca.components_.shape == (1, 5)


def test_pca_economic_risk_preserves_number_of_rows():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 15000.0, 20000.0],
            "inflation": [2.0, 3.0, 4.0],
            "unemployment": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 75.0, 80.0],
            "exports": [100.0, 150.0, 200.0],
        }
    )

    model = PCAEconomicRisk()

    result = model.fit_transform(df)

    assert len(result) == len(df)


def test_pca_economic_risk_fits_scaler_and_pca():
    df = pd.DataFrame(
        {
            "gdp_per_capita": [10000.0, 15000.0, 20000.0],
            "inflation": [2.0, 3.0, 4.0],
            "unemployment": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 75.0, 80.0],
            "exports": [100.0, 150.0, 200.0],
        }
    )

    model = PCAEconomicRisk()

    model.fit(df)

    assert hasattr(model.scaler, "mean_")
    assert hasattr(model.pca, "components_")