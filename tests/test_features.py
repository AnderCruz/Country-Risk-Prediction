import pandas as pd
import pytest

from features.clean_data import clean_dataset
from features.target import create_future_risk_target
from features.engineering import create_features
from features.imputer import CountryDataImputer


def test_clean_dataset_removes_rows_without_country_code():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", None, "ARG"],
            "date": ["2022", "2022", "2021"],
            "value": [10, 20, 30],
        }
    )

    result = clean_dataset(df)

    assert len(result) == 2
    assert result["countryiso3code"].isna().sum() == 0


def test_clean_dataset_keeps_only_valid_iso3_codes():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "BR", "ARG", "BRAZ"],
            "date": ["2022", "2022", "2021", "2020"],
            "value": [10, 20, 30, 40],
        }
    )

    result = clean_dataset(df)

    assert result["countryiso3code"].tolist() == ["ARG", "BRA"]


def test_clean_dataset_converts_date_to_integer():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "ARG"],
            "date": ["2022", "2021"],
            "value": [10, 20],
        }
    )

    result = clean_dataset(df)

    assert result["date"].dtype.kind in "iu"


def test_clean_dataset_sorts_by_country_and_date():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "ARG", "BRA", "ARG"],
            "date": [2022, 2022, 2020, 2020],
            "value": [10, 20, 30, 40],
        }
    )

    result = clean_dataset(df)

    expected = [
        ("ARG", 2020),
        ("ARG", 2022),
        ("BRA", 2020),
        ("BRA", 2022),
    ]

    assert list(
        zip(result["countryiso3code"], result["date"])
    ) == expected


def test_create_future_risk_target_uses_next_year_risk():
    df = pd.DataFrame(
        {
            "country": ["BRA", "BRA", "BRA"],
            "date": [2020, 2021, 2022],
            "country_risk_index": [50.0, 60.0, 70.0],
        }
    )

    result = create_future_risk_target(df)

    assert result["future_country_risk"].iloc[0] == 60.0
    assert result["future_country_risk"].iloc[1] == 70.0
    assert pd.isna(result["future_country_risk"].iloc[2])


def test_create_future_risk_target_is_calculated_per_country():
    df = pd.DataFrame(
        {
            "country": ["BRA", "ARG", "BRA", "ARG"],
            "date": [2020, 2020, 2021, 2021],
            "country_risk_index": [50.0, 30.0, 60.0, 40.0],
        }
    )

    result = create_future_risk_target(df)

    brazil = result[result["country"] == "BRA"]
    argentina = result[result["country"] == "ARG"]

    assert brazil["future_country_risk"].iloc[0] == 60.0
    assert pd.isna(brazil["future_country_risk"].iloc[1])

    assert argentina["future_country_risk"].iloc[0] == 40.0
    assert pd.isna(argentina["future_country_risk"].iloc[1])


def test_create_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "BRA", "BRA"],
            "date": [2020, 2021, 2022],
            "gdp_per_capita": [10000.0, 11000.0, 12100.0],
            "population": [100.0, 110.0, 121.0],
            "gdp_growth": [2.0, 3.0, 4.0],
            "inflation": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 71.0, 72.0],
        }
    )

    result = create_features(df)

    expected_columns = [
        "gdp_per_capita_growth",
        "population_growth",
        "gdp_growth_next_year",
        "gdp_lag1",
        "inflation_lag1",
        "life_expectancy_lag1",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_create_features_calculates_growth():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "BRA"],
            "date": [2020, 2021],
            "gdp_per_capita": [10000.0, 11000.0],
            "population": [100.0, 110.0],
            "gdp_growth": [2.0, 3.0],
            "inflation": [5.0, 6.0],
            "life_expectancy": [70.0, 71.0],
        }
    )

    result = create_features(df)

    assert pd.isna(result["gdp_per_capita_growth"].iloc[0])
    assert result["gdp_per_capita_growth"].iloc[1] == pytest.approx(10.0)

    assert pd.isna(result["population_growth"].iloc[0])
    assert result["population_growth"].iloc[1] == pytest.approx(10.0)


def test_create_features_creates_lag_features():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "BRA"],
            "date": [2020, 2021],
            "gdp_per_capita": [10000.0, 11000.0],
            "population": [100.0, 110.0],
            "gdp_growth": [2.0, 3.0],
            "inflation": [5.0, 6.0],
            "life_expectancy": [70.0, 71.0],
        }
    )

    result = create_features(df)

    assert pd.isna(result["gdp_lag1"].iloc[0])
    assert result["gdp_lag1"].iloc[1] == 10000.0

    assert pd.isna(result["inflation_lag1"].iloc[0])
    assert result["inflation_lag1"].iloc[1] == 5.0

    assert pd.isna(result["life_expectancy_lag1"].iloc[0])
    assert result["life_expectancy_lag1"].iloc[1] == 70.0


def test_create_features_calculates_next_year_gdp_growth():
    df = pd.DataFrame(
        {
            "countryiso3code": ["BRA", "BRA", "BRA"],
            "date": [2020, 2021, 2022],
            "gdp_per_capita": [10000.0, 11000.0, 12100.0],
            "population": [100.0, 110.0, 121.0],
            "gdp_growth": [2.0, 3.0, 4.0],
            "inflation": [5.0, 6.0, 7.0],
            "life_expectancy": [70.0, 71.0, 72.0],
        }
    )

    result = create_features(df)

    assert result["gdp_growth_next_year"].iloc[0] == 3.0
    assert result["gdp_growth_next_year"].iloc[1] == 4.0
    assert pd.isna(result["gdp_growth_next_year"].iloc[2])


def test_imputer_fills_missing_macroeconomic_values():
    df = pd.DataFrame(
        {
            "country": ["BRA", "BRA", "BRA"],
            "date": [2020, 2021, 2022],
            "gdp_growth": [2.0, None, 4.0],
        }
    )

    result = CountryDataImputer().transform(df)

    assert result["gdp_growth"].isna().sum() == 0
    assert result["gdp_growth"].iloc[1] == pytest.approx(3.0)


def test_imputer_does_not_fill_missing_governance_values():
    df = pd.DataFrame(
        {
            "country": ["BRA", "BRA", "BRA"],
            "date": [2020, 2021, 2022],
            "gdp_growth": [2.0, None, 4.0],
            "voice_accountability": [0.5, None, 0.7],
        }
    )

    result = CountryDataImputer().transform(df)

    assert result["gdp_growth"].isna().sum() == 0
    assert result["voice_accountability"].isna().sum() == 1

def test_imputer_interpolates_within_each_country():
    df = pd.DataFrame(
        {
            "country": ["BRA", "BRA", "ARG", "ARG"],
            "date": [2020, 2021, 2020, 2021],
            "gdp_growth": [2.0, None, 5.0, None],
        }
    )

    result = CountryDataImputer().transform(df)

    brazil = result[result["country"] == "BRA"]
    argentina = result[result["country"] == "ARG"]

    assert brazil["gdp_growth"].isna().sum() == 0
    assert argentina["gdp_growth"].isna().sum() == 0