import pytest

from scripts.inference import invoke_endpoint


EXPECTED_PREDICTION = 0.2815675834378577
TOLERANCE = 1e-10


@pytest.mark.integration
def test_sagemaker_endpoint_returns_expected_prediction():
    result = invoke_endpoint()

    assert "predictions" in result
    assert len(result["predictions"]) == 1

    prediction = result["predictions"][0]

    assert prediction == pytest.approx(
        EXPECTED_PREDICTION,
        abs=TOLERANCE,
    )


@pytest.mark.integration
def test_sagemaker_endpoint_supports_batch_predictions():
    payload = {
        "dataframe_records": [
            {
                "gdp_per_capita": 15000.0,
                "inflation": 3.0,
                "life_expectancy": 75.0,
                "population": 50000000.0,
                "population_growth": 1.0,
                "unemployment": 6.0,
                "exports": 25.0,
                "gdp_lag1": 14500.0,
                "inflation_lag1": 3.2,
                "life_expectancy_lag1": 74.8,
                "economic_risk": 0.30,
                "governance_risk": 0.25,
            },
            {
                "gdp_per_capita": 30000.0,
                "inflation": 2.0,
                "life_expectancy": 82.0,
                "population": 10000000.0,
                "population_growth": 0.5,
                "unemployment": 4.0,
                "exports": 40.0,
                "gdp_lag1": 29000.0,
                "inflation_lag1": 2.1,
                "life_expectancy_lag1": 81.8,
                "economic_risk": 0.15,
                "governance_risk": 0.10,
            },
            {
                "gdp_per_capita": 5000.0,
                "inflation": 8.0,
                "life_expectancy": 65.0,
                "population": 80000000.0,
                "population_growth": 2.0,
                "unemployment": 12.0,
                "exports": 15.0,
                "gdp_lag1": 4800.0,
                "inflation_lag1": 7.5,
                "life_expectancy_lag1": 64.5,
                "economic_risk": 0.65,
                "governance_risk": 0.70,
            },
        ]
    }

    result = invoke_endpoint(payload=payload)

    predictions = result["predictions"]

    assert len(predictions) == 3

    assert all(
        isinstance(prediction, (int, float))
        for prediction in predictions
    )