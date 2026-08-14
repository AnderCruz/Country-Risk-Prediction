import numpy as np
import pandas as pd
import pytest

from monitoring.drift.calculate_drift import (
    FEATURES,
    calculate_distribution_drift,
    calculate_prediction_drift,
    calculate_psi,
    classify_psi,
    calculate_data_drift,
    determine_overall_status,
)


def test_identical_distributions_have_zero_psi():
    reference = pd.Series(np.arange(1, 101))
    production = reference.copy()

    psi = calculate_psi(reference, production)

    assert psi == pytest.approx(0.0)


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "STABLE"
    assert classify_psi(0.10) == "WARNING"
    assert classify_psi(0.19) == "WARNING"
    assert classify_psi(0.20) == "SIGNIFICANT_DRIFT"
    assert classify_psi(1.0) == "SIGNIFICANT_DRIFT"


def test_classify_nan_as_insufficient_data():
    assert classify_psi(float("nan")) == "INSUFFICIENT_DATA"


def test_distribution_drift_identical_data_is_stable():
    reference = pd.Series(np.arange(1, 101))
    production = reference.copy()

    result = calculate_distribution_drift(
        reference,
        production,
    )

    assert result["status"] == "STABLE"
    assert result["psi"] == pytest.approx(0.0)
    assert result["reference_count"] == 100
    assert result["production_count"] == 100
    assert result["ks_statistic"] == pytest.approx(0.0)


def test_distribution_drift_detects_significant_shift():
    reference = pd.Series(np.arange(1, 101))
    production = pd.Series(np.arange(101, 201))

    result = calculate_distribution_drift(
        reference,
        production,
    )

    assert result["status"] == "SIGNIFICANT_DRIFT"
    assert result["psi"] >= 0.20
    assert result["ks_statistic"] > 0.5


def test_distribution_drift_handles_empty_data():
    reference = pd.Series([1, 2, 3])
    production = pd.Series([np.nan, np.nan])

    result = calculate_distribution_drift(
        reference,
        production,
    )

    assert result["status"] == "INSUFFICIENT_DATA"
    assert np.isnan(result["psi"])
    assert result["reference_count"] == 3
    assert result["production_count"] == 0


def test_calculate_data_drift_returns_all_features():
    reference = pd.DataFrame(
        {
            feature: np.arange(1, 101)
            for feature in FEATURES
        }
    )

    production = reference.copy()

    result = calculate_data_drift(
        reference,
        production,
    )

    assert len(result) == len(FEATURES)
    assert list(result["feature"]) == FEATURES
    assert (result["status"] == "STABLE").all()
    assert (result["production_count"] == 100).all()


def test_prediction_drift_identical_predictions_is_stable():
    predictions = pd.DataFrame(
        {
            "prediction": np.arange(1, 101),
        }
    )

    production = pd.DataFrame(
        {
            "prediction": np.arange(1, 101),
        }
    )

    result = calculate_prediction_drift(
        predictions,
        production,
    )

    assert result["status"] == "STABLE"
    assert result["psi"] == pytest.approx(0.0)
    assert result["reference_count"] == 100
    assert result["production_count"] == 100


def test_prediction_drift_requires_prediction_column():
    reference = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    production = pd.DataFrame(
        {
            "prediction": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="prediction",
    ):
        calculate_prediction_drift(
            reference,
            production,
        )


def test_overall_status_stable():
    data_drift = pd.DataFrame(
        {
            "status": [
                "STABLE",
                "STABLE",
                "STABLE",
            ]
        }
    )

    prediction_drift = {
        "status": "STABLE",
    }

    result = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    assert result == "STABLE"


def test_overall_status_warning():
    data_drift = pd.DataFrame(
        {
            "status": [
                "STABLE",
                "WARNING",
                "STABLE",
            ]
        }
    )

    prediction_drift = {
        "status": "STABLE",
    }

    result = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    assert result == "WARNING"


def test_overall_status_critical_from_data_drift():
    data_drift = pd.DataFrame(
        {
            "status": [
                "STABLE",
                "SIGNIFICANT_DRIFT",
                "STABLE",
            ]
        }
    )

    prediction_drift = {
        "status": "STABLE",
    }

    result = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    assert result == "CRITICAL"


def test_overall_status_critical_from_prediction_drift():
    data_drift = pd.DataFrame(
        {
            "status": [
                "STABLE",
                "STABLE",
                "STABLE",
            ]
        }
    )

    prediction_drift = {
        "status": "SIGNIFICANT_DRIFT",
    }

    result = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    assert result == "CRITICAL"
