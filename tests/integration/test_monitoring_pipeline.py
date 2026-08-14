import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from monitoring.run_monitoring import (
    calculate_monitoring,
    publish_cloudwatch_metric,
    publish_report_to_s3,
    status_to_metric,
    validate_datasets,
)


REFERENCE_PATH = Path(
    "monitoring/drift/reference/reference_dataset.csv"
)

REFERENCE_PREDICTIONS_PATH = Path(
    "monitoring/drift/reference/reference_predictions.csv"
)


FEATURES = [
    "gdp_per_capita",
    "inflation",
    "life_expectancy",
    "population",
    "population_growth",
    "unemployment",
    "exports",
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",
    "economic_risk",
    "governance_risk",
]


def load_reference_data():

    reference = pd.read_csv(
        REFERENCE_PATH
    )

    reference_predictions = pd.read_csv(
        REFERENCE_PREDICTIONS_PATH
    )

    return (
        reference,
        reference_predictions,
    )


def test_monitoring_pipeline_stable():

    reference, reference_predictions = (
        load_reference_data()
    )

    production = reference.sample(
        n=300,
        replace=True,
        random_state=42,
    ).reset_index(
        drop=True
    )

    production["prediction"] = (
        reference_predictions
        .sample(
            n=300,
            replace=True,
            random_state=42,
        )["prediction"]
        .to_numpy()
    )

    validate_datasets(
        reference,
        production,
        reference_predictions,
    )

    (
        data_drift,
        prediction_drift,
        overall_status,
    ) = calculate_monitoring(
        reference,
        production,
        reference_predictions,
    )

    assert len(data_drift) == len(FEATURES)

    assert overall_status in {
        "STABLE",
        "WARNING",
    }

    assert prediction_drift["status"] in {
        "STABLE",
        "WARNING",
    }


def test_status_to_metric_mapping():

    assert status_to_metric("STABLE") == 0
    assert status_to_metric("WARNING") == 1
    assert status_to_metric("CRITICAL") == 2


def test_status_to_metric_rejects_invalid_status():

    try:
        status_to_metric("INVALID")
    except ValueError as exc:
        assert "Unknown monitoring status" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError"
        )


@patch("monitoring.run_monitoring.boto3.client")
def test_cloudwatch_metric_publish(
    mock_boto_client,
):

    mock_cloudwatch = MagicMock()

    mock_boto_client.return_value = (
        mock_cloudwatch
    )

    publish_cloudwatch_metric(
        status="CRITICAL",
        endpoint="country-risk-prediction-v7-v3",
        region="us-east-1",
    )

    mock_boto_client.assert_called_once_with(
        "cloudwatch",
        region_name="us-east-1",
    )

    mock_cloudwatch.put_metric_data.assert_called_once()

    call = (
        mock_cloudwatch
        .put_metric_data
        .call_args
    )

    kwargs = call.kwargs

    assert kwargs["Namespace"] == (
        "CountryRisk/ModelMonitoring"
    )

    metric = kwargs["MetricData"][0]

    assert metric["MetricName"] == (
        "MonitoringStatus"
    )

    assert metric["Value"] == 2

    dimensions = {
        item["Name"]: item["Value"]
        for item in metric["Dimensions"]
    }

    assert dimensions["Endpoint"] == (
        "country-risk-prediction-v7-v3"
    )

    assert dimensions["Environment"] == (
        "production"
    )


@patch("monitoring.run_monitoring.boto3.client")
def test_monitoring_report_is_published_to_s3(
    mock_boto_client,
):

    mock_s3 = MagicMock()

    mock_boto_client.return_value = (
        mock_s3
    )

    report = {
        "monitoring": {
            "overall_status": "CRITICAL",
            "alert_triggered": True,
        }
    }

    key = publish_report_to_s3(
        report=report,
        bucket="country-risk-prediction-monitoring-2026",
        region="us-east-1",
        output_prefix="monitoring/reports",
    )

    assert key.startswith(
        "monitoring/reports/"
    )

    assert key.endswith(
        ".json"
    )

    mock_boto_client.assert_called_once_with(
        "s3",
        region_name="us-east-1",
    )

    mock_s3.put_object.assert_called_once()

    kwargs = (
        mock_s3
        .put_object
        .call_args
        .kwargs
    )

    assert kwargs["Bucket"] == (
        "country-risk-prediction-monitoring-2026"
    )

    assert kwargs["ContentType"] == (
        "application/json"
    )

    uploaded_report = json.loads(
        kwargs["Body"].decode("utf-8")
    )

    assert (
        uploaded_report["monitoring"]
        ["overall_status"]
        == "CRITICAL"
    )

    assert (
        uploaded_report["monitoring"]
        ["alert_triggered"]
        is True
    )
